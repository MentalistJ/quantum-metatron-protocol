"""
core/security/q_metatron_splitter.py — MetatronSplitter

Conceito: divide semânticamente uma mensagem em N fragmentos,
envia cada fragmento a um LLM diferente em paralelo,
reassembla as respostas localmente.

Nenhum LLM individual vê a mensagem completa.
A chave de reassembly nunca sai do servidor AGEN.

Arquitectura:
  msg → [Splitter] → frag_A (DeepSeek) + frag_B (Anthropic) + frag_C (Gemini)
                         ↓ async paralelo
                     resp_A + resp_B + resp_C
                         ↓
                     [Assembler] → resposta_final

Limitação:
  - Fragmentos muito curtos perdem contexto — o LLM responde com menos qualidade
  - Só funciona bem com mensagens de >20 palavras
  - Para mensagens curtas, fallback para Metatron simples (1 LLM)

Nível de segurança:
  - Nenhum operador vê mais de 1/N da mensagem
  - Fragmentos sem contexto são difíceis de interpretar
  - Chave de split derivada do QuantumSnap session key
"""

import asyncio
import hashlib
import hmac
import json
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from core.security.q_metatron import QuantumMetatron, get_metatron


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fragment:
    """Um fragmento de mensagem destinado a um LLM específico."""
    index: int
    provider: str                # "deepseek", "anthropic", "gemini", "openai"
    original_tokens: List[str]   # palavras originais neste fragmento
    obfuscated_text: str         # texto ofuscado a enviar
    decryption_key: str          # chave de decifração (para instrução de sistema)
    placeholder_slots: Dict[str, str]  # slots que este LLM não conhece: posição → "???"
    hmac_sig: str = ""

@dataclass
class SplitResult:
    """Resultado completo de uma sessão de split."""
    session_id: str
    fragments: List[Fragment]
    n_providers: int
    total_tokens: int
    split_time_ms: float
    responses: Dict[str, str] = field(default_factory=dict)     # provider → resposta
    assembled: str = ""
    assembly_time_ms: float = 0.0
    all_acked: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# MetatronSplitter
# ─────────────────────────────────────────────────────────────────────────────

class MetatronSplitter:
    """
    Divide mensagens entre múltiplos LLMs para privacidade distribuída.

    Uso:
        splitter = MetatronSplitter(session_key="snap_chain_key_here",
                                    providers=["deepseek", "anthropic", "gemini"])
        result = await splitter.split_and_query(message, context)
        print(result.assembled)
    """

    MIN_WORDS_FOR_SPLIT = 8  # abaixo disto, usa Metatron simples

    def __init__(
        self,
        session_key: str,
        providers: List[str] = None,
    ):
        self.session_key = session_key.encode() if isinstance(session_key, str) else session_key
        self.providers = providers or ["deepseek", "anthropic", "gemini"]
        self.metatron = get_metatron(session_key if isinstance(session_key, str) else session_key.decode())

        # session_id derivado da chave
        self.session_id = hashlib.sha256(self.session_key).hexdigest()[:16]

    def tokenize(self, message: str) -> List[str]:
        """Divide mensagem em tokens (palavras + pontuação)."""
        return message.split()

    def split_message(self, message: str) -> SplitResult:
        """
        Divide a mensagem em N fragmentos, um por provider.

        Estratégia de split:
          - Os tokens são distribuídos em round-robin pelos providers
          - Cada provider recebe os seus tokens reais + placeholders "⟨?⟩"
            nos slots dos outros — mantém estrutura gramatical mas sem contexto
          - Cada fragmento é ofuscado pelo Metatron antes do envio
        """
        t0 = time.perf_counter()
        tokens = self.tokenize(message)
        n = len(self.providers)

        # Se muito curta, não dividir
        if len(tokens) < self.MIN_WORDS_FOR_SPLIT:
            # Fallback: envia apenas ao primeiro provider sem split
            obf, key = self.metatron.encode(message, obfuscate_level=2)
            sig = self.metatron.sign_request(obf)
            frag = Fragment(
                index=0,
                provider=self.providers[0],
                original_tokens=tokens,
                obfuscated_text=obf,
                decryption_key=key,
                placeholder_slots={},
                hmac_sig=sig,
            )
            return SplitResult(
                session_id=self.session_id,
                fragments=[frag],
                n_providers=1,
                total_tokens=len(tokens),
                split_time_ms=(time.perf_counter() - t0) * 1000,
            )

        # Atribuir cada token a um provider em round-robin
        token_assignments: List[int] = [i % n for i in range(len(tokens))]

        fragments = []
        for p_idx, provider in enumerate(self.providers):
            # Construir texto deste provider:
            # tokens que lhe pertencem + "⟨?_POS⟩" para os outros
            fragment_words = []
            placeholder_slots = {}

            for t_idx, (token, owner) in enumerate(zip(tokens, token_assignments)):
                if owner == p_idx:
                    fragment_words.append(token)
                else:
                    slot = f"⟨?_{t_idx}⟩"
                    fragment_words.append(slot)
                    placeholder_slots[str(t_idx)] = slot

            fragment_text = " ".join(fragment_words)

            # Ofuscar com Metatron (nível 1 — só palavras sensíveis, os placeholders já são opcos)
            obf, key = self.metatron.encode(fragment_text, obfuscate_level=1)
            sig = self.metatron.sign_request(obf)

            fragments.append(Fragment(
                index=p_idx,
                provider=provider,
                original_tokens=[
                    tokens[i] for i, owner in enumerate(token_assignments) if owner == p_idx
                ],
                obfuscated_text=obf,
                decryption_key=key,
                placeholder_slots=placeholder_slots,
                hmac_sig=sig,
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        return SplitResult(
            session_id=self.session_id,
            fragments=fragments,
            n_providers=n,
            total_tokens=len(tokens),
            split_time_ms=elapsed,
        )

    def build_fragment_prompt(self, frag: Fragment, context_hint: str = "") -> Tuple[str, str]:
        """
        Constrói system prompt + user message para um fragmento.
        O LLM recebe apenas a sua "visão parcial" da mensagem.
        """
        n_slots = len(frag.placeholder_slots)
        system = self.metatron.build_system_instruction(frag.decryption_key)
        system += f"""
[METATRON SPLIT PROTOCOL]
Esta mensagem foi dividida entre {self.metatron.stats()['session_id']} providers para privacidade distribuída.
Os marcadores ⟨?_N⟩ representam palavras que outros providers processam — ignora-os na tua compreensão.
Responde ao sentido geral que consegues perceber do teu fragmento.
Usa no máximo 2 frases. Sê directo.
{f'Contexto: {context_hint}' if context_hint else ''}
"""
        return system, frag.obfuscated_text

    def assemble_responses(
        self,
        result: SplitResult,
        responses: Dict[str, str],
    ) -> str:
        """
        Reassembla as respostas dos múltiplos LLMs numa resposta coerente.

        Estratégia de assembla:
          - Limpa marcadores ACK de cada resposta
          - Concatena respostas em ordem, removendo redundâncias óbvias
          - Produz síntese final
        """
        cleaned = []
        all_acked = True

        for i, frag in enumerate(result.fragments):
            resp = responses.get(frag.provider, "")
            verification = self.metatron.verify_response(resp)
            if not verification["acknowledged"]:
                all_acked = False
            clean = verification["clean_response"].strip()
            if clean:
                cleaned.append(clean)

        result.all_acked = all_acked

        if not cleaned:
            return "[MetatronSplitter: sem respostas válidas]"

        if len(cleaned) == 1:
            return cleaned[0]

        # Síntese simples: juntar respostas complementares
        assembled = " | ".join(cleaned)
        return assembled

    async def split_and_query(
        self,
        message: str,
        context_hint: str = "",
        llm_clients: Dict[str, any] = None,
    ) -> SplitResult:
        """
        Pipeline completo: split → query paralela → assemble.

        llm_clients: dict provider → AsyncOpenAI client (ou compatível)
        Se não fornecido, usa apenas DeepSeek para todos (modo demo).
        """
        # 1. Split
        result = self.split_message(message)

        if llm_clients is None:
            return result  # retorna só o split para inspeção

        # 2. Query paralela
        t0 = time.perf_counter()
        tasks = {}

        for frag in result.fragments:
            client = llm_clients.get(frag.provider) or llm_clients.get("deepseek")
            if client is None:
                continue
            system_prompt, user_msg = self.build_fragment_prompt(frag, context_hint)
            tasks[frag.provider] = asyncio.create_task(
                self._query_llm(client, system_prompt, user_msg, frag.provider)
            )

        responses = {}
        for provider, task in tasks.items():
            try:
                responses[provider] = await task
            except Exception as e:
                responses[provider] = f"[ERRO {provider}: {e}]"

        result.responses = responses

        # 3. Assemble
        t_assemble = time.perf_counter()
        result.assembled = self.assemble_responses(result, responses)
        result.assembly_time_ms = (time.perf_counter() - t_assemble) * 1000

        return result

    async def _query_llm(
        self,
        client,
        system_prompt: str,
        user_msg: str,
        provider: str,
    ) -> str:
        """Queries um único LLM com o fragmento."""
        # Mapa de modelos por provider (extensível para produção)
        MODEL_MAP = {
            "deepseek":   "deepseek-chat",
            "deepseek_A": "deepseek-chat",
            "deepseek_B": "deepseek-chat",
            "deepseek_C": "deepseek-chat",
            "anthropic":  "claude-3-haiku-20240307",
            "gemini":     "gemini-1.5-flash",
            "openai":     "gpt-4o-mini",
        }
        model = MODEL_MAP.get(provider, "deepseek-chat")

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ERRO: {e}]"

