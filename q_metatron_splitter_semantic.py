"""
core/security/q_metatron_splitter_semantic.py — MetatronSplitterSemantic

Variante experimental do MetatronSplitter original (q_metatron_splitter.py).

DIFERENÇAS:
  Original (MetatronSplitter):
    - Divide tokens em round-robin (word-level)
    - Rápido, zero custo extra
    - Cada provider recebe fragmentos incoerentes → respostas fracas

  Esta variante ->(MetatronSplitterSemantic):
    - Usa DeepSeek para decompor a query em N sub-perguntas completas
    - Cada sub-pergunta é autónoma e compreensível
    - Providers respondem melhor → qualidade de resposta superior
    - Custo: +1 call DeepSeek barato (~$0.0001 por decomposição)
    - Latência: +500-800ms para a decomposição

QUANDO USAR:
  - METATRON_MODE=semantic no .env
  - Ou instanciar directamente: MetatronSplitterSemantic(...)

O MetatronSplitter original continua como padrão (rápido, sem custo extra).

Arquitectura:
  msg → [DeepSeek Decompose] → sub_q_A + sub_q_B + sub_q_C
          ↓ async paralelo
        resp_A (DeepSeek) + resp_B (OpenAI) + resp_C (Anthropic)
          ↓
        [DeepSeek Assembler] → resposta_final
"""

import asyncio
import hashlib
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from core.security.q_metatron import get_metatron


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticFragment:
    """Uma sub-pergunta semântica destinada a um provider."""
    index: int
    provider: str
    sub_question: str       # sub-pergunta completa (compreensível)
    obfuscated_text: str    # sub-pergunta ofuscada (enviada ao provider)
    decryption_key: str     # chave HMAC desta sessão


@dataclass
class SemanticSplitResult:
    """Resultado de uma sessão de split semântico."""
    session_id: str
    original_message: str
    sub_questions: List[str]         # perguntas antes de enviar
    fragments: List[SemanticFragment]
    n_providers: int
    decompose_time_ms: float         # tempo da decomposição DeepSeek
    split_time_ms: float
    responses: Dict[str, str] = field(default_factory=dict)
    provider_times_ms: Dict[str, float] = field(default_factory=dict) # tempo de cada provider
    parallel_phase_ms: float = 0.0   # tempo da execução de query paralela
    assembled: str = ""
    assembly_time_ms: float = 0.0
    total_cost_note: str = "1 decompose + N parallel + 1 assemble (all DeepSeek-class)"


# ─────────────────────────────────────────────────────────────────────────────
# MetatronSplitterSemantic
# ─────────────────────────────────────────────────────────────────────────────

class MetatronSplitterSemantic:
    """
    Variante semântica do MetatronSplitter.

    Uso:
        splitter = MetatronSplitterSemantic(
            session_key="snap_chain_key",
            providers=["deepseek", "openai", "anthropic"],
            decompose_client=deepseek_client,   # AsyncOpenAI compatível
        )
        result = await splitter.split_and_query(message, llm_clients=clients)
        print(result.assembled)
    """

    MIN_WORDS_FOR_SPLIT = 6  # mensagens curtas: sem decomposição

    def __init__(
        self,
        session_key: str,
        providers: List[str] = None,
        decompose_client=None,   # AsyncOpenAI ou compatível — para decomposição
        language: str = "pt",
    ):
        self.session_key = session_key.encode() if isinstance(session_key, str) else session_key
        self.providers = providers or ["deepseek", "openai", "anthropic"]
        self.decompose_client = decompose_client
        self.language = language
        self.metatron = get_metatron(
            session_key if isinstance(session_key, str) else session_key.decode()
        )
        self.session_id = hashlib.sha256(self.session_key).hexdigest()[:16]

    # ── Decomposição semântica ─────────────────────────────────────────────────

    async def _decompose(self, message: str, n: int) -> List[str]:
        """
        Usa DeepSeek para decompor a message em n sub-perguntas semânticas.

        Prompt minimalista — barato e rápido.
        Fallback: divide por frases ou divide em partes iguais.
        """
        if self.decompose_client is None:
            return self._fallback_decompose(message, n)

        lang_hint = "em português" if self.language.startswith("pt") else "in English"
        prompt = (
            f"Extrai exactamente {n} conceitos-chave desta pergunta {lang_hint}. "
            f"ESTRANGULADOR: Usa no MÁXIMO 5 a 7 palavras por conceito. "
            f"Responde APENAS com um JSON no formato: "
            f'[{{"q": "conceito curto 1"}}, {{"q": "conceito curto 2"}}]'
            f"\n\nPergunta: {message}"
        )

        try:
            resp = await self.decompose_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,  # Reduzido porque esperamos pouco texto (latência menor)
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()

            # Extrair JSON da resposta (pode vir com markdown)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)
            sub_qs = [item["q"] for item in parsed if "q" in item]

            if len(sub_qs) == n:
                return sub_qs
            # Ajustar se o modelo devolveu diferente número
            while len(sub_qs) < n:
                sub_qs.append(message)  # replica original como fallback
            return sub_qs[:n]

        except Exception as e:
            # Fallback silencioso — não bloquear o fluxo principal
            return self._fallback_decompose(message, n)

    def _fallback_decompose(self, message: str, n: int) -> List[str]:
        """
        Decomposição sem LLM — divide por frases ou por partes.
        Menos boa que a semântica mas melhor que word round-robin.
        """
        import re
        # Tentar dividir em frases
        sentences = re.split(r'(?<=[.!?])\s+', message.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= n:
            # Distribuir frases pelos providers
            result = []
            chunk_size = max(1, len(sentences) // n)
            for i in range(n):
                start = i * chunk_size
                end = start + chunk_size if i < n - 1 else len(sentences)
                result.append(" ".join(sentences[start:end]))
            return result

        # Fallback final: todos recebem a mensagem completa (sem privacidade,
        # mas permite resposta de qualidade se a mensagem for muito curta)
        return [message] * n

    # ── Split e query ──────────────────────────────────────────────────────────

    async def split_and_query(
        self,
        message: str,
        context_hint: str = "",
        llm_clients: Dict[str, any] = None,
    ) -> SemanticSplitResult:
        """
        Pipeline completo:
          1. Decomposição semântica (DeepSeek)
          2. Ofuscação HMAC de cada sub-pergunta
          3. Envio paralelo para N providers
          4. Assemblagem das respostas (DeepSeek)
        """
        t0 = time.perf_counter()
        n = len(self.providers)
        tokens = message.split()

        # Mensagens curtas: sem split, vai para primeiro provider
        if len(tokens) < self.MIN_WORDS_FOR_SPLIT:
            obf, key = self.metatron.encode(message, obfuscate_level=1)
            frag = SemanticFragment(
                index=0,
                provider=self.providers[0],
                sub_question=message,
                obfuscated_text=obf,
                decryption_key=key,
            )
            result = SemanticSplitResult(
                session_id=self.session_id,
                original_message=message,
                sub_questions=[message],
                fragments=[frag],
                n_providers=1,
                decompose_time_ms=0.0,
                split_time_ms=(time.perf_counter() - t0) * 1000,
            )
            if llm_clients and self.providers[0] in llm_clients:
                resp = await self._query_provider(
                    llm_clients[self.providers[0]], message, context_hint, self.providers[0]
                )
                result.responses = {self.providers[0]: resp}
                result.assembled = resp
            return result

        # 1. Decomposição semântica
        t_decompose = time.perf_counter()
        sub_questions = await self._decompose(message, n)
        decompose_ms = (time.perf_counter() - t_decompose) * 1000

        # 2. Criar fragmentos com ofuscação HMAC
        fragments = []
        for idx, (provider, sub_q) in enumerate(zip(self.providers, sub_questions)):
            obf, key = self.metatron.encode(sub_q, obfuscate_level=1)
            fragments.append(SemanticFragment(
                index=idx,
                provider=provider,
                sub_question=sub_q,
                obfuscated_text=obf,
                decryption_key=key,
            ))

        split_ms = (time.perf_counter() - t0) * 1000

        result = SemanticSplitResult(
            session_id=self.session_id,
            original_message=message,
            sub_questions=sub_questions,
            fragments=fragments,
            n_providers=n,
            decompose_time_ms=decompose_ms,
            split_time_ms=split_ms,
        )

        if llm_clients is None:
            return result  # só inspecção do split

        # 3. Query paralela
        t_parallel_start = time.perf_counter()
        
        async def query_with_timer(c, f, ctx):
            ts = time.perf_counter()
            resp = await self._query_provider(c, f.obfuscated_text, ctx, f.provider)
            ms = (time.perf_counter() - ts) * 1000
            return f.provider, resp, ms

        tasks = []
        for frag in fragments:
            client = llm_clients.get(frag.provider) or llm_clients.get("deepseek")
            if client is None:
                continue
            tasks.append(query_with_timer(client, frag, context_hint))

        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = {}
        provider_times = {}
        for res in task_results:
            if isinstance(res, Exception):
                # Se falhar, res é Exception
                pass # ou podes mapear provider, mas o wrapper não diz se falhou. O nosso query_provider já trata as excepções por dentro.
            else:
                provider, resp, ms = res
                responses[provider] = resp
                provider_times[provider] = ms

        result.responses = responses
        result.provider_times_ms = provider_times
        result.parallel_phase_ms = (time.perf_counter() - t_parallel_start) * 1000

        # 4. Assemblagem com DeepSeek
        t_assemble = time.perf_counter()
        result.assembled = await self._assemble(
            original=message,
            responses=responses,
            context_hint=context_hint,
            assemble_client=self.decompose_client or (llm_clients.get("deepseek")),
        )
        result.assembly_time_ms = (time.perf_counter() - t_assemble) * 1000

        return result

    async def _query_provider(
        self,
        client,
        sub_question: str,
        context_hint: str,
        provider: str,
    ) -> str:
        """Envia uma sub-pergunta a um provider e retorna a resposta."""
        provider_lower = provider.lower()
        if "openai" in provider_lower:
            model = "gpt-4o-mini"
        elif "anthropic" in provider_lower:
            model = "claude-3-haiku-20240307"
        elif "gemini" in provider_lower:
            model = "gemini-1.5-flash"
        else:
            model = "deepseek-chat"

        system = (
            "Responde de forma directa e concisa (máx 3 frases). "
            f"{f'Contexto: {context_hint}' if context_hint else ''}"
        )

        try:
            if "anthropic" in provider_lower:
                resp = await client.messages.create(
                    model=model,
                    system=system,
                    messages=[{"role": "user", "content": sub_question}],
                    max_tokens=250,
                )
                return resp.content[0].text
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": sub_question},
                    ],
                    max_tokens=250,
                )
                return resp.choices[0].message.content
        except Exception as e:
            return f"[ERRO {provider}: {e}]"

    async def _assemble(
        self,
        original: str,
        responses: Dict[str, str],
        context_hint: str,
        assemble_client=None,
    ) -> str:
        """
        Assembla respostas parciais numa resposta coerente.
        Usa DeepSeek se disponível, caso contrário concatena.
        """
        if not responses:
            return "[MetatronSemântico: sem respostas]"

        parts = "\n".join(
            f"[{provider.upper()}]: {resp}"
            for provider, resp in responses.items()
            if not resp.startswith("[ERRO")
        )

        if assemble_client is None:
            # Fallback: concatenar
            return " | ".join(
                r for r in responses.values() if not r.startswith("[ERRO")
            )

        prompt = (
            f"Pergunta original: {original}\n\n"
            f"Perspectivas parciais de diferentes modelos:\n{parts}\n\n"
            "Sintetiza numa resposta directa, coerente e completa (máx 5 frases)."
        )

        try:
            resp = await assemble_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5,
            )
            return resp.choices[0].message.content
        except Exception as e:
            # Fallback silencioso
            return " | ".join(r for r in responses.values() if not r.startswith("[ERRO"))
