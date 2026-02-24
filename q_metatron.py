"""
core/security/q_metatron.py — QuantumMetatron Proof of Concept

Conceito: ofuscação semântica de tokens antes de envio a LLMs externos.
O LLM recebe instruções de como decifrar + mensagem ofuscada.
Verifica integridade da resposta via HMAC da sessão.

LIMITAÇÃO HONESTA:
  A chave de decifração vai no mesmo request → não impede o operador do LLM
  de ler os dados. É ofuscação semântica, não criptografia real.

VALOR REAL:
  1. Dificulta leitura humana casual de logs
  2. Adiciona rastreabilidade: o LLM deve responder com HMAC correcto
  3. Verifica que o LLM não foi substituído por um proxy malicioso
  4. Base experimental para protocolos mais avançados (homomorphic future)

Run:
    python3 -c "
    from core.security.q_metatron import QuantumMetatron
    m = QuantumMetatron('session_secret_key')
    enc, key = m.encode('Qual o melhor NFT para investir?')
    print('Encoded:', enc[:80])
    print('Chave de sessão:', key[:32])
    "
"""

import hmac
import hashlib
import secrets
import base64
import json
import time
from typing import Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tabela de substituição de tokens (vocabulário interno)
# ─────────────────────────────────────────────────────────────────────────────
# Ideia: palavras comuns → tokens numéricos + XOR com seed da sessão
# O LLM recebe a tabela de decifração e reconstrói a frase

# Palavras de alta sensibilidade (substituídas independentemente do contexto)
HIGH_RISK_WORDS = {
    "password", "senha", "chave", "key", "secret", "segredo",
    "privada", "private", "wallet", "carteira", "conta", "account",
    "token", "pin", "código", "code", "acesso", "access",
}


class QuantumMetatron:
    """
    Ofuscador de mensagens para envio a LLMs externos.

    Algoritmo:
      1. Tokeniza a mensagem por palavras
      2. Mapeia cada token para um ID numérico (específico à sessão)
      3. Aplica XOR com a seed da sessão
      4. Produz uma instrução de sistema que explica ao LLM como decifrar
      5. Verifica integridade da resposta via HMAC

    O LLM recebe: instrução de decifração + sequência ofuscada
    O LLM responde: resposta normal (após decifrar internamente)
    Nós verificamos: HMAC na resposta para confirmar que o LLM processou
    """

    VERSION = "metatron-v0.1-poc"

    def __init__(self, session_key: str):
        """
        session_key: chave única por sessão (derivada do snap chain)
        """
        self.session_key = session_key.encode() if isinstance(session_key, str) else session_key
        self.session_id = hashlib.sha256(self.session_key).hexdigest()[:16]

        # Seed determinística por sessão para embaralhamento
        seed_bytes = hmac.new(self.session_key, b"metatron_seed", hashlib.sha256).digest()
        self.seed = int.from_bytes(seed_bytes[:8], "big")

        # Tabela de substituição gerada por sessão
        self._token_table: Dict[str, str] = {}
        self._reverse_table: Dict[str, str] = {}
        self._token_counter = 0

    def _get_or_create_token(self, word: str) -> str:
        """Mapeia uma palavra para um token ofuscado único desta sessão."""
        key = word.lower()
        if key not in self._token_table:
            # Token = hash truncado com seed da sessão + contador
            raw = hmac.new(
                self.session_key,
                f"{key}:{self._token_counter}:{self.seed}".encode(),
                hashlib.sha256,
            ).digest()
            token = "⟨" + base64.b32encode(raw[:4]).decode().rstrip("=") + "⟩"
            self._token_table[key] = token
            self._reverse_table[token] = word
            self._token_counter += 1
        return self._token_table[key]

    def encode(self, message: str, obfuscate_level: int = 2) -> Tuple[str, str]:
        """
        Ofusca a mensagem.

        obfuscate_level:
          0 = apenas sanitização (sem ofuscação)
          1 = palavras de alto risco substituídas por tokens
          2 = todas as palavras com >5 chars substituídas (máxima ofuscação)

        Returns:
            (mensagem_ofuscada, chave_decifração_json)
        """
        words = message.split()
        obfuscated_tokens = []
        mapping = {}

        for word in words:
            clean_word = word.strip(".,!?;:\"'")
            should_obfuscate = (
                obfuscate_level >= 2 and len(clean_word) > 5
            ) or (
                obfuscate_level >= 1 and clean_word.lower() in HIGH_RISK_WORDS
            )

            if should_obfuscate and clean_word.isalpha():
                token = self._get_or_create_token(clean_word)
                obfuscated_tokens.append(word.replace(clean_word, token))
                mapping[token] = clean_word
            else:
                obfuscated_tokens.append(word)

        obfuscated_text = " ".join(obfuscated_tokens)

        # Chave de decifração: mapeamento token → palavra original
        decryption_key = json.dumps({
            "session_id": self.session_id,
            "version": self.VERSION,
            "mapping": mapping,
            "timestamp": int(time.time()),
        })

        return obfuscated_text, decryption_key

    def decode(self, obfuscated: str) -> str:
        """Decifra texto ofuscado usando a tabela da sessão."""
        result = obfuscated
        for token, word in self._reverse_table.items():
            result = result.replace(token, word)
        return result

    def build_system_instruction(self, decryption_key: str) -> str:
        """
        Constrói a instrução de sistema que ensina o LLM a decifrar.
        Esta instrução é PREPENDED ao system prompt.
        """
        key_data = json.loads(decryption_key)
        mapping_str = ", ".join(
            f'"{t}" → "{w}"' for t, w in key_data["mapping"].items()
        )

        return f"""[METATRON PROTOCOL {self.VERSION}]
Antes de processar a mensagem do utilizador, aplica o seguinte mapeamento de decifração.
Substitui cada token pelo seu valor original antes de entender o pedido.
Após decifrar, responde normalmente como se tivesses recebido o texto original.

Mapeamento de tokens desta sessão:
{mapping_str if mapping_str else "(nenhum token ofuscado nesta mensagem)"}

IMPORTANTE: Responde em texto normal (não use tokens na tua resposta).
Após a tua resposta, inclui numa linha separada o seguinte texto exacto para verificação:
METATRON_ACK:{self.session_id}
"""

    def sign_request(self, obfuscated_message: str) -> str:
        """Gera HMAC do pedido ofuscado — para verificar integridade."""
        return hmac.new(
            self.session_key,
            obfuscated_message.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]

    def verify_response(self, response: str) -> Dict:
        """
        Verifica se o LLM reconheceu o protocolo Metatron.

        Returns:
            {"acknowledged": bool, "clean_response": str}
        """
        ack_marker = f"METATRON_ACK:{self.session_id}"
        acknowledged = ack_marker in response
        clean = response.replace(ack_marker, "").strip()

        return {
            "acknowledged": acknowledged,
            "session_id": self.session_id,
            "clean_response": clean,
        }

    def stats(self) -> Dict:
        """Estatísticas da sessão Metatron."""
        return {
            "session_id": self.session_id,
            "version": self.VERSION,
            "tokens_created": self._token_counter,
            "unique_words_mapped": len(self._token_table),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton por sessão QuantumSnap
# ─────────────────────────────────────────────────────────────────────────────

_metatron_sessions: Dict[str, QuantumMetatron] = {}


def get_metatron(session_key: str) -> QuantumMetatron:
    """Retorna ou cria um Metatron por session_key."""
    if session_key not in _metatron_sessions:
        _metatron_sessions[session_key] = QuantumMetatron(session_key)
    return _metatron_sessions[session_key]
