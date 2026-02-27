#!/usr/bin/env python3
"""
test_metatron_dance.py — Teste completo: MetatronSplitterSemantic + QuantumDanceAssembler

Pipeline:
  mensagem → [MetatronSplitterSemantic] → N sub-perguntas (paralelo, cada provider vê só 1/N)
                                        → respostas parciais
                                        → [QuantumDanceAssembler] → resposta final rotativa

Pré-requisitos:
  pip install openai anthropic python-dotenv

Uso (dentro da pasta quantum-metatron-protocol):
  # Com .env em ../Agen/.env ou definir vars manualmente:
  python3 test_metatron_dance.py

  # 2 providers apenas:
  python3 test_metatron_dance.py --providers deepseek openai

  # Mensagem custom:
  python3 test_metatron_dance.py --msg "Como funciona o Metatron na prática?"

  # Com guard Ollama local:
  python3 test_metatron_dance.py --guard --guard-url http://localhost:11434/api/generate
"""

import asyncio
import json
import os
import sys
import time
import argparse

# ─── Garantir imports locais do repo ────────────────────────────────────────
# Adiciona o directório do script ao sys.path para encontrar os módulos do repo
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ─── Carregar .env automaticamente ──────────────────────────────────────────
def _load_env():
    """Procura .env em vários locais comuns."""
    candidates = [
        os.path.join(_HERE, ".env"),
        os.path.join(_HERE, "..", "Agen", ".env"),
        os.path.join(_HERE, "..", ".env"),
        os.path.expanduser("~/.agen.env"),
    ]
    try:
        from dotenv import load_dotenv
        for path in candidates:
            if os.path.exists(path):
                load_dotenv(path)
                print(f"  📄 .env carregado: {os.path.abspath(path)}")
                return
        print("  ⚠️  Nenhum .env encontrado — a usar variáveis de ambiente do shell.")
    except ImportError:
        print("  ⚠️  python-dotenv não instalado. Usa: pip install python-dotenv")

_load_env()

# ─── Verificar dependências ──────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import openai  # noqa
    except ImportError:
        missing.append("openai")
    try:
        import anthropic  # noqa
    except ImportError:
        missing.append("anthropic")
    if missing:
        print(f"\n  ❌ Dependências em falta: {', '.join(missing)}")
        print(f"     Instala com: pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()

# ─── Imports do protocolo ────────────────────────────────────────────────────
from q_metatron_splitter_semantic import MetatronSplitterSemantic
from q_quantum_dance import QuantumDanceAssembler

# ─── Configuração de providers ───────────────────────────────────────────────

PROVIDERS_CONFIG = {
    "deepseek": {
        "key_env":  "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model":    "deepseek-chat",
        "openai_compat": True,
    },
    "openai": {
        "key_env":  "OPENAI_API_KEY",
        "base_url": None,
        "model":    "gpt-4o-mini",
        "openai_compat": True,
    },
    "anthropic": {
        "key_env":  "ANTHROPIC_API_KEY",
        "base_url": None,
        "model":    "claude-3-haiku-20240307",
        "openai_compat": False,   # usa SDK nativo; splitter semântico trata isto
    },
}

DEFAULT_MESSAGE = (
    "Como posso proteger a minha privacidade ao usar modelos de linguagem na cloud? "
    "Existem técnicas de fragmentação ou ofuscação de queries que funcionem na prática?"
)

SEP = "─" * 72


def build_clients(providers: list) -> dict:
    """Cria clientes para cada provider. Retorna dict provider→client."""
    from openai import AsyncOpenAI
    import anthropic as anthropic_lib

    clients = {}
    missing_keys = []

    for p in providers:
        cfg = PROVIDERS_CONFIG.get(p)
        if not cfg:
            print(f"  ⚠️  Provider desconhecido ignorado: {p}")
            continue
        key = os.getenv(cfg["key_env"], "").strip()
        if not key:
            missing_keys.append(f"{p} (env: {cfg['key_env']})")
            continue

        if cfg["openai_compat"]:
            kwargs = {"api_key": key}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            clients[p] = AsyncOpenAI(**kwargs)
        else:
            # Anthropic — SDK nativo (suportado pelo q_metatron_splitter_semantic.py)
            clients[p] = anthropic_lib.AsyncAnthropic(api_key=key)

    if missing_keys:
        print(f"  ⚠️  Chaves em falta (providers ignorados): {', '.join(missing_keys)}")

    return clients


def print_banner(message: str, providers: list):
    print(f"\n{SEP}")
    print("  🔐 QUANTUM METATRON + DANCE — Teste End-to-End")
    print(f"  Providers: {', '.join(providers)}")
    print(f"  Data/hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    print(f"\n  📩 Mensagem original:\n  \"{message}\"\n")


def print_split_report(result):
    print(f"{SEP}")
    print("  📦 FASE 1 — MetatronSplitterSemantic")
    print(SEP)
    print(f"  ⏱️  Decomposição semântica : {result.decompose_time_ms:.0f}ms")
    print(f"  ⏱️  Fase paralela         : {result.parallel_phase_ms:.0f}ms")
    print(f"  ⏱️  Assemblagem interna    : {result.assembly_time_ms:.0f}ms")
    print(f"  📊 Sub-perguntas          : {len(result.sub_questions)}")
    print()

    for frag in result.fragments:
        resp  = result.responses.get(frag.provider, "[sem resposta]")
        t_ms  = result.provider_times_ms.get(frag.provider, 0)
        icon  = "❌" if str(resp).startswith("[ERRO") else "✅"
        print(f"  {icon} [{frag.provider.upper()}] ({t_ms:.0f}ms)")
        print(f"     Sub-pergunta enviada : \"{frag.sub_question}\"")
        print(f"     Fragmento ofuscado   : {frag.obfuscated_text[:90]}...")
        print(f"     Resposta recebida    : {str(resp)[:220]}")
        print()

    print(f"  📎 Assembla Metatron (antes do Dance):")
    print(f"     {result.assembled[:350]}")
    print()


def print_dance_report(dance_result, t_total_ms: float):
    print(f"{SEP}")
    print("  💃 FASE 2 — QuantumDanceAssembler (rotação cloud)")
    print(SEP)
    print(f"  🎯 Assembler escolhido  : {dance_result.assembler_provider or 'fallback concat'}")
    print(f"  🛡️  Guard Ollama         : {'✅ PASS_TOKEN' if dance_result.guard_passed else '❌ REWORK_TOKEN'}")
    print(f"  🔄 Fallback local       : {'sim' if dance_result.fallback_used else 'não'}")
    print(f"  ⏱️  Tempo QuantumDance   : {dance_result.elapsed_ms:.0f}ms")
    print(f"  ⏱️  TOTAL pipeline      : {t_total_ms:.0f}ms")
    print()
    print(f"  ✨ RESPOSTA FINAL (QuantumDance + assembler rotativo):")
    print(f"  {SEP}")
    for line in dance_result.assembled.split("\n"):
        print(f"  {line}")
    print(f"  {SEP}\n")


def print_privacy_report(message: str, split_result):
    print(f"{SEP}")
    print("  🔒 RELATÓRIO DE PRIVACIDADE")
    print(SEP)
    total_words = len(message.split())
    print(f"  Mensagem original: {total_words} palavras")
    print()
    for frag in split_result.fragments:
        words_seen = len(frag.sub_question.split())
        pct = words_seen / max(total_words, 1) * 100
        print(f"  [{frag.provider.upper()}] viu ≈ {pct:.0f}% da query")
        print(f"    Sub-pergunta: \"{frag.sub_question}\"")
    print()
    print("  ✅ Nenhum provider recebeu a mensagem original.")
    print("  ✅ QuantumDance recebeu apenas as respostas parciais.")
    print(f"{SEP}\n")


async def run_test(message: str, providers: list, guard: bool, guard_url: str):
    print_banner(message, providers)

    print("  🔧 A inicializar clientes...\n")
    clients = build_clients(providers)

    if len(clients) < 2:
        print("  ❌ São necessários pelo menos 2 providers com chaves válidas.")
        print("     Define as variáveis de ambiente ou acrescenta um .env.\n")
        sys.exit(1)

    active = list(clients.keys())
    print(f"  ✅ Activos: {', '.join(active)}\n")

    SESSION_KEY = f"metatron_dance_test_{int(time.time())}"
    t0 = time.perf_counter()

    # ── FASE 1: MetatronSplitterSemantic ────────────────────────────────────
    # DeepSeek preferido como decompositor (barato + rápido)
    decompose_client = clients.get("deepseek") or clients.get("openai")

    splitter = MetatronSplitterSemantic(
        session_key=SESSION_KEY,
        providers=active,
        decompose_client=decompose_client,
        language="pt",
    )

    print("  🚀 A executar MetatronSplitterSemantic (split + queries paralelas)...")
    split_result = await splitter.split_and_query(
        message=message,
        context_hint="Responde em português, máx 3 frases.",
        llm_clients=clients,
    )
    print_split_report(split_result)

    # ── FASE 2: QuantumDanceAssembler ───────────────────────────────────────
    # Apenas providers OpenAI-compat para o dance assembler
    dance_clients = {p: c for p, c in clients.items() if PROVIDERS_CONFIG[p]["openai_compat"]}

    dancer = QuantumDanceAssembler(
        session_key=SESSION_KEY,
        providers=list(dance_clients.keys()),
        non_repeat_window=1,
        guard_enabled=guard,
        guard_url=guard_url if guard else None,
    )

    print("  💃 A executar QuantumDanceAssembler...")
    dance_result = await dancer.assemble(
        partial_responses=split_result.responses,
        system_prompt="Responde em português de forma directa e concisa.",
        llm_clients=dance_clients,
    )

    t_total = (time.perf_counter() - t0) * 1000
    print_dance_report(dance_result, t_total)
    print_privacy_report(message, split_result)


def main():
    parser = argparse.ArgumentParser(
        description="Teste end-to-end: MetatronSplitterSemantic + QuantumDanceAssembler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--msg", default=DEFAULT_MESSAGE, help="Mensagem a testar")
    parser.add_argument(
        "--providers", nargs="+",
        default=["deepseek", "openai", "anthropic"],
        choices=["deepseek", "openai", "anthropic"],
        help="Providers a usar",
    )
    parser.add_argument("--guard", action="store_true",
                        help="Activar guard Ollama (requer Ollama local)")
    parser.add_argument("--guard-url", default="http://localhost:11434/api/generate",
                        help="URL do Ollama para o guard (default: localhost:11434)")
    args = parser.parse_args()

    asyncio.run(run_test(args.msg, args.providers, args.guard, args.guard_url))


if __name__ == "__main__":
    main()
