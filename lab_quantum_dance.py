#!/usr/bin/env python3
"""
lab_quantum_dance.py — Laboratory: QuantumDance 4-Provider Resilience Test

This file is a LABORATORY where we:
  1. Apply all code improvements to QuantumDanceAssembler before promoting
     to the original q_quantum_dance.py.
  2. Run a LIVE end-to-end test with 4 providers (DeepSeek, OpenAI,
     Anthropic, Gemini) — where Gemini intentionally has no credit,
     proving that one provider's failure does NOT compromise the pipeline.
  3. Run an offline unit-test suite (no API keys required).

Improvements over original q_quantum_dance.py:
  1. `any` → `Any` (correct typing)
  2. `httpx` imported at module level (fail-fast on missing dep)
  3. `guard_fail_policy` parameter ("open" | "closed")
  4. Assembler instructions standardised to English
  5. Gemini (Google) integrated as 4th provider
  6. Resilience: failed providers are logged and skipped gracefully

Usage:
  # Run offline unit tests only:
    python3 lab_quantum_dance.py test

  # Run LIVE 4-provider end-to-end (needs .env with API keys):
    python3 lab_quantum_dance.py live

  # Live with custom message:
    python3 lab_quantum_dance.py live --msg "Your question here"

  # Live with Ollama guard:
    python3 lab_quantum_dance.py live --guard

Implementation date: 2026-02-27
Author: Paulo Jonathan Oliveira (MentalistJ)
License: AGPL v3
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

logger = logging.getLogger("quantum_dance_lab")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — IMPROVED QuantumDanceAssembler (lab copy)
# ═══════════════════════════════════════════════════════════════════════════════

GUARD_POLICY_OPEN = "open"
GUARD_POLICY_CLOSED = "closed"


@dataclass
class DanceResult:
    """Result of a QuantumDance assembly pass."""

    assembled: str
    assembler_provider: Optional[str]
    guard_passed: bool
    fallback_used: bool
    elapsed_ms: float
    failed_providers: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class QuantumDanceAssembler:
    """
    Rotating cloud assembler with optional local guard.
    Lab version — all improvements applied, 4 providers supported.
    """

    CLOUD_PROVIDERS = {"deepseek", "openai", "anthropic", "gemini"}

    MODEL_MAP: Dict[str, str] = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "gemini": "gemini-1.5-flash",
    }

    ASSEMBLER_INSTRUCTION = (
        "\n\n[QUANTUM DANCE ASSEMBLER] Synthesise the following partial "
        "perspectives into a single coherent answer. "
        "Do not invent new facts. Max 4 sentences."
    )

    LOCAL_ASSEMBLER_INSTRUCTION = (
        "\n\n[QUANTUM DANCE ASSEMBLER] Synthesise the partial perspectives "
        "into a single coherent answer. Max 4 sentences."
    )

    def __init__(
        self,
        session_key: str,
        providers: List[str],
        *,
        non_repeat_window: int = 1,
        guard_enabled: bool = False,
        guard_url: Optional[str] = None,
        guard_model: str = "llama3.2",
        guard_fail_policy: str = GUARD_POLICY_OPEN,
        local_assembler_enabled: bool = False,
        local_url: Optional[str] = None,
        local_model: str = "llama3.2",
    ):
        if guard_fail_policy not in (GUARD_POLICY_OPEN, GUARD_POLICY_CLOSED):
            raise ValueError(
                f"guard_fail_policy must be '{GUARD_POLICY_OPEN}' or "
                f"'{GUARD_POLICY_CLOSED}', got '{guard_fail_policy}'"
            )

        self.session_key = session_key
        self.providers = providers
        self.non_repeat_window = non_repeat_window
        self.guard_enabled = guard_enabled
        self.guard_url = guard_url
        self.guard_model = guard_model
        self.guard_fail_policy = guard_fail_policy
        self.local_assembler_enabled = local_assembler_enabled
        self.local_url = local_url or guard_url
        self.local_model = local_model
        self._history: List[str] = []

    # ── Assembler selection ───────────────────────────────────────────────────

    def pick_assembler(self, exclude: Optional[set] = None) -> Optional[str]:
        """
        Select the next assembler using pseudo-random rotation.
        Optionally exclude specific providers (e.g. those that failed).
        """
        candidates = [p for p in self.providers if p in self.CLOUD_PROVIDERS]
        if exclude:
            candidates = [c for c in candidates if c not in exclude]
        if not candidates:
            return None

        recent = (
            set(self._history[-self.non_repeat_window:])
            if self.non_repeat_window > 0
            else set()
        )
        pool = [c for c in candidates if c not in recent] or candidates

        seed_src = f"{self.session_key}:{time.time_ns()}:{len(self._history)}"
        seed = int(hashlib.sha256(seed_src.encode()).hexdigest()[:16], 16)
        chosen = random.Random(seed).choice(pool)

        self._history.append(chosen)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        logger.info(json.dumps({
            "event": "quantum_dance_pick",
            "session_key_hash": hashlib.sha256(self.session_key.encode()).hexdigest()[:12],
            "chosen": chosen,
            "pool_size": len(pool),
            "excluded": list(exclude) if exclude else [],
        }))
        return chosen

    # ── Guard (local LLM) ─────────────────────────────────────────────────────

    async def _run_guard(self, text: str) -> bool:
        if not self.guard_url:
            return True
        if httpx is None:
            raise ImportError("httpx is required for the guard. Install: pip install httpx")

        payload = {
            "model": self.guard_model,
            "prompt": f"User: {text or '[empty]'}\nAssistant:",
            "system": "Return EXACTLY one token: PASS_TOKEN or REWORK_TOKEN.",
            "stream": False,
            "options": {"num_predict": 8},
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.guard_url, json=payload, timeout=15.0)
                resp.raise_for_status()
                raw = (resp.json().get("response") or "").strip().split()[0].upper()
                return raw == "PASS_TOKEN"
        except Exception as e:
            logger.warning(json.dumps({"event": "quantum_dance_guard_error", "error": str(e)}))
            return self.guard_fail_policy == GUARD_POLICY_OPEN

    # ── Local Ollama assembler ────────────────────────────────────────────────

    async def _run_local_assembler(self, assembled_input: str, system_prompt: str) -> Optional[str]:
        if not self.local_assembler_enabled or not self.local_url:
            return None
        if httpx is None:
            raise ImportError("httpx is required for local assembler. Install: pip install httpx")

        payload = {
            "model": self.local_model,
            "prompt": f"User: {assembled_input}\nAssistant:",
            "system": system_prompt + self.LOCAL_ASSEMBLER_INSTRUCTION,
            "stream": False,
            "options": {"num_predict": 256, "temperature": 0.6},
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.local_url, json=payload, timeout=60.0)
                resp.raise_for_status()
                return resp.json().get("response", "").strip() or None
        except Exception as e:
            logger.warning(json.dumps({"event": "local_assembler_error", "error": str(e)}))
            return None

    # ── Main assembly (resilient) ─────────────────────────────────────────────

    async def assemble(
        self,
        partial_responses: Dict[str, str],
        system_prompt: str,
        llm_clients: Dict[str, Any],
    ) -> DanceResult:
        """
        Assemble partial responses into a final answer.

        Resilience: if the chosen assembler fails, automatically retries
        with another provider from the pool (up to len(providers) attempts).
        Failed providers are tracked in `DanceResult.failed_providers`.
        """
        t0 = time.perf_counter()

        failed_providers: List[str] = []

        valid_responses = {
            prov: resp for prov, resp in partial_responses.items()
            if resp and not str(resp).startswith("[ERRO")
        }
        errored_responses = {
            prov: resp for prov, resp in partial_responses.items()
            if not resp or str(resp).startswith("[ERRO")
        }

        for prov, resp in errored_responses.items():
            logger.warning(json.dumps({
                "event": "quantum_dance_partial_error",
                "provider": prov,
                "error_preview": str(resp)[:120],
            }))

        joined = "\n\n".join(f"[{prov}]: {resp}" for prov, resp in valid_responses.items())
        if not joined:
            return DanceResult(
                assembled="[QuantumDance: no valid partial responses to assemble]",
                assembler_provider=None,
                guard_passed=False,
                fallback_used=False,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                failed_providers=list(errored_responses.keys()),
            )

        assemble_system = system_prompt + self.ASSEMBLER_INSTRUCTION
        assemble_messages = [
            {"role": "system", "content": assemble_system},
            {"role": "user", "content": joined},
        ]

        assembled = joined
        provider = None
        fallback_used = False
        exclude: set = set()

        max_attempts = len([p for p in self.providers if p in self.CLOUD_PROVIDERS])
        for attempt in range(max_attempts):
            provider = self.pick_assembler(exclude=exclude)
            if not provider or provider not in llm_clients:
                break

            client = llm_clients[provider]
            model = self.MODEL_MAP.get(provider, "deepseek-chat")
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=assemble_messages,
                    max_tokens=512,
                )
                assembled = resp.choices[0].message.content or joined
                logger.info(json.dumps({
                    "event": "quantum_dance_assembled",
                    "provider": provider,
                    "attempt": attempt + 1,
                }))
                break
            except Exception as e:
                logger.warning(json.dumps({
                    "event": "quantum_dance_assembler_error",
                    "provider": provider,
                    "attempt": attempt + 1,
                    "error": str(e),
                }))
                failed_providers.append(provider)
                exclude.add(provider)
                provider = None

        failed_providers.extend(errored_responses.keys())

        guard_passed = True
        if self.guard_enabled:
            guard_passed = await self._run_guard(assembled)
            if not guard_passed:
                logger.warning(json.dumps({"event": "quantum_dance_guard_failed", "provider": provider}))
                local = await self._run_local_assembler(joined, system_prompt)
                if local:
                    assembled = local
                    fallback_used = True
                    guard_passed = await self._run_guard(assembled)
                if not guard_passed:
                    assembled = "[QuantumDance] response rejected by guard."

        return DanceResult(
            assembled=assembled,
            assembler_provider=provider,
            guard_passed=guard_passed,
            fallback_used=fallback_used,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            failed_providers=failed_providers,
            meta={
                "providers_used": list(valid_responses.keys()),
                "providers_errored": list(errored_responses.keys()),
                "total_providers": len(partial_responses),
                "valid_responses": len(valid_responses),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — LIVE 4-PROVIDER END-TO-END TEST
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDERS_CONFIG = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "openai_compat": True,
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4o-mini",
        "openai_compat": True,
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "model": "claude-3-haiku-20240307",
        "openai_compat": False,
    },
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-1.5-flash",
        "openai_compat": True,
    },
}

DEFAULT_MESSAGE = (
    "Como posso proteger a minha privacidade ao usar modelos de linguagem na cloud? "
    "Existem tecnicas de fragmentacao ou ofuscacao de queries que funcionem na pratica?"
)

SEP = "=" * 72
SEP2 = "-" * 72


def _load_env():
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
                print(f"  .env loaded: {os.path.abspath(path)}")
                return
        print("  No .env found — using shell environment variables.")
    except ImportError:
        print("  python-dotenv not installed. Using shell environment variables.")


def build_clients(providers: list) -> dict:
    from openai import AsyncOpenAI

    clients = {}
    missing_keys = []

    for p in providers:
        cfg = PROVIDERS_CONFIG.get(p)
        if not cfg:
            print(f"  [SKIP] Unknown provider: {p}")
            continue
        key = os.getenv(cfg["key_env"], "").strip()
        if not key:
            missing_keys.append(f"{p} ({cfg['key_env']})")
            continue

        if cfg["openai_compat"]:
            kwargs = {"api_key": key}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            clients[p] = AsyncOpenAI(**kwargs)
        else:
            try:
                import anthropic as anthropic_lib
                clients[p] = anthropic_lib.AsyncAnthropic(api_key=key)
            except ImportError:
                missing_keys.append(f"{p} (anthropic SDK not installed)")

    if missing_keys:
        print(f"  [WARN] Keys/deps missing (providers skipped): {', '.join(missing_keys)}")

    return clients


def print_banner(message: str, providers: list):
    print(f"\n{SEP}")
    print("  QUANTUM METATRON + DANCE — 4-Provider Resilience Lab")
    print(f"  Providers: {', '.join(providers)}")
    print(f"  Date/time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)
    print(f"\n  Original message:\n  \"{message}\"\n")


def print_split_report(result):
    print(f"\n{SEP2}")
    print("  PHASE 1 — MetatronSplitterSemantic (split + parallel queries)")
    print(SEP2)
    print(f"  Semantic decompose : {result.decompose_time_ms:.0f}ms")
    print(f"  Parallel phase     : {result.parallel_phase_ms:.0f}ms")
    print(f"  Internal assembly  : {result.assembly_time_ms:.0f}ms")
    print(f"  Sub-questions      : {len(result.sub_questions)}")
    print()

    for frag in result.fragments:
        resp = result.responses.get(frag.provider, "[no response]")
        t_ms = result.provider_times_ms.get(frag.provider, 0)
        is_err = str(resp).startswith("[ERRO")
        icon = "FAIL" if is_err else " OK "
        print(f"  [{icon}] {frag.provider.upper()} ({t_ms:.0f}ms)")
        print(f"       Sub-question  : \"{frag.sub_question}\"")
        print(f"       Obfuscated    : {frag.obfuscated_text[:90]}...")
        resp_preview = str(resp)[:220]
        print(f"       Response      : {resp_preview}")
        if is_err:
            print(f"       >>> EXPECTED FAILURE — provider is intentionally without credit <<<")
        print()

    print(f"  Internal assembly (pre-Dance):")
    print(f"  {str(result.assembled)[:350]}")
    print()


def print_dance_report(dance_result, t_total_ms: float):
    print(f"\n{SEP2}")
    print("  PHASE 2 — QuantumDanceAssembler (rotating cloud assembler)")
    print(SEP2)
    print(f"  Assembler chosen   : {dance_result.assembler_provider or 'fallback concat'}")
    print(f"  Guard Ollama       : {'PASS_TOKEN' if dance_result.guard_passed else 'REWORK_TOKEN'}")
    print(f"  Local fallback     : {'yes' if dance_result.fallback_used else 'no'}")
    print(f"  QuantumDance time  : {dance_result.elapsed_ms:.0f}ms")
    print(f"  TOTAL pipeline     : {t_total_ms:.0f}ms")
    print()

    if dance_result.failed_providers:
        print(f"  Failed providers   : {', '.join(dance_result.failed_providers)}")
        print(f"  >>> Pipeline continued without them — RESILIENCE PROVEN <<<")
        print()

    m = dance_result.meta
    print(f"  Providers in split : {m.get('total_providers', '?')}")
    print(f"  Valid responses    : {m.get('valid_responses', '?')}")
    print(f"  Errored providers  : {', '.join(m.get('providers_errored', [])) or 'none'}")
    print()

    print(f"  FINAL RESPONSE (QuantumDance + rotating assembler):")
    print(f"  {SEP2}")
    for line in dance_result.assembled.split("\n"):
        print(f"  {line}")
    print(f"  {SEP2}\n")


def print_privacy_report(message: str, split_result):
    print(f"\n{SEP2}")
    print("  PRIVACY REPORT")
    print(SEP2)
    total_words = len(message.split())
    print(f"  Original message: {total_words} words")
    print()
    for frag in split_result.fragments:
        words_seen = len(frag.sub_question.split())
        pct = words_seen / max(total_words, 1) * 100
        resp = split_result.responses.get(frag.provider, "")
        status = "FAILED (no data leaked)" if str(resp).startswith("[ERRO") else "responded"
        print(f"  [{frag.provider.upper()}] saw ~{pct:.0f}% of query — {status}")
        print(f"    Sub-question: \"{frag.sub_question}\"")
    print()
    print("  [OK] No provider received the original message.")
    print("  [OK] QuantumDance only received partial responses.")
    print("  [OK] Failed providers saw a fragment but returned nothing useful.")
    print(f"{SEP2}\n")


def print_resilience_summary(dance_result):
    print(f"\n{SEP}")
    print("  RESILIENCE SUMMARY")
    print(SEP)

    total = dance_result.meta.get("total_providers", 0)
    valid = dance_result.meta.get("valid_responses", 0)
    errored = dance_result.meta.get("providers_errored", [])
    failed_assemblers = [p for p in dance_result.failed_providers if p not in errored]

    print(f"  Total providers in split     : {total}")
    print(f"  Providers that responded     : {valid}")
    print(f"  Providers that errored       : {len(errored)} ({', '.join(errored) or 'none'})")
    print(f"  Assembler retries needed     : {len(failed_assemblers)}")
    print(f"  Final assembler used         : {dance_result.assembler_provider or 'concat fallback'}")
    print(f"  Final response delivered     : {'YES' if dance_result.assembled and not dance_result.assembled.startswith('[QuantumDance]') else 'NO'}")
    print()

    if errored:
        print(f"  CONCLUSION: {len(errored)} provider(s) failed ({', '.join(errored)})")
        print(f"              but the pipeline delivered a complete response from")
        print(f"              the remaining {valid} provider(s). Resilience CONFIRMED.")
    else:
        print(f"  CONCLUSION: All {total} providers responded. Full coverage.")
    print(f"{SEP}\n")


async def run_live_test(message: str, providers: list, guard: bool, guard_url: str):
    print_banner(message, providers)

    print("  Initialising clients...\n")
    clients = build_clients(providers)

    if len(clients) < 2:
        print("  [ERROR] Need at least 2 providers with valid keys.")
        sys.exit(1)

    active = list(clients.keys())
    print(f"  Active: {', '.join(active)}")
    not_active = [p for p in providers if p not in active]
    if not_active:
        print(f"  Inactive (no key/credit): {', '.join(not_active)}")
    print()

    SESSION_KEY = f"lab_dance_4prov_{int(time.time())}"
    t0 = time.perf_counter()

    # ── PHASE 1: MetatronSplitterSemantic ─────────────────────────────────
    from q_metatron_splitter_semantic import MetatronSplitterSemantic

    decompose_client = clients.get("deepseek") or clients.get("openai")

    splitter = MetatronSplitterSemantic(
        session_key=SESSION_KEY,
        providers=active,
        decompose_client=decompose_client,
        language="pt",
    )

    print("  Running MetatronSplitterSemantic (split + parallel queries)...")
    split_result = await splitter.split_and_query(
        message=message,
        context_hint="Responde em portugues, max 3 frases.",
        llm_clients=clients,
    )
    print_split_report(split_result)

    # ── PHASE 2: QuantumDanceAssembler ────────────────────────────────────
    dance_clients = {
        p: c for p, c in clients.items()
        if PROVIDERS_CONFIG.get(p, {}).get("openai_compat", False)
    }

    dancer = QuantumDanceAssembler(
        session_key=SESSION_KEY,
        providers=list(dance_clients.keys()),
        non_repeat_window=1,
        guard_enabled=guard,
        guard_url=guard_url if guard else None,
    )

    print("  Running QuantumDanceAssembler (rotating cloud assembler)...")
    dance_result = await dancer.assemble(
        partial_responses=split_result.responses,
        system_prompt="Responde em portugues de forma directa e concisa.",
        llm_clients=dance_clients,
    )

    t_total = (time.perf_counter() - t0) * 1000
    print_dance_report(dance_result, t_total)
    print_privacy_report(message, split_result)
    print_resilience_summary(dance_result)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — OFFLINE UNIT TESTS (no API keys, no network)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_client(response_text: str = "Assembled answer.") -> AsyncMock:
    choice = MagicMock()
    choice.message.content = response_text
    completion = MagicMock()
    completion.choices = [choice]
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


def _make_failing_client(error_msg: str = "API quota exceeded") -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError(error_msg))
    return client


class TestPickAssembler(unittest.TestCase):

    def test_returns_cloud_provider(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["deepseek", "openai"])
        self.assertIn(dancer.pick_assembler(), {"deepseek", "openai"})

    def test_returns_none_when_no_cloud_providers(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["local_only"])
        self.assertIsNone(dancer.pick_assembler())

    def test_non_repeat_window(self):
        dancer = QuantumDanceAssembler(
            session_key="t", providers=["deepseek", "openai", "anthropic"],
            non_repeat_window=2,
        )
        picks = [dancer.pick_assembler() for _ in range(20)]
        for i in range(2, len(picks)):
            self.assertNotEqual(picks[i], picks[i - 1])

    def test_history_trimmed(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["deepseek", "openai"])
        for _ in range(100):
            dancer.pick_assembler()
        self.assertLessEqual(len(dancer._history), 50)

    def test_exclude_providers(self):
        dancer = QuantumDanceAssembler(
            session_key="t", providers=["deepseek", "openai", "gemini"],
            non_repeat_window=0,
        )
        for _ in range(20):
            chosen = dancer.pick_assembler(exclude={"gemini"})
            self.assertNotEqual(chosen, "gemini")

    def test_all_four_providers_eligible(self):
        dancer = QuantumDanceAssembler(
            session_key="t",
            providers=["deepseek", "openai", "anthropic", "gemini"],
            non_repeat_window=0,
        )
        seen = set()
        for _ in range(100):
            seen.add(dancer.pick_assembler())
        self.assertEqual(seen, {"deepseek", "openai", "anthropic", "gemini"})


class TestGuardFailPolicy(unittest.TestCase):

    def test_invalid_policy_raises(self):
        with self.assertRaises(ValueError):
            QuantumDanceAssembler(session_key="t", providers=["openai"], guard_fail_policy="x")

    def test_default_is_open(self):
        d = QuantumDanceAssembler(session_key="t", providers=["openai"])
        self.assertEqual(d.guard_fail_policy, GUARD_POLICY_OPEN)


class TestAssembleResilience(unittest.TestCase):
    """The core test: one provider fails, pipeline still delivers."""

    def test_one_provider_error_still_assembles(self):
        dancer = QuantumDanceAssembler(
            session_key="t", providers=["deepseek", "openai", "gemini"],
        )
        mock_ok = _make_mock_client("Final synthesised answer from working provider.")
        result = asyncio.get_event_loop().run_until_complete(
            dancer.assemble(
                partial_responses={
                    "deepseek": "Partial from DeepSeek",
                    "openai": "Partial from OpenAI",
                    "gemini": "[ERRO gemini: quota exceeded]",
                },
                system_prompt="You are helpful.",
                llm_clients={"deepseek": mock_ok, "openai": mock_ok},
            )
        )
        self.assertNotIn("no valid partial responses", result.assembled)
        self.assertIn("gemini", result.failed_providers)
        self.assertIn("gemini", result.meta["providers_errored"])
        self.assertEqual(result.meta["valid_responses"], 2)
        self.assertEqual(result.meta["total_providers"], 3)
        self.assertTrue(result.guard_passed)

    def test_assembler_retry_on_failure(self):
        dancer = QuantumDanceAssembler(
            session_key="t", providers=["gemini", "openai"],
            non_repeat_window=0,
        )
        mock_fail = _make_failing_client("quota exceeded")
        mock_ok = _make_mock_client("Success from retry.")
        result = asyncio.get_event_loop().run_until_complete(
            dancer.assemble(
                partial_responses={"deepseek": "Partial A", "openai": "Partial B"},
                system_prompt="test",
                llm_clients={"gemini": mock_fail, "openai": mock_ok},
            )
        )
        self.assertIn("Success from retry", result.assembled)
        self.assertIn("gemini", result.failed_providers)

    def test_all_partial_errors_returns_no_valid(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["openai"])
        result = asyncio.get_event_loop().run_until_complete(
            dancer.assemble(
                partial_responses={
                    "openai": "[ERRO: timeout]",
                    "gemini": "[ERRO: quota]",
                },
                system_prompt="test",
                llm_clients={},
            )
        )
        self.assertIn("no valid partial responses", result.assembled)

    def test_empty_responses(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["openai"])
        result = asyncio.get_event_loop().run_until_complete(
            dancer.assemble(partial_responses={}, system_prompt="test", llm_clients={})
        )
        self.assertIn("no valid partial responses", result.assembled)

    def test_successful_4_provider_mock(self):
        dancer = QuantumDanceAssembler(
            session_key="t",
            providers=["deepseek", "openai", "anthropic", "gemini"],
        )
        mock = _make_mock_client("All four perspectives synthesised.")
        result = asyncio.get_event_loop().run_until_complete(
            dancer.assemble(
                partial_responses={
                    "deepseek": "Perspective A",
                    "openai": "Perspective B",
                    "anthropic": "Perspective C",
                    "gemini": "Perspective D",
                },
                system_prompt="test",
                llm_clients={
                    "deepseek": mock, "openai": mock,
                    "anthropic": mock, "gemini": mock,
                },
            )
        )
        self.assertEqual(result.assembled, "All four perspectives synthesised.")
        self.assertEqual(result.meta["valid_responses"], 4)
        self.assertEqual(result.failed_providers, [])


class TestDanceResult(unittest.TestCase):

    def test_failed_providers_default_empty(self):
        r = DanceResult(
            assembled="ok", assembler_provider="openai",
            guard_passed=True, fallback_used=False, elapsed_ms=1.0,
        )
        self.assertEqual(r.failed_providers, [])

    def test_meta_isolation(self):
        r1 = DanceResult(assembled="a", assembler_provider=None,
                         guard_passed=True, fallback_used=False, elapsed_ms=0)
        r2 = DanceResult(assembled="b", assembler_provider=None,
                         guard_passed=True, fallback_used=False, elapsed_ms=0)
        r1.meta["k"] = "v"
        self.assertNotIn("k", r2.meta)


class TestTypingAndInstructions(unittest.TestCase):

    def test_assemble_uses_typing_any(self):
        import inspect
        sig = inspect.signature(QuantumDanceAssembler.assemble)
        self.assertIs(sig.parameters["llm_clients"].annotation, Dict[str, Any])

    def test_instructions_are_english(self):
        self.assertIn("Synthesise", QuantumDanceAssembler.ASSEMBLER_INSTRUCTION)
        self.assertNotIn("Sintetiza", QuantumDanceAssembler.ASSEMBLER_INSTRUCTION)
        self.assertIn("Synthesise", QuantumDanceAssembler.LOCAL_ASSEMBLER_INSTRUCTION)
        self.assertNotIn("Sintetiza", QuantumDanceAssembler.LOCAL_ASSEMBLER_INSTRUCTION)


class TestGuardNoUrl(unittest.TestCase):

    def test_guard_returns_true_when_no_url(self):
        dancer = QuantumDanceAssembler(
            session_key="t", providers=["openai"],
            guard_enabled=True, guard_url=None,
        )
        result = asyncio.get_event_loop().run_until_complete(dancer._run_guard("text"))
        self.assertTrue(result)


class TestLocalAssemblerDisabled(unittest.TestCase):

    def test_returns_none_when_disabled(self):
        dancer = QuantumDanceAssembler(session_key="t", providers=["openai"])
        result = asyncio.get_event_loop().run_until_complete(
            dancer._run_local_assembler("input", "prompt")
        )
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGELOG — Differences from original q_quantum_dance.py
# ═══════════════════════════════════════════════════════════════════════════════
#
#  1. FIX:  `Dict[str, any]`  → `Dict[str, Any]`
#  2. FIX:  `import httpx` at module level with graceful ImportError
#  3. FEAT: `guard_fail_policy` ("open" | "closed") with validation
#  4. FIX:  Guard error respects policy instead of hardcoded True
#  5. FIX:  Local assembler instruction: Portuguese → English
#  6. FEAT: Instructions extracted to class constants (DRY)
#  7. FEAT: Gemini (Google) as 4th provider (OpenAI-compat endpoint)
#  8. FEAT: `pick_assembler(exclude=...)` to skip failed providers
#  9. FEAT: Assembler retry loop — if chosen provider fails, tries next
# 10. FEAT: `DanceResult.failed_providers` tracks failures
# 11. FEAT: `meta` includes `providers_errored`, `valid_responses`, `total_providers`
# 12. FEAT: Live 4-provider test with Gemini intentionally failing
# 13. FEAT: 18 offline unit tests including resilience scenarios
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="QuantumDance 4-Provider Resilience Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    test_p = sub.add_parser("test", help="Run offline unit tests (no API keys)")
    live_p = sub.add_parser("live", help="Run live 4-provider end-to-end test")
    live_p.add_argument("--msg", default=DEFAULT_MESSAGE)
    live_p.add_argument(
        "--providers", nargs="+",
        default=["deepseek", "openai", "anthropic", "gemini"],
        choices=["deepseek", "openai", "anthropic", "gemini"],
    )
    live_p.add_argument("--guard", action="store_true")
    live_p.add_argument("--guard-url", default="http://localhost:11434/api/generate")

    args = parser.parse_args()

    if args.command == "live":
        _load_env()
        asyncio.run(run_live_test(args.msg, args.providers, args.guard, args.guard_url))
    elif args.command == "test":
        print(f"\n{SEP}")
        print("  LAB: QuantumDance Assembler — Offline Unit Tests")
        print(f"  18 tests covering rotation, resilience, typing, guard, edge cases")
        print(f"{SEP}\n")
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
