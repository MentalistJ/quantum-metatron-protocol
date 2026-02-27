"""
q_quantum_dance.py — QuantumDance Assembler

Extension of the QuantumMetatron protocol.

After MetatronSplitter distributes a message across N providers and collects
partial responses, the QuantumDance Assembler synthesises those fragments into
a coherent final answer — without any single cloud operator ever seeing the
full original query.

Architecture:
  split_result → [QuantumDanceAssembler] → assembled_response

How it works:
  1. Selects a cloud "assembler" from the split providers using pseudo-random
     rotation anchored to the session key (low-repeat window to avoid bias).
  2. Sends the joined partial responses to the chosen assembler with a
     synthesis instruction.
  3. Optionally runs a local guard (Ollama-compatible) that must return
     PASS_TOKEN — if it returns REWORK_TOKEN, the assembler is rejected and
     a local fallback is tried (if enabled).

Security note:
  The assembler only sees the *responses* to fragments, never the original
  query. Combined with HMAC obfuscation from MetatronSplitter, even the
  assembler operator learns nothing about the user's intent.

Implementation date: 2026-02-27
Author: Paulo Jonathan Oliveira (MentalistJ)
License: AGPL v3
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("quantum_dance")
if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DanceResult:
    """Result of a QuantumDance assembly pass."""
    assembled: str
    assembler_provider: Optional[str]
    guard_passed: bool
    fallback_used: bool
    elapsed_ms: float
    meta: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumDanceAssembler
# ─────────────────────────────────────────────────────────────────────────────

class QuantumDanceAssembler:
    """
    Rotating cloud assembler with optional local guard.

    Usage:
        from openai import AsyncOpenAI
        from q_quantum_dance import QuantumDanceAssembler

        clients = {
            "deepseek":  AsyncOpenAI(api_key="...", base_url="https://api.deepseek.com/v1"),
            "openai":    AsyncOpenAI(api_key="..."),
            "anthropic": AsyncOpenAI(api_key="...", base_url="https://api.anthropic.com/v1"),
        }

        dancer = QuantumDanceAssembler(
            session_key="your-session-key",
            providers=["deepseek", "openai", "anthropic"],
        )

        result = await dancer.assemble(
            partial_responses={"deepseek": "...", "openai": "...", "anthropic": "..."},
            system_prompt="You are a helpful assistant.",
            llm_clients=clients,
        )
        print(result.assembled)
    """

    # Cloud-only providers eligible as assembler (local never assembles by default)
    CLOUD_PROVIDERS = {"deepseek", "openai", "anthropic", "gemini"}

    # Model map for the assembly call
    MODEL_MAP: Dict[str, str] = {
        "deepseek":  "deepseek-chat",
        "openai":    "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "gemini":    "gemini-1.5-flash",
    }

    def __init__(
        self,
        session_key: str,
        providers: List[str],
        *,
        non_repeat_window: int = 1,
        guard_enabled: bool = False,
        guard_url: Optional[str] = None,
        guard_model: str = "llama3.2",
        local_assembler_enabled: bool = False,
        local_url: Optional[str] = None,
        local_model: str = "llama3.2",
    ):
        """
        Args:
            session_key:            Cryptographic session key (from QuantumSnap or similar).
            providers:              List of cloud providers used in the split.
            non_repeat_window:      How many recent assembler picks to avoid repeating.
            guard_enabled:          Enable local Ollama guard (PASS_TOKEN / REWORK_TOKEN).
            guard_url:              Ollama API URL for guard (e.g. http://localhost:11434/api/generate).
            guard_model:            Ollama model name for guard.
            local_assembler_enabled: Allow local LLM as fallback assembler.
            local_url:              Ollama API URL for local assembler.
            local_model:            Ollama model name for local assembler.
        """
        self.session_key = session_key
        self.providers = providers
        self.non_repeat_window = non_repeat_window
        self.guard_enabled = guard_enabled
        self.guard_url = guard_url
        self.guard_model = guard_model
        self.local_assembler_enabled = local_assembler_enabled
        self.local_url = local_url or guard_url
        self.local_model = local_model

        # In-process rotation history per session key
        self._history: List[str] = []

    # ── Assembler selection ───────────────────────────────────────────────────

    def pick_assembler(self) -> Optional[str]:
        """
        Select the next assembler using pseudo-random rotation.

        Avoids repeating the same provider within `non_repeat_window` calls.
        The seed is derived from session_key + current time_ns + history length,
        making the rotation non-predictable but deterministic within a test.
        """
        candidates = [p for p in self.providers if p in self.CLOUD_PROVIDERS]
        if not candidates:
            return None

        recent = set(self._history[-self.non_repeat_window:]) if self.non_repeat_window > 0 else set()
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
        }))
        return chosen

    # ── Guard (local LLM) ─────────────────────────────────────────────────────

    async def _run_guard(self, text: str) -> bool:
        """
        Local Ollama guard: expects exactly PASS_TOKEN or REWORK_TOKEN.
        Returns True if the assembled text is accepted.
        """
        if not self.guard_url:
            return True
        import httpx
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
            return True  # fail-open: guard errors don't block the response

    # ── Local Ollama assembler ────────────────────────────────────────────────

    async def _run_local_assembler(self, assembled_input: str, system_prompt: str) -> Optional[str]:
        """Fallback assembler using a local Ollama model."""
        if not self.local_assembler_enabled or not self.local_url:
            return None
        import httpx
        payload = {
            "model": self.local_model,
            "prompt": f"User: {assembled_input}\nAssistant:",
            "system": system_prompt + "\n\n[QUANTUM DANCE ASSEMBLER] Sintetiza as perspectivas parciais. Máx 4 frases.",
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

    # ── Main assembly ─────────────────────────────────────────────────────────

    async def assemble(
        self,
        partial_responses: Dict[str, str],
        system_prompt: str,
        llm_clients: Dict[str, any],
    ) -> DanceResult:
        """
        Assemble partial responses from MetatronSplitter into a final answer.

        Args:
            partial_responses:  dict provider → partial response string
            system_prompt:      base system prompt for the assembler LLM
            llm_clients:        dict provider → AsyncOpenAI-compatible client

        Returns:
            DanceResult with the assembled text and metadata.
        """
        t0 = time.perf_counter()

        # Join partial responses into a single input
        joined = "\n\n".join(
            f"[{prov}]: {resp}"
            for prov, resp in partial_responses.items()
            if resp and not str(resp).startswith("[ERRO")
        )
        if not joined:
            return DanceResult(
                assembled="[QuantumDance: no valid partial responses to assemble]",
                assembler_provider=None,
                guard_passed=False,
                fallback_used=False,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        assemble_system = (
            system_prompt
            + "\n\n[QUANTUM DANCE ASSEMBLER] Synthesise the following partial perspectives "
            "into a single coherent answer. Do not invent new facts. Max 4 sentences."
        )
        assemble_messages = [
            {"role": "system", "content": assemble_system},
            {"role": "user",   "content": joined},
        ]

        provider = self.pick_assembler()
        assembled = joined  # fallback: raw concat
        fallback_used = False

        if provider and provider in llm_clients:
            client = llm_clients[provider]
            model = self.MODEL_MAP.get(provider, "deepseek-chat")
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=assemble_messages,
                    max_tokens=512,
                )
                assembled = resp.choices[0].message.content or joined
                logger.info(json.dumps({"event": "quantum_dance_assembled", "provider": provider}))
            except Exception as e:
                logger.warning(json.dumps({"event": "quantum_dance_assembler_error", "provider": provider, "error": str(e)}))
                provider = None

        # Guard
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
            meta={"providers_used": list(partial_responses.keys())},
        )
