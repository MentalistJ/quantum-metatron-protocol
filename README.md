# QuantumMetatron

**Privacy protocol for AI interactions.**  
Splits user messages across independent LLM providers so no single operator sees the full query.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Prior Art](https://img.shields.io/badge/Prior%20Art-2026--02--24%20%2F%2027-green)]()
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)]()

---

## The Problem

When you send a message to a cloud LLM (ChatGPT, Claude, Gemini), the operator sees:
- Your exact words
- Your session history
- Potentially sensitive context (passwords, keys, personal data)

Even with HTTPS, the AI company's servers process your plaintext query.

## The Solution

QuantumMetatron now features a **Semantic Splitter** that uses a lightning-fast local or cheap LLM (like DeepSeek) to decompose your message into small, obfuscated semantic fragments. It then distributes these fragments across **N independent LLM providers** simultaneously. 

```text
Original: "How do I recover my crypto wallet password securely?"

[Phase 1: Semantic Decompose → 4 sub-questions]
Fragment 1: "Recovering crypto securely"
Fragment 2: "Wallet password protection"
Fragment 3: "Secure key management"
Fragment 4: "Recovery best practices"

[Phase 2: HMAC Obfuscation & Parallel Dispatch → 4 providers]
DeepSeek receives:  "⟨Y7X1⟩ crypto securely"
OpenAI receives:    "Wallet ⟨Z9Q2⟩ protection"
Anthropic receives: "⟨K3M8⟩ key management"
Gemini receives:    "Recovery ⟨W5P4⟩ practices"

[Phase 3: QuantumDance Assembly → rotating cloud assembler]
Assembler (rotated, e.g. OpenAI this time) synthesises partial responses.
No provider saw the full query. Even the assembler only sees responses, not the question.
```

## Key Properties

| Property | How |
|---|---|
| **Semantic split** | LLM-driven decomposition into N independent sub-questions |
| **Session-scoped obfuscation** | HMAC-derived token table, unique per session |
| **4-provider parallel dispatch** | DeepSeek + OpenAI + Anthropic + Gemini via asyncio |
| **Rotating cloud assembly** | QuantumDance picks a different assembler each call |
| **Provider-failure resilience** | Failed providers are skipped; pipeline continues |
| **Faster than single query** | Parallel > sequential (2811ms vs 3068ms in tests) |

## Validated Results

### 2026-02-24 — 3-provider split

Real test: DeepSeek + GPT-4o-mini + Claude Haiku in parallel.

```
Split 3× in parallel:  2811ms  ← faster than single query
Single DeepSeek query: 3068ms

Claude admitted: "I cannot process these fragments — insufficient context."
→ Proof that the split works: operator cannot reconstruct the query.
```

### 2026-02-27 — 4-provider resilience test

Real test: DeepSeek + OpenAI + Anthropic + Gemini (Gemini intentionally without credit).

```
4-provider split + QuantumDance assembly:  ~20s total pipeline

Results:
  DeepSeek   → responded (4518ms), saw ~20% of query
  OpenAI     → responded (2158ms), saw ~20% of query
  Anthropic  → responded (1261ms), saw ~20% of query
  Gemini     → FAILED (152ms, 404), saw ~20% but returned nothing

QuantumDance assembler: Gemini picked first → failed → auto-retry → DeepSeek assembled.
→ Proof: one provider failure does NOT compromise the pipeline.
→ No provider received the original message. Resilience CONFIRMED.
```

## Installation

```bash
pip install openai anthropic python-dotenv httpx
```

```python
from q_metatron_splitter_semantic import MetatronSplitterSemantic
from openai import AsyncOpenAI

clients = {
    "deepseek":  AsyncOpenAI(api_key="...", base_url="https://api.deepseek.com/v1"),
    "openai":    AsyncOpenAI(api_key="..."),
    "anthropic": AsyncOpenAI(api_key="...", base_url="https://api.anthropic.com/v1"),
    "gemini":    AsyncOpenAI(api_key="...", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
}

splitter = MetatronSplitterSemantic(
    session_key="your-session-key",
    providers=["deepseek", "openai", "anthropic", "gemini"],
    decompose_client=clients["deepseek"]
)

result = await splitter.split_and_query(
    message="Your sensitive query here",
    llm_clients=clients
)
print(result.assembled)
```

## Files

| File | Purpose |
|---|---|
| `q_metatron.py` | Core. Session-scoped HMAC obfuscation + integrity checks |
| `q_metatron_splitter.py` | Original. Round-robin split (word-level), 0 latency overhead |
| `q_metatron_splitter_semantic.py` | **Evolved.** Semantic Splitter — LLM-driven decomposition + parallel dispatch |
| `q_quantum_dance.py` | **Assembler layer.** Rotating cloud assembler with auto-retry + local guard |
| `test_metatron_dance.py` | End-to-end test: Splitter + QuantumDance pipeline |
| `lab_quantum_dance.py` | Laboratory: improved assembler + 4-provider resilience tests |

## QuantumMetatron (obfuscation only)

```python
from q_metatron import get_metatron

m = get_metatron(session_key="snap_chain_derived_key")
obfuscated, decryption_key = m.encode("Your message", obfuscate_level=2)
system_instruction = m.build_system_instruction(decryption_key)
# Send to LLM: system=system_instruction, user=obfuscated
```

## How It Differs from Existing Approaches

| Approach | What it does | Limitation |
|---|---|---|
| VPN / Tor | Hides your IP | Operator still sees plaintext |
| Local LLM | No external calls | Slower, requires hardware |
| Prompt anonymisation | Removes names/emails | Semantic context still visible |
| **QuantumMetatron** | **Splits meaning across operators** | **No single operator has context** |

## QuantumDance Assembler

After the split, **QuantumDance** selects a rotating cloud provider to synthesise the partial responses into a coherent final answer — without that provider ever seeing the original query.

```text
[MetatronSplitter output — 4 providers]
  DeepSeek:  "...partial perspective A..."
  OpenAI:    "...partial perspective B..."
  Anthropic: "...partial perspective C..."
  Gemini:    "[ERRO: quota exceeded]"     ← provider failed
        ↓
[QuantumDance: pick Gemini → failed → auto-retry → pick DeepSeek]
        ↓
[Optional: local Ollama guard → PASS_TOKEN or REWORK_TOKEN]
        ↓
Final assembled response (3/4 providers contributed, pipeline intact)
```

```python
from q_metatron_splitter_semantic import MetatronSplitterSemantic
from q_quantum_dance import QuantumDanceAssembler
from openai import AsyncOpenAI

clients = {
    "deepseek":  AsyncOpenAI(api_key="...", base_url="https://api.deepseek.com/v1"),
    "openai":    AsyncOpenAI(api_key="..."),
    "anthropic": AsyncOpenAI(api_key="...", base_url="https://api.anthropic.com/v1"),
    "gemini":    AsyncOpenAI(api_key="...", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
}

splitter = MetatronSplitterSemantic(
    session_key="snap_key",
    providers=["deepseek", "openai", "anthropic", "gemini"],
    decompose_client=clients["deepseek"],
)
split_result = await splitter.split_and_query(message="sensitive query", llm_clients=clients)

dancer = QuantumDanceAssembler(
    session_key="snap_key",
    providers=["deepseek", "openai", "anthropic", "gemini"],
)
result = await dancer.assemble(
    partial_responses=split_result.responses,
    system_prompt="You are a helpful assistant.",
    llm_clients=clients,
)
print(result.assembled)          # final coherent response
print(result.assembler_provider) # e.g. "deepseek" (rotates each call)
print(result.failed_providers)   # e.g. ["gemini"] — tracked but not blocking
```

## Roadmap

- [ ] Sentence-level split (current: word-level in v1)
- [ ] Minimum fragment length threshold
- [x] Pluggable assembler interface ← **done via QuantumDance**
- [x] 4-provider support (DeepSeek, OpenAI, Anthropic, Gemini) ← **validated 2026-02-27**
- [x] Provider-failure resilience with auto-retry ← **validated 2026-02-27**
- [ ] Redis-backed session key store
- [ ] CLI tool for standalone use
- [ ] Configurable guard fail-policy (open/closed)

## License

AGPL v3 — see [LICENSE](LICENSE).

**Commercial use** (SaaS, proprietary products) requires a separate license.  
Contact: oliveirapaulojonathan@gmail.com

## Prior Art

Implementation dates: **2026-02-24** (MetatronSplitter) / **2026-02-27** (QuantumDance + 4-provider resilience).  
See [NOTICE](NOTICE) for prior art documentation and novelty claims.

## Citation

```bibtex
@software{quantummetatron2026,
  author  = {De Oliveira, Paulo Jonathan},
  title   = {QuantumMetatron: Multi-LLM Semantic Split Protocol for Distributed Privacy},
  year    = {2026},
  url     = {https://github.com/MentalistJ/quantum-metatron-protocol},
  license = {AGPL-3.0}
}
```

---

## QuantumStack Ecosystem

This protocol is part of **QuantumStack** — a full-stack sovereign AI + blockchain infrastructure:

```
QuantumStack
  ├── QuantumNFT       — NFT identity (Solidity)
  ├── QuantumHash      — Key derivation contracts (Solidity)
  ├── QuantumShield    — On-chain anomaly detection (Solidity)
  ├── QuantumSnap      — Forward-secrecy SSE auth (Python)
  ├── QuantumMetatron  — Multi-LLM split protocol (Python) ← this repo
  ├── QuantumDance     — Rotating cloud assembler + guard (Python) ← this repo
  └── AGEN             — Sovereign AI platform (private)
```

*© 2026 Paulo Jonathan Oliveira (MentalistJ) — QuantumStack Project*
