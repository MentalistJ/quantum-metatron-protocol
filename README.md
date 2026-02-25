# QuantumMetatron

**Privacy protocol for AI interactions.**  
Splits user messages across independent LLM providers so no single operator sees the full query.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Prior Art](https://img.shields.io/badge/Prior%20Art-2026--02--24-green)]()
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

[Phase 1: Semantic Decompose]
Fragment 1: "Recovering crypto securely"
Fragment 2: "Wallet password protection"

[Phase 2: HMAC Obfuscation & Parallel Dispatch]
DeepSeek receives:  "⟨Y7X1⟩ crypto securely"
OpenAI receives:    "Wallet ⟨Z9Q2⟩ protection"
Anthropic receives: (another fragment...)

[Phase 3: Assembly]
Assembler intelligently joins perspectives locally. No provider saw the full query.
```

## Key Properties

| Property | How |
|---|---|
| **Semantic split** | Round-robin token distribution across N providers |
| **Session-scoped obfuscation** | HMAC-derived token table, unique per session |
| **Parallel dispatch** | All fragments sent simultaneously via asyncio |
| **Local assembly** | Responses joined locally or via local LLM |
| **Faster than single query** | Parallel > sequential (2811ms vs 3068ms in tests) |

## Validated Results (2026-02-24)

Real test: DeepSeek + GPT-4o-mini + Claude Haiku in parallel.

```
Split 3× in parallel:  2811ms  ← faster than single query
Single DeepSeek query: 3068ms

Claude admitted: "I cannot process these fragments — insufficient context."
→ Proof that the split works: operator cannot reconstruct the query.
```

## Installation

```bash
pip install openai anthropic python-dotenv
```

```python
from q_metatron_splitter_semantic import MetatronSplitterSemantic
from openai import AsyncOpenAI

clients = {
    "deepseek":  AsyncOpenAI(api_key="...", base_url="https://api.deepseek.com/v1"),
    "openai":    AsyncOpenAI(api_key="..."),
    "anthropic": AsyncOpenAI(api_key="...", base_url="https://api.anthropic.com/v1"),
}

splitter = MetatronSplitterSemantic(
    session_key="your-session-key",
    providers=["deepseek", "openai", "anthropic"],
    decompose_client=clients["deepseek"]  # Required for Semantic Decompose
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
| `q_metatron_splitter.py` | Original. Round-robin split (word-level), 0 latency overhead. |
| `q_metatron_splitter_semantic.py` | **Evolved.** Advanced Semantic Splitter using LLM for intelligent parallel query routing and local context assembly. |

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

## Roadmap

- [ ] Sentence-level split (current: word-level)
- [ ] Minimum fragment length threshold
- [ ] Pluggable assembler interface
- [ ] Redis-backed session key store
- [ ] CLI tool for standalone use

## License

AGPL v3 — see [LICENSE](LICENSE).

**Commercial use** (SaaS, proprietary products) requires a separate license.  
Contact: oliveirapaulojonathan@gmail.com

## Prior Art

Implementation date: **2026-02-24**.  
See [NOTICE](NOTICE) for prior art documentation.

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
  └── AGEN             — Sovereign AI platform (private)
```

*© 2026 Paulo Jonathan Oliveira (MentalistJ) — QuantumStack Project*
