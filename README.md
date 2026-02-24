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

QuantumMetatron distributes your message across **N independent LLM providers** simultaneously. Each provider receives only a fraction of the tokens — never the complete message.

```
Original: "How do I recover my crypto wallet password securely?"

DeepSeek receives:  "How _ _ my _ wallet _ securely?"         ← no "password", no "crypto"
OpenAI receives:    "_ do I _ crypto _ password _ ?"          ← no "recover", no "wallet"
Anthropic receives: "_ _ _ recover _ _ _ securely ?"          ← no "password", no "wallet"

Assembly happens locally — no provider sees the full query.
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
from q_metatron_splitter import MetatronSplitter
from openai import AsyncOpenAI

splitter = MetatronSplitter(
    session_key="your-session-key",
    providers=["deepseek", "openai", "anthropic"]
)

clients = {
    "deepseek":  AsyncOpenAI(api_key="...", base_url="https://api.deepseek.com/v1"),
    "openai":    AsyncOpenAI(api_key="..."),
    "anthropic": AsyncOpenAI(api_key="...", base_url="https://api.anthropic.com/v1"),
}

result = await splitter.split_and_query(
    message="Your sensitive query here",
    llm_clients=clients
)
print(result.assembled)
```

## Files

| File | Purpose |
|---|---|
| `q_metatron.py` | Session-scoped HMAC token obfuscation + ACK verification |
| `q_metatron_splitter.py` | Multi-provider round-robin split + parallel async query |

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
  author  = {Oliveira, Paulo Jonathan},
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
