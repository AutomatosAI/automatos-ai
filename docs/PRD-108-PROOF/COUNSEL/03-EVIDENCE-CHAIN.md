# Evidence Chain

## Purpose

This document separates evidence by type and strength so counsel can assess what is solid and what needs caveats.

## Conception Timeline

| Date | Event | Evidence |
|------|-------|---------|
| 2026-03-15 | PRD-108 specification completed | `docs/PRDS/108-MEMORY-FIELD-PROTOTYPE.md` |
| 2026-03-21 03:23 UTC | First commit with working implementation | Git commit `674474722ac998c06a016cda4c2e5d05f9fbda68` |
| 2026-03-21 ~04:00 UTC | Unit tests passing (57/57) | `orchestrator/tests/test_vector_field.py` |
| 2026-03-21 ~04:30 UTC | Proof suite passing (33/33) | `orchestrator/tests/test_prd108_proof.py` |
| 2026-03-21 ~05:00 UTC | Multi-scenario suite passing (13/13) | `orchestrator/tests/test_prd108_scenarios.py` |
| 2026-03-21 05:59 UTC | Production mission started | Mission 77c58227, Railway deployment logs |
| 2026-03-21 06:17 UTC | Production mission completed | Mission accepted by user |
| 2026-03-21 ~06:30 UTC | Evidence documents written | This folder |

All timestamps are verifiable via git log and Railway deployment logs.

## Evidence Type 1: Code (Strong)

**What it proves:** The invention is implemented and functional.

| Artifact | Path | What It Shows |
|----------|------|---------------|
| Abstract interface | `orchestrator/core/ports/context.py` | SharedContextPort ABC with 4 methods |
| Vector field adapter | `orchestrator/modules/context/adapters/vector_field.py` | Full implementation (~357 lines) |
| Message-passing baseline | `orchestrator/modules/context/adapters/redis_context.py` | Comparison backend |
| Backend factory | `orchestrator/modules/context/factory.py` | A/B switch via environment variable |
| Coordinator integration | `orchestrator/services/coordinator_service.py` | Mission lifecycle (create/seed/inject/destroy) |
| Agent tool definitions | `orchestrator/modules/tools/discovery/actions_field.py` | 3 ActionDefinitions |
| Agent tool handlers | `orchestrator/modules/tools/discovery/handlers_field.py` | 3 handler functions |
| Configuration | `orchestrator/config.py` | 8 environment variables |

**Strength:** Strong. Code is in a private repository with git history showing authorship and dates.

## Evidence Type 2: Local Tests with Deterministic Embeddings (Moderate)

**What it proves:** The algorithms work correctly. The resonance formula, decay, reinforcement, deduplication, and scoring pipeline produce the expected mathematical results.

**What it does NOT prove:** That the system works with production-grade embeddings or at production scale with real network latency.

**Important caveat:** These tests use bag-of-words deterministic embeddings, not real neural network embeddings. This is intentional — it makes the tests reproducible and isolates the algorithmic behavior from embedding model variability. But it means the A/B comparison results (62% vs 29% coverage) are mechanism tests, not production benchmarks. The embeddings produce predictable cosine similarities based on word overlap, which is sufficient to verify the algorithms but does not represent real-world semantic similarity quality.

| Test Suite | Count | What It Tests | Embedding Type |
|------------|-------|---------------|----------------|
| Unit tests (`test_vector_field.py`) | 57 | Adapter mechanics (mocked Qdrant) | Mocked |
| Proof suite (`test_prd108_proof.py`) | 33 | All claim elements on real Qdrant in-memory | Bag-of-words deterministic |
| Multi-scenario (`test_prd108_scenarios.py`) | 13 | 5 domains, A/B comparison | Bag-of-words deterministic |
| Stress tests (`demo_field_stress.py`) | 16 | Scale, performance, cross-agent visibility | Bag-of-words deterministic |

**Saved outputs:**

| File | Content |
|------|---------|
| `runs/2026-03-21/01-proof-suite.txt` | 33/33 passed in 1.55s |
| `runs/2026-03-21/02-unit-tests.txt` | 57/57 passed in 0.33s |
| `runs/2026-03-21/03-ab-comparison.txt` | VF 86% vs MP 43% (single scenario) |
| `runs/2026-03-21/04-stress-tests.txt` | 16/16 passed, 755 qps |
| `runs/2026-03-21/05-environment.txt` | Python 3.10.9, qdrant-client 1.17.1 |
| `runs/2026-03-21/06-multi-scenario.txt` | 5-scenario summary, VF 62% avg vs MP 29% avg |
| `runs/2026-03-21/07-multi-scenario-pytest.txt` | 13/13 passed in 1.66s |

**Strength:** Moderate. Proves the algorithms work as specified. Does not prove production-grade effectiveness. The distinction must be stated clearly.

## Evidence Type 3: Production Mission (Strong for Reduction to Practice)

**What it proves:** The system runs in production with real agents, real embeddings (Qwen 2048-dim via OpenRouter), and a real vector database (Qdrant on Railway). Three agents shared a field, queried by meaning, and the writer produced output incorporating findings from all research agents.

**What it does NOT prove:** That the vector field systematically outperforms message passing in production. This was a single mission without a controlled baseline comparison on the same prompt.

| Detail | Value |
|--------|-------|
| Mission ID | 77c58227-defb-42c9-b070-c04a1b918764 |
| Field ID | 8bdb19ba-cb03-45d7-a005-3ba04765ad17 |
| Qdrant instance | qdrant-production-c691.up.railway.app |
| Embedding model | qwen/qwen3-embedding-8b (2048 dimensions) |
| Agents | 141 (Researcher), 191 (Writer), 102 (Document Generator) |
| Field queries | 7 (all via platform_field_query) |
| Tasks completed | 6 |
| Total tokens | 158,564 |
| Duration | ~18 minutes (05:59 to 06:17 UTC) |
| Branch | ralph/prd-82b-mission-intelligence |
| Platform | Automatos AI on Railway |

**Evidence source:** Railway deployment logs (timestamped), Qdrant collection creation logs, API logs showing `[PRD-108]` tagged events.

**Strength:** Strong for proving reduction to practice. The system works end-to-end in a real environment. This is the single strongest piece of evidence for a provisional filing.

## Evidence Type 4: Technical Documentation (Supporting)

| Document | Purpose |
|----------|---------|
| `PRD-108-ALGORITHMS.md` | 15 algorithms with math, pseudocode, implementation references |
| `PRD-108-IMPLEMENTATION.md` | Architecture decisions, files changed, engineering narrative |
| `PRD-108-TECHNICAL-DISCLOSURE.md` | Formal disclosure with prior art analysis and 6 novelty claims |

**Strength:** Supporting. These establish that the invention was thoughtfully designed and documented, not accidental.

## What the Evidence Chain Proves (Honestly)

1. **The invention is real.** Working code in a private repository with git timestamps.
2. **The algorithms are correct.** 119 automated assertions verify the mathematical behavior.
3. **The system runs in production.** One live mission with 3 agents, real embeddings, real Qdrant.
4. **The A/B comparison shows a directional advantage.** In controlled local tests with deterministic embeddings, the vector field recovered more information than message passing across 5 scenarios.

## What the Evidence Chain Does NOT Prove

1. **Production superiority.** No controlled A/B comparison has been run in production (same mission, both backends, real embeddings).
2. **Generalizability.** The 5 test scenarios use deterministic embeddings and synthetic data. Real mission performance may vary.
3. **Novelty under formal patent examination.** The inventor's prior art review is preliminary. Counsel should conduct a formal search.
4. **Commercial viability.** Technical evidence only. No market validation data.

## Reproduction Instructions

```bash
cd automatos-ai/orchestrator
pip install qdrant-client pytest pytest-asyncio
python -m pytest tests/test_prd108_proof.py -v      # 33 assertions
python -m pytest tests/test_prd108_scenarios.py -v   # 13 assertions
python -m pytest tests/test_vector_field.py -v       # 57 assertions
```

No external API calls. No credentials needed. Qdrant runs in-memory mode. Fully deterministic.
