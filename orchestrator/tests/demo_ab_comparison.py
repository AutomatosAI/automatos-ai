#!/usr/bin/env python3
"""
PRD-108 A/B Comparison — Vector Field vs Redis Baseline
========================================================

Same 3-agent mission scenario, run against BOTH backends.
Measures: context coverage, information loss, query quality, latency.

This is the proof. Same inputs. Different coordination. Measured results.

Run:  python tests/demo_ab_comparison.py
"""

import asyncio
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ──────────────────────────────────────────────────────
DIM = 128
DECAY_RATE = 0.1
REINFORCE_BONUS = 0.05
REINFORCE_CAP = 2.0
ARCHIVAL_THRESHOLD = 0.05


# ── Semantic-aware fake embeddings ──────────────────────────────
# Uses word-overlap bags instead of SHA-256 hashes.
# Similar texts produce similar vectors because they share words.
_VOCAB: dict[str, int] = {}


def _vocab_index(word: str) -> int:
    if word not in _VOCAB:
        _VOCAB[word] = len(_VOCAB)
    return _VOCAB[word]


def fake_embed(text: str) -> list[float]:
    """Generate embeddings where semantic similarity tracks word overlap.

    Words map to fixed dimension slots. Texts sharing words have
    high cosine similarity. This lets us test semantic retrieval
    without real embeddings.
    """
    vec = [0.0] * DIM
    words = text.lower().split()
    for w in words:
        idx = _vocab_index(w) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def decay_strength(s, hours, accesses):
    return s * math.exp(-DECAY_RATE * hours) * min(1.0 + accesses * REINFORCE_BONUS, REINFORCE_CAP)


# ── Metric tracking ─────────────────────────────────────────────
@dataclass
class RunMetrics:
    backend: str
    inject_count: int = 0
    query_count: int = 0
    total_results: int = 0
    total_latency_ms: float = 0.0
    patterns_available: int = 0
    patterns_found_by_agent_c: list = field(default_factory=list)
    telephone_game_loss: list = field(default_factory=list)  # Findings Agent B skipped that C needed


# ═════════════════════════════════════════════════════════════════
# VECTOR FIELD BACKEND (real Qdrant, resonance scoring)
# ═════════════════════════════════════════════════════════════════
async def run_vector_field(findings, analyses, agent_c_queries):
    metrics = RunMetrics(backend="vector_field")
    client = AsyncQdrantClient(":memory:")
    col = f"field_{uuid.uuid4().hex[:8]}"
    await client.create_collection(col, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    # Agent A: Researcher injects findings
    for key, value in findings:
        content = f"{key}: {value}"
        embedding = fake_embed(content)
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=col,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "agent_id": 1, "key": key, "value": value,
                    "strength": 1.0, "created_at": now,
                    "last_accessed": now, "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )
        metrics.inject_count += 1

    # Agent B: Analyst queries top 3, injects analysis
    embedding = fake_embed("compliance requirements and deadlines")
    resp = await client.query_points(collection_name=col, query=embedding, limit=3)
    b_accessed_keys = [r.payload["key"] for r in resp.points]

    for key, value in analyses:
        content = f"{key}: {value}"
        embedding = fake_embed(content)
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=col,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "agent_id": 2, "key": key, "value": value,
                    "strength": 1.0, "created_at": now,
                    "last_accessed": now, "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )
        metrics.inject_count += 1

    metrics.patterns_available = (await client.count(col)).count

    # Agent C: Writer queries — can it find what B skipped?
    for query_text, expected_key in agent_c_queries:
        start = time.monotonic()
        embedding = fake_embed(query_text)
        resp = await client.query_points(collection_name=col, query=embedding, limit=5)
        elapsed = (time.monotonic() - start) * 1000
        metrics.query_count += 1
        metrics.total_latency_ms += elapsed

        result_keys = [r.payload["key"] for r in resp.points]
        metrics.total_results += len(result_keys)
        found = expected_key in result_keys
        metrics.patterns_found_by_agent_c.append({
            "query": query_text,
            "expected": expected_key,
            "found": found,
            "results": result_keys,
            "was_in_b_context": expected_key in b_accessed_keys,
        })
        if not found:
            metrics.telephone_game_loss.append(expected_key)

    await client.delete_collection(col)
    return metrics


# ═════════════════════════════════════════════════════════════════
# REDIS BASELINE (keyword matching, no embeddings)
# ═════════════════════════════════════════════════════════════════
def keyword_score(query: str, text: str) -> float:
    words = set(query.lower().split())
    if not words:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for w in words if w in text_lower)
    return hits / len(words)


async def run_redis_baseline(findings, analyses, agent_c_queries):
    """Simulates message-passing baseline — THE TELEPHONE GAME.

    This models how real sequential multi-agent pipelines work:
    - Agent A produces output (all 10 findings)
    - Agent B receives A's output, processes it, produces its own output
    - Agent C receives ONLY B's output — NOT A's original findings
    - If B didn't mention finding #7, C can't see it

    This is the fundamental limitation of message passing.
    """
    metrics = RunMetrics(backend="redis_message_passing")

    # Agent A: Researcher produces findings
    a_output = findings[:]
    metrics.inject_count += len(a_output)

    # Agent B: Analyst receives A's output, references top 3 by keyword match
    b_query = "compliance requirements and deadlines"
    b_scored = []
    for key, value in a_output:
        score = keyword_score(b_query, f"{key} {value}")
        b_scored.append((score, key, value))
    b_scored.sort(key=lambda x: -x[0])

    # B only references top 3 findings in its output
    b_referenced_keys = [key for _, key, _ in b_scored[:3]]
    b_referenced_findings = [(key, value) for _, key, value in b_scored[:3]]

    # B's output to C = B's analyses + the findings B actually mentioned
    c_available_patterns: list[dict] = []
    for key, value in b_referenced_findings:
        c_available_patterns.append({"agent_id": 1, "key": key, "value": value})
    for key, value in analyses:
        c_available_patterns.append({"agent_id": 2, "key": key, "value": value})
        metrics.inject_count += 1

    metrics.patterns_available = len(c_available_patterns)  # C only sees B's output

    # Agent C: Writer queries ONLY what B forwarded
    for query_text, expected_key in agent_c_queries:
        start = time.monotonic()
        scored = []
        for p in c_available_patterns:
            score = keyword_score(query_text, f"{p['key']} {p['value']}")
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        top_results = scored[:5]
        elapsed = (time.monotonic() - start) * 1000
        metrics.query_count += 1
        metrics.total_latency_ms += elapsed

        result_keys = [p["key"] for _, p in top_results]
        metrics.total_results += len(result_keys)
        found = expected_key in result_keys
        metrics.patterns_found_by_agent_c.append({
            "query": query_text,
            "expected": expected_key,
            "found": found,
            "results": result_keys,
            "was_in_b_context": expected_key in b_referenced_keys,
        })
        if not found:
            metrics.telephone_game_loss.append(expected_key)

    return metrics


# ═════════════════════════════════════════════════════════════════
# THE EXPERIMENT
# ═════════════════════════════════════════════════════════════════
async def main():
    print("=" * 70)
    print("PRD-108 A/B EXPERIMENT — VECTOR FIELD vs REDIS BASELINE")
    print("=" * 70)

    # ── Same inputs for both backends ─────────────────────────
    findings = [
        ("eu_ai_act_scope", "The EU AI Act classifies AI systems into 4 risk tiers: unacceptable, high, limited, minimal"),
        ("compliance_deadline", "High-risk AI systems must comply by August 2027. General-purpose AI models by August 2026"),
        ("penalties", "Non-compliance: up to 35M EUR or 7% of global annual turnover for prohibited practices"),
        ("documentation_req", "High-risk systems require technical documentation, risk management, data governance, human oversight"),
        ("saas_impact", "Cloud-hosted SaaS AI features likely fall under deployer obligations"),
        ("transparency_rules", "Limited-risk systems must inform users they are interacting with AI"),
        ("foundation_models", "Foundation model providers must publish training data summaries and comply with copyright law"),
        ("biometric_ban", "Real-time remote biometric identification in public spaces is prohibited except for specific law enforcement"),
        ("social_scoring", "AI systems that score citizens based on social behavior are classified as unacceptable risk"),
        ("employee_monitoring", "AI systems used for employee monitoring and evaluation may be classified as high-risk"),
    ]

    analyses = [
        ("risk_assessment", "Based on the 4-tier classification, typical SaaS AI features like chatbots fall into limited risk"),
        ("timeline_analysis", "SaaS companies have 18 months to comply. Priority: classify all AI features by risk tier"),
        ("action_items", "Immediate actions: transparency notices for chatbots, documentation for any hiring/credit AI features"),
    ]

    # Agent C queries — deliberately asking about things Agent B DIDN'T reference
    agent_c_queries = [
        ("What are the penalties for non-compliance with the EU AI Act?", "penalties"),
        ("Are there any biometric surveillance restrictions?", "biometric_ban"),
        ("What about social scoring systems?", "social_scoring"),
        ("How does this affect employee monitoring AI?", "employee_monitoring"),
        ("What do foundation model providers need to do?", "foundation_models"),
        ("What are the transparency requirements?", "transparency_rules"),
        ("What risk tier do SaaS chatbots fall into?", "risk_assessment"),
    ]

    # ── Run both backends ─────────────────────────────────────
    print("\n" + "-" * 70)
    print("Running Vector Field backend...")
    print("-" * 70)
    vf_metrics = await run_vector_field(findings, analyses, agent_c_queries)

    print("Running Redis Baseline backend...")
    print("-" * 70)
    redis_metrics = await run_redis_baseline(findings, analyses, agent_c_queries)

    # ── RESULTS ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Vector Field':>15} {'Redis Baseline':>15}")
    print(f"{'-'*35} {'-'*15} {'-'*15}")
    print(f"{'Patterns injected':<35} {vf_metrics.inject_count:>15} {redis_metrics.inject_count:>15}")
    print(f"{'Patterns available':<35} {vf_metrics.patterns_available:>15} {redis_metrics.patterns_available:>15}")
    print(f"{'Queries executed':<35} {vf_metrics.query_count:>15} {redis_metrics.query_count:>15}")
    print(f"{'Total results returned':<35} {vf_metrics.total_results:>15} {redis_metrics.total_results:>15}")

    avg_latency_vf = vf_metrics.total_latency_ms / max(vf_metrics.query_count, 1)
    avg_latency_redis = redis_metrics.total_latency_ms / max(redis_metrics.query_count, 1)
    print(f"{'Avg query latency':<35} {avg_latency_vf:>14.2f}ms {avg_latency_redis:>14.2f}ms")

    # ── CONTEXT COVERAGE ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONTEXT COVERAGE — Did Agent C find what it needed?")
    print(f"{'='*70}")

    vf_found = sum(1 for r in vf_metrics.patterns_found_by_agent_c if r["found"])
    redis_found = sum(1 for r in redis_metrics.patterns_found_by_agent_c if r["found"])
    total_queries = len(agent_c_queries)

    print(f"\n{'Query':<50} {'VF':>5} {'Redis':>7}")
    print(f"{'-'*50} {'-'*5} {'-'*7}")

    for vf_r, redis_r in zip(vf_metrics.patterns_found_by_agent_c, redis_metrics.patterns_found_by_agent_c):
        vf_mark = "FOUND" if vf_r["found"] else "MISS"
        redis_mark = "FOUND" if redis_r["found"] else "MISS"
        query_short = vf_r["query"][:48]
        print(f"  {query_short:<48} {vf_mark:>5} {redis_mark:>7}")

    print(f"\n  Vector Field: {vf_found}/{total_queries} found ({vf_found/total_queries*100:.0f}%)")
    print(f"  Redis:        {redis_found}/{total_queries} found ({redis_found/total_queries*100:.0f}%)")

    # ── INFORMATION LOSS (The Telephone Game Metric) ──────────
    print(f"\n{'='*70}")
    print("INFORMATION LOSS — The Telephone Game Metric")
    print(f"{'='*70}")

    vf_loss = len(vf_metrics.telephone_game_loss)
    redis_loss = len(redis_metrics.telephone_game_loss)

    print(f"\n  Vector Field lost: {vf_loss}/{total_queries} queries")
    if vf_metrics.telephone_game_loss:
        print(f"    Missing: {vf_metrics.telephone_game_loss}")

    print(f"  Redis lost:        {redis_loss}/{total_queries} queries")
    if redis_metrics.telephone_game_loss:
        print(f"    Missing: {redis_metrics.telephone_game_loss}")

    # ── VERDICT ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    coverage_delta = vf_found - redis_found
    loss_delta = redis_loss - vf_loss

    if coverage_delta > 0:
        print(f"\n  VECTOR FIELD WINS on context coverage: +{coverage_delta} more queries answered")
    elif coverage_delta < 0:
        print(f"\n  REDIS WINS on context coverage: +{-coverage_delta} more queries answered")
    else:
        print(f"\n  TIE on context coverage")

    if loss_delta > 0:
        print(f"  VECTOR FIELD WINS on information loss: {loss_delta} fewer findings lost")
    elif loss_delta < 0:
        print(f"  REDIS WINS on information loss: {-loss_delta} fewer findings lost")
    else:
        print(f"  TIE on information loss")

    print(f"\n  Vector Field: Agent C queried {vf_metrics.patterns_available} patterns (ALL agent contributions)")
    print(f"  Message Pass: Agent C could only see {redis_metrics.patterns_available} patterns (only what B forwarded)")
    print(f"  B referenced: {13 - (redis_metrics.patterns_available - len(analyses))} of {len(findings)} research findings — the rest were LOST")
    print(f"\n  The telephone game cost message-passing {redis_loss} missed findings.")
    print(f"  The vector field found {vf_found}/{total_queries} by semantic meaning alone.")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
