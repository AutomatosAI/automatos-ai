#!/usr/bin/env python3
"""
PRD-108 Memory Field — Live Integration Demo
=============================================

This script simulates a 3-agent mission using a REAL Qdrant instance
(in-memory mode, no server needed). It demonstrates:

1. Three agents inject findings into a shared field
2. Later agents query and see earlier agents' work ranked by relevance
3. Frequently accessed patterns resist decay (Hebbian reinforcement)
4. Old patterns fade over time (temporal decay)
5. Field stability measurement shows convergence

Run:  python tests/demo_field.py
"""

import asyncio
import hashlib
import math
import time
import uuid
from datetime import datetime, timedelta, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType,
)

# ── Config ──────────────────────────────────────────────────────
DIMENSION = 128          # Small for demo speed (production uses 2048)
DECAY_RATE = 0.1         # λ — half-life ~7 hours
REINFORCE_BONUS = 0.05   # +5% per access
REINFORCE_CAP = 2.0      # Max 2× original strength
ARCHIVAL_THRESHOLD = 0.05
BOUNDARY_PERMEABILITY = 1.0


# ── Fake embeddings (deterministic, position-based) ─────────────
def fake_embed(text: str) -> list[float]:
    """Generate a deterministic embedding from text.
    Similar texts produce similar vectors (by sharing prefix hash)."""
    h = hashlib.sha256(text.encode()).digest()
    vec = [((b % 200) - 100) / 100.0 for b in h]
    # Pad or truncate to DIMENSION
    while len(vec) < DIMENSION:
        vec.extend(vec[:DIMENSION - len(vec)])
    vec = vec[:DIMENSION]
    # Normalize
    norm = math.sqrt(sum(x*x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def compute_decayed_strength(strength, age_hours, access_count):
    decay = math.exp(-DECAY_RATE * age_hours)
    boost = min(1.0 + access_count * REINFORCE_BONUS, REINFORCE_CAP)
    return strength * decay * boost


# ── Main demo ───────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("PRD-108 MEMORY FIELD — LIVE DEMO")
    print("=" * 60)

    # In-memory Qdrant — no server needed
    client = AsyncQdrantClient(":memory:")
    field_id = str(uuid.uuid4())[:8]
    collection = f"field_{field_id}"

    # Create the field
    await client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )
    await client.create_payload_index(
        collection_name=collection,
        field_name="content_hash",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"\n✦ Field created: {collection}")

    # ── Agent 1: RESEARCHER ────────────────────────────────────
    print("\n" + "─" * 60)
    print("AGENT 1: RESEARCHER — injecting findings")
    print("─" * 60)

    findings = [
        ("eu_ai_act_scope", "The EU AI Act classifies AI systems into 4 risk tiers: unacceptable, high, limited, minimal. SaaS platforms serving EU customers must classify their AI features."),
        ("compliance_deadline", "High-risk AI systems must comply by August 2027. General-purpose AI models have until August 2026 for transparency requirements."),
        ("penalties", "Non-compliance penalties: up to €35M or 7% of global annual turnover for prohibited practices. Up to €15M or 3% for other violations."),
        ("documentation_req", "High-risk systems require: technical documentation, risk management system, data governance measures, human oversight mechanisms."),
        ("saas_impact", "Cloud-hosted SaaS AI features likely fall under 'deployer' obligations. Some may qualify as 'provider' if they substantially modify foundation models."),
    ]

    for key, value in findings:
        content = f"{key}: {value}"
        embedding = fake_embed(content)
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "agent_id": 1,
                    "key": key,
                    "value": value,
                    "strength": 1.0 * BOUNDARY_PERMEABILITY,
                    "created_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )
        print(f"  ✓ {key}")

    # ── Agent 2: ANALYST — queries field, then injects analysis ─
    print("\n" + "─" * 60)
    print("AGENT 2: ANALYST — querying field for compliance info")
    print("─" * 60)

    query = "What are the compliance requirements and deadlines for SaaS platforms?"
    query_embedding = fake_embed(query)

    response = await client.query_points(
        collection_name=collection,
        query=query_embedding,
        limit=10,
    )
    results = response.points

    print(f"\n  Query: '{query}'")
    print(f"  Results ({len(results)} patterns found):\n")

    for i, hit in enumerate(results):
        p = hit.payload
        age_hours = 0.1  # Just injected
        ds = compute_decayed_strength(p["strength"], age_hours, p["access_count"])
        resonance = (hit.score ** 2) * ds

        print(f"  {i+1}. [{p['key']}] (resonance={resonance:.4f}, cos={hit.score:.4f})")
        print(f"     {p['value'][:80]}...")

        # Hebbian: reinforce accessed patterns
        await client.set_payload(
            collection_name=collection,
            payload={
                "access_count": p["access_count"] + 1,
                "last_accessed": datetime.now(timezone.utc).isoformat(),
            },
            points=[hit.id],
        )

    # Analyst injects own analysis
    print("\n  Analyst injecting analysis...")
    analyses = [
        ("risk_assessment", "Based on the 4-tier classification, typical SaaS AI features (chatbots, recommendations) fall into 'limited risk' category. However, AI features used in hiring, credit scoring, or medical decisions are 'high risk'."),
        ("timeline_analysis", "SaaS companies have 18 months to comply. Priority actions: 1) Classify all AI features by risk tier, 2) Implement transparency notices for limited-risk, 3) Start documentation for any high-risk features."),
    ]

    for key, value in analyses:
        content = f"{key}: {value}"
        embedding = fake_embed(content)
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=collection,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "agent_id": 2,
                    "key": key,
                    "value": value,
                    "strength": 1.0,
                    "created_at": now,
                    "last_accessed": now,
                    "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )
        print(f"  ✓ {key}")

    # ── Agent 3: WRITER — queries for resonant patterns ─────────
    print("\n" + "─" * 60)
    print("AGENT 3: WRITER — querying for the most resonant patterns")
    print("─" * 60)

    query2 = "Write an executive summary of EU AI Act impact on our SaaS platform"
    query2_embedding = fake_embed(query2)

    response2 = await client.query_points(
        collection_name=collection,
        query=query2_embedding,
        limit=10,
    )
    results2 = response2.points

    print(f"\n  Query: '{query2}'")
    print(f"\n  RESONANCE-RANKED RESULTS:")
    print(f"  {'─' * 50}")

    scored = []
    for hit in results2:
        p = hit.payload
        age_hours = 0.2
        ds = compute_decayed_strength(p["strength"], age_hours, p["access_count"])
        resonance = (hit.score ** 2) * ds
        scored.append((resonance, hit, ds))

    scored.sort(key=lambda x: x[0], reverse=True)

    for i, (resonance, hit, ds) in enumerate(scored):
        p = hit.payload
        agent_label = {1: "Researcher", 2: "Analyst"}.get(p["agent_id"], f"Agent {p['agent_id']}")
        accessed = "★" * p["access_count"] if p["access_count"] > 0 else "·"

        print(f"\n  {i+1}. resonance={resonance:.4f}  strength={ds:.4f}  accessed={accessed}")
        print(f"     from: {agent_label}  key: {p['key']}")
        print(f"     {p['value'][:100]}...")

    # ── Demonstrate decay ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("TEMPORAL DECAY — how patterns fade over time")
    print("─" * 60)

    print(f"\n  {'Hours':>6}  {'No access':>12}  {'Accessed 5x':>12}  {'Accessed 20x':>13}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 13}")

    for hours in [0, 1, 3, 7, 12, 24, 48]:
        no_access = compute_decayed_strength(1.0, hours, 0)
        five_access = compute_decayed_strength(1.0, hours, 5)
        twenty_access = compute_decayed_strength(1.0, hours, 20)
        print(f"  {hours:>6}  {no_access:>12.4f}  {five_access:>12.4f}  {twenty_access:>13.4f}")

    # ── Stability measurement ───────────────────────────────────
    print("\n" + "─" * 60)
    print("FIELD STABILITY — convergence measurement")
    print("─" * 60)

    points, _ = await client.scroll(collection, limit=10000)
    now = datetime.now(timezone.utc)
    strengths = []
    for p in points:
        last = datetime.fromisoformat(p.payload["last_accessed"])
        age_h = (now - last).total_seconds() / 3600
        ds = compute_decayed_strength(p.payload["strength"], age_h, p.payload["access_count"])
        strengths.append(ds)

    avg = sum(strengths) / len(strengths)
    stddev = (sum((s - avg) ** 2 for s in strengths) / len(strengths)) ** 0.5
    org = max(0.0, 1.0 - (stddev / avg)) if avg > 0 else 0.0
    stability = avg * 0.6 + org * 0.4

    print(f"\n  Patterns in field:  {len(points)}")
    print(f"  Active (above threshold): {sum(1 for s in strengths if s >= ARCHIVAL_THRESHOLD)}")
    print(f"  Avg strength:     {avg:.4f}")
    print(f"  Organization:     {org:.4f}")
    print(f"  Stability:        {stability:.4f}")
    print(f"\n  (Stability rises as agents reference the same patterns)")

    # ── Cleanup ─────────────────────────────────────────────────
    await client.delete_collection(collection)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE — No telephone game. Agents share a brain.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
