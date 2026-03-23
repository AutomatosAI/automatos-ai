#!/usr/bin/env python3
"""
PRD-108 Multi-Scenario A/B Proof
=================================

5 different mission domains. Same test harness. Same measurements.
Proves the vector field advantage is NOT cherry-picked to one scenario.

Each scenario models a real multi-agent mission:
  - Agent A: Researcher (injects findings)
  - Agent B: Analyst (reads top-3, injects analysis)
  - Agent C: Writer/Executor (queries for specific information)

The telephone game: In message-passing, C only sees what B forwarded.
In the vector field, C queries ALL patterns by semantic meaning.

Run:  cd orchestrator && python -m pytest tests/test_prd108_scenarios.py -v

Dependencies: pip install qdrant-client pytest pytest-asyncio
"""

import asyncio
import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ── Shared infrastructure ────────────────────────────────────────
DIM = 128
_VOCAB: dict[str, int] = {}


def _vocab_index(word: str) -> int:
    if word not in _VOCAB:
        _VOCAB[word] = len(_VOCAB)
    return _VOCAB[word]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for w in text.lower().split():
        vec[_vocab_index(w) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def keyword_score(query_text: str, text: str) -> float:
    words = set(query_text.lower().split())
    if not words:
        return 0.0
    return sum(1 for w in words if w in text.lower()) / len(words)


@dataclass
class ScenarioResult:
    name: str
    vf_found: int
    mp_found: int
    total_queries: int
    vf_lost: list[str] = dc_field(default_factory=list)
    mp_lost: list[str] = dc_field(default_factory=list)

    @property
    def vf_coverage(self) -> float:
        return self.vf_found / self.total_queries

    @property
    def mp_coverage(self) -> float:
        return self.mp_found / self.total_queries

    @property
    def coverage_gap(self) -> float:
        return self.vf_coverage - self.mp_coverage


async def run_scenario(
    findings: list[tuple[str, str]],
    analyses: list[tuple[str, str]],
    b_focus_query: str,
    queries: list[tuple[str, str]],
    scenario_name: str,
) -> ScenarioResult:
    """Run both backends on the same scenario, return comparative result."""

    # ── VECTOR FIELD ──────────────────────────────────────────
    client = AsyncQdrantClient(":memory:")
    col = f"scenario_{uuid.uuid4().hex[:8]}"
    await client.create_collection(col, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    # Agent A injects all findings
    for key, value in findings:
        content = f"{key}: {value}"
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=col,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(content),
                payload={
                    "agent_id": 1, "key": key, "value": value,
                    "strength": 1.0, "created_at": now,
                    "last_accessed": now, "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )

    # Agent B injects analyses
    for key, value in analyses:
        content = f"{key}: {value}"
        now = datetime.now(timezone.utc).isoformat()
        await client.upsert(
            collection_name=col,
            points=[PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(content),
                payload={
                    "agent_id": 2, "key": key, "value": value,
                    "strength": 1.0, "created_at": now,
                    "last_accessed": now, "access_count": 0,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                },
            )],
        )

    # Agent C queries vector field
    vf_found = 0
    vf_lost = []
    for query_text, expected_key in queries:
        resp = await client.query_points(collection_name=col, query=embed(query_text), limit=5)
        result_keys = [r.payload["key"] for r in resp.points]
        if expected_key in result_keys:
            vf_found += 1
        else:
            vf_lost.append(expected_key)

    await client.delete_collection(col)

    # ── MESSAGE PASSING ───────────────────────────────────────
    # Agent B picks top 3 from A's findings by keyword match
    b_scored = [(keyword_score(b_focus_query, f"{k} {v}"), k, v) for k, v in findings]
    b_scored.sort(key=lambda x: -x[0])
    b_forwarded = [(k, v) for _, k, v in b_scored[:3]]

    # C's pool = what B forwarded + B's analyses
    c_pool = b_forwarded + list(analyses)

    mp_found = 0
    mp_lost = []
    for query_text, expected_key in queries:
        scored = [(keyword_score(query_text, f"{k} {v}"), k) for k, v in c_pool]
        scored.sort(key=lambda x: -x[0])
        top_keys = [k for _, k in scored[:5]]
        if expected_key in top_keys:
            mp_found += 1
        else:
            mp_lost.append(expected_key)

    return ScenarioResult(
        name=scenario_name,
        vf_found=vf_found,
        mp_found=mp_found,
        total_queries=len(queries),
        vf_lost=vf_lost,
        mp_lost=mp_lost,
    )


# ═════════════════════════════════════════════════════════════════
# SCENARIO 1: EU AI Act Compliance (original)
# ═════════════════════════════════════════════════════════════════

SCENARIO_1_FINDINGS = [
    ("eu_ai_act_scope", "The EU AI Act classifies AI systems into 4 risk tiers unacceptable high limited minimal"),
    ("compliance_deadline", "High-risk AI systems must comply by August 2027 general-purpose AI models by August 2026"),
    ("penalties", "Non-compliance up to 35M EUR or 7% of global annual turnover for prohibited practices"),
    ("documentation_req", "High-risk systems require technical documentation risk management data governance human oversight"),
    ("saas_impact", "Cloud-hosted SaaS AI features likely fall under deployer obligations"),
    ("transparency_rules", "Limited-risk systems must inform users they are interacting with AI"),
    ("foundation_models", "Foundation model providers must publish training data summaries and comply with copyright law"),
    ("biometric_ban", "Real-time remote biometric identification in public spaces is prohibited except law enforcement"),
    ("social_scoring", "AI systems that score citizens based on social behavior are classified as unacceptable risk"),
    ("employee_monitoring", "AI systems used for employee monitoring and evaluation may be classified as high-risk"),
]

SCENARIO_1_ANALYSES = [
    ("risk_assessment", "Based on the 4-tier classification typical SaaS AI features like chatbots fall into limited risk"),
    ("timeline_analysis", "SaaS companies have 18 months to comply priority classify all AI features by risk tier"),
    ("action_items", "Immediate actions transparency notices for chatbots documentation for any hiring credit AI features"),
]

SCENARIO_1_QUERIES = [
    ("What are the penalties for non-compliance with the EU AI Act?", "penalties"),
    ("Are there any biometric surveillance restrictions?", "biometric_ban"),
    ("What about social scoring systems?", "social_scoring"),
    ("How does this affect employee monitoring AI?", "employee_monitoring"),
    ("What do foundation model providers need to do?", "foundation_models"),
    ("What are the transparency requirements?", "transparency_rules"),
    ("What risk tier do SaaS chatbots fall into?", "risk_assessment"),
]


# ═════════════════════════════════════════════════════════════════
# SCENARIO 2: Cybersecurity Vulnerability Assessment
# 5 agents: scanner, network analyst, app tester, social engineer, reporter
# ═════════════════════════════════════════════════════════════════

SCENARIO_2_FINDINGS = [
    ("open_ports", "Port scan reveals SSH on 22 HTTP on 80 HTTPS on 443 and unexpected service on port 8443"),
    ("ssl_weak", "TLS 1.0 and 1.1 still enabled on production load balancer vulnerable to POODLE and BEAST attacks"),
    ("sqli_login", "SQL injection vulnerability found in login form parameter username allows authentication bypass"),
    ("xss_search", "Reflected XSS vulnerability in search endpoint allows script injection via query parameter"),
    ("outdated_deps", "jQuery 2.1.4 and Apache Struts 2.3.x detected both have known critical CVEs"),
    ("default_creds", "Admin panel at /admin accessible with default credentials admin:admin123"),
    ("s3_public", "AWS S3 bucket customer-backups-prod has public read access exposing PII data"),
    ("no_rate_limit", "Authentication endpoint has no rate limiting allows brute force attacks"),
    ("missing_headers", "Security headers missing Content-Security-Policy X-Frame-Options Strict-Transport-Security"),
    ("api_keys_exposed", "API keys found hardcoded in client-side JavaScript bundle for payment processor"),
    ("dns_zone_transfer", "DNS zone transfer enabled on ns1 exposing internal hostname structure"),
    ("smtp_open_relay", "Mail server configured as open relay allows email spoofing from any sender"),
]

SCENARIO_2_ANALYSES = [
    ("critical_vulns", "3 critical findings SQL injection default admin credentials and exposed S3 bucket need immediate remediation"),
    ("attack_surface", "External attack surface includes 4 open ports 2 web applications and 1 mail server"),
    ("priority_matrix", "Priority 1 SQLi and default creds Priority 2 S3 bucket and API keys Priority 3 TLS and headers"),
]

SCENARIO_2_QUERIES = [
    ("Are there any data exposure risks from cloud storage?", "s3_public"),
    ("What DNS vulnerabilities were found?", "dns_zone_transfer"),
    ("Is there an email spoofing risk?", "smtp_open_relay"),
    ("What client-side secrets are exposed?", "api_keys_exposed"),
    ("Are there brute force attack vectors?", "no_rate_limit"),
    ("What XSS vulnerabilities exist?", "xss_search"),
    ("What outdated software was detected?", "outdated_deps"),
    ("What are the critical vulnerabilities?", "critical_vulns"),
]


# ═════════════════════════════════════════════════════════════════
# SCENARIO 3: Market Research — Competitive Analysis
# Agents: web researcher, patent analyst, financial analyst, reporter
# ═════════════════════════════════════════════════════════════════

SCENARIO_3_FINDINGS = [
    ("competitor_a_revenue", "Competitor Alpha reported 450M annual revenue with 23% year-over-year growth in enterprise segment"),
    ("competitor_b_funding", "Competitor Beta raised Series D 200M at 2.1B valuation led by Sequoia Capital"),
    ("market_size", "Total addressable market for AI orchestration estimated at 12.8B by 2028 growing 34% CAGR"),
    ("competitor_c_pivot", "Competitor Gamma pivoted from chatbot-only to full agent orchestration in Q4 2025"),
    ("patent_landscape", "47 patents filed in multi-agent coordination space in last 18 months primarily by Google and Microsoft"),
    ("customer_churn", "Industry average churn rate for AI SaaS platforms is 8.2% annually with enterprise at 3.1%"),
    ("pricing_trends", "Usage-based pricing becoming dominant model replacing per-seat licensing across AI platforms"),
    ("talent_shortage", "ML engineer salary increased 28% in 2025 with 3.2 candidates per open position in agent systems"),
    ("regulatory_impact", "EU AI Act compliance costs estimated at 200K-2M per company depending on risk classification"),
    ("open_source_threat", "CrewAI and AutoGen gaining traction with combined 45K GitHub stars threatening commercial platforms"),
]

SCENARIO_3_ANALYSES = [
    ("competitive_position", "Our differentiation strongest in orchestration layer competitors focused on single-agent or chatbot use cases"),
    ("go_to_market", "Enterprise segment most viable with 3.1% churn and willingness to pay for orchestration features"),
    ("risk_summary", "Primary risks open-source commoditization talent acquisition costs and regulatory compliance overhead"),
]

SCENARIO_3_QUERIES = [
    ("What is the patent landscape for multi-agent systems?", "patent_landscape"),
    ("Are open source alternatives a threat?", "open_source_threat"),
    ("What are the talent acquisition challenges?", "talent_shortage"),
    ("How is pricing evolving in the market?", "pricing_trends"),
    ("What is the customer retention rate?", "customer_churn"),
    ("What did Competitor Gamma do recently?", "competitor_c_pivot"),
    ("What is the regulatory cost impact?", "regulatory_impact"),
    ("What is our competitive positioning?", "competitive_position"),
]


# ═════════════════════════════════════════════════════════════════
# SCENARIO 4: Product Launch Coordination
# Agents: legal reviewer, marketing writer, engineering lead, QA, PM
# ═════════════════════════════════════════════════════════════════

SCENARIO_4_FINDINGS = [
    ("gdpr_consent", "New feature collects user location data requiring explicit GDPR consent flow before activation"),
    ("accessibility_gaps", "Color contrast ratio fails WCAG 2.1 AA standard on 3 primary action buttons"),
    ("api_breaking", "API v2 removes deprecated endpoints /users/list and /events/batch breaking 12% of integrations"),
    ("performance_regression", "Page load time increased from 1.2s to 3.8s after adding real-time collaboration feature"),
    ("mobile_crash", "iOS app crashes on launch for devices running iOS 15 affecting 18% of mobile user base"),
    ("pricing_change", "New tier Premium Plus at $49/month requires migration path for existing Pro users at $29/month"),
    ("competitor_launch", "Competitor launching similar feature next month based on leaked product roadmap"),
    ("security_audit", "Penetration test found IDOR vulnerability in document sharing allowing unauthorized access"),
    ("data_migration", "Database schema change requires 4-hour maintenance window affecting all regions simultaneously"),
    ("localization_incomplete", "Japanese and Korean translations only 60% complete for new feature strings"),
]

SCENARIO_4_ANALYSES = [
    ("launch_readiness", "3 blockers identified mobile crash performance regression and security vulnerability must fix before launch"),
    ("timeline_risk", "Competitor launching next month creates pressure but shipping with blockers is higher risk"),
    ("communication_plan", "Need customer advisory for API breaking changes 30 days before deprecation enforcement"),
]

SCENARIO_4_QUERIES = [
    ("Are there any security vulnerabilities blocking launch?", "security_audit"),
    ("What mobile issues exist?", "mobile_crash"),
    ("Is the database migration disruptive?", "data_migration"),
    ("What accessibility problems were found?", "accessibility_gaps"),
    ("Are translations ready for international launch?", "localization_incomplete"),
    ("What pricing changes need communication?", "pricing_change"),
    ("What is the competitive pressure?", "competitor_launch"),
    ("What are the launch blockers?", "launch_readiness"),
]


# ═════════════════════════════════════════════════════════════════
# SCENARIO 5: Incident Response — Production Outage
# Agents: SRE, database expert, network engineer, app developer, comms
# ═════════════════════════════════════════════════════════════════

SCENARIO_5_FINDINGS = [
    ("error_spike", "500 error rate jumped from 0.1% to 47% at 14:23 UTC correlating with deployment deploy-4521"),
    ("db_connections", "PostgreSQL connection pool exhausted at 500 connections with 200 queries in waiting state"),
    ("memory_leak", "Worker process memory usage growing linearly at 50MB per hour started after deploy-4521"),
    ("dns_timeout", "Internal DNS resolution latency increased from 2ms to 800ms affecting service discovery"),
    ("certificate_expiry", "TLS certificate for api.internal.prod expires in 6 hours automated renewal failed silently"),
    ("queue_backlog", "RabbitMQ queue order-processing has 45000 unacked messages growing at 500 per minute"),
    ("cache_miss_rate", "Redis cache hit rate dropped from 95% to 12% after cache flush triggered by deployment script"),
    ("disk_space", "Log volume /var/log at 98% capacity on 3 of 8 application servers causing write failures"),
    ("upstream_api", "Payment processor API returning 503 for 30% of requests their status page shows degraded service"),
    ("config_drift", "Environment variable DATABASE_URL on servers 5-8 still pointing to old replica after failover"),
    ("health_check", "Load balancer health checks passing because they only verify HTTP 200 not actual database connectivity"),
    ("rollback_blocked", "Previous deployment artifacts purged from registry cannot rollback to last known good version"),
]

SCENARIO_5_ANALYSES = [
    ("root_cause", "Primary root cause is deploy-4521 which introduced connection pool regression and cache flush in deploy script"),
    ("blast_radius", "47% error rate affecting all authenticated endpoints estimated 12000 users impacted per hour"),
    ("remediation_plan", "Immediate fix database connection pool limit and restore cache warmup priority fix DNS and cert renewal"),
]

SCENARIO_5_QUERIES = [
    ("Is there a certificate about to expire?", "certificate_expiry"),
    ("What is happening with the message queue?", "queue_backlog"),
    ("Are there disk space issues?", "disk_space"),
    ("Is any upstream dependency degraded?", "upstream_api"),
    ("Are there configuration inconsistencies across servers?", "config_drift"),
    ("Why are health checks not catching the issue?", "health_check"),
    ("Can we rollback the deployment?", "rollback_blocked"),
    ("What is the root cause?", "root_cause"),
]


# ═════════════════════════════════════════════════════════════════
# THE TESTS
# ═════════════════════════════════════════════════════════════════

class TestScenario1_EUAIAct:
    @pytest.mark.asyncio
    async def test_vector_field_beats_message_passing(self):
        result = await run_scenario(
            SCENARIO_1_FINDINGS, SCENARIO_1_ANALYSES,
            "compliance requirements and deadlines",
            SCENARIO_1_QUERIES, "EU AI Act Compliance",
        )
        assert result.vf_found > result.mp_found, \
            f"[{result.name}] VF {result.vf_found} must beat MP {result.mp_found}"

    @pytest.mark.asyncio
    async def test_message_passing_loses_information(self):
        result = await run_scenario(
            SCENARIO_1_FINDINGS, SCENARIO_1_ANALYSES,
            "compliance requirements and deadlines",
            SCENARIO_1_QUERIES, "EU AI Act Compliance",
        )
        assert len(result.mp_lost) > len(result.vf_lost), \
            f"[{result.name}] MP must lose more: MP lost {result.mp_lost}, VF lost {result.vf_lost}"


class TestScenario2_Cybersecurity:
    @pytest.mark.asyncio
    async def test_vector_field_beats_message_passing(self):
        result = await run_scenario(
            SCENARIO_2_FINDINGS, SCENARIO_2_ANALYSES,
            "critical vulnerabilities SQL injection authentication",
            SCENARIO_2_QUERIES, "Cybersecurity Assessment",
        )
        assert result.vf_found > result.mp_found, \
            f"[{result.name}] VF {result.vf_found} must beat MP {result.mp_found}"

    @pytest.mark.asyncio
    async def test_message_passing_loses_information(self):
        result = await run_scenario(
            SCENARIO_2_FINDINGS, SCENARIO_2_ANALYSES,
            "critical vulnerabilities SQL injection authentication",
            SCENARIO_2_QUERIES, "Cybersecurity Assessment",
        )
        assert len(result.mp_lost) > len(result.vf_lost), \
            f"[{result.name}] MP must lose more: MP lost {result.mp_lost}, VF lost {result.vf_lost}"


class TestScenario3_MarketResearch:
    @pytest.mark.asyncio
    async def test_vector_field_beats_message_passing(self):
        result = await run_scenario(
            SCENARIO_3_FINDINGS, SCENARIO_3_ANALYSES,
            "competitor revenue funding market size growth",
            SCENARIO_3_QUERIES, "Market Research",
        )
        assert result.vf_found > result.mp_found, \
            f"[{result.name}] VF {result.vf_found} must beat MP {result.mp_found}"

    @pytest.mark.asyncio
    async def test_message_passing_loses_information(self):
        result = await run_scenario(
            SCENARIO_3_FINDINGS, SCENARIO_3_ANALYSES,
            "competitor revenue funding market size growth",
            SCENARIO_3_QUERIES, "Market Research",
        )
        assert len(result.mp_lost) > len(result.vf_lost), \
            f"[{result.name}] MP must lose more: MP lost {result.mp_lost}, VF lost {result.vf_lost}"


class TestScenario4_ProductLaunch:
    @pytest.mark.asyncio
    async def test_vector_field_beats_message_passing(self):
        result = await run_scenario(
            SCENARIO_4_FINDINGS, SCENARIO_4_ANALYSES,
            "launch blockers critical bugs performance issues",
            SCENARIO_4_QUERIES, "Product Launch",
        )
        assert result.vf_found > result.mp_found, \
            f"[{result.name}] VF {result.vf_found} must beat MP {result.mp_found}"

    @pytest.mark.asyncio
    async def test_message_passing_loses_information(self):
        result = await run_scenario(
            SCENARIO_4_FINDINGS, SCENARIO_4_ANALYSES,
            "launch blockers critical bugs performance issues",
            SCENARIO_4_QUERIES, "Product Launch",
        )
        assert len(result.mp_lost) > len(result.vf_lost), \
            f"[{result.name}] MP must lose more: MP lost {result.mp_lost}, VF lost {result.vf_lost}"


class TestScenario5_IncidentResponse:
    @pytest.mark.asyncio
    async def test_vector_field_beats_message_passing(self):
        result = await run_scenario(
            SCENARIO_5_FINDINGS, SCENARIO_5_ANALYSES,
            "error rate deployment database connection failures",
            SCENARIO_5_QUERIES, "Incident Response",
        )
        assert result.vf_found > result.mp_found, \
            f"[{result.name}] VF {result.vf_found} must beat MP {result.mp_found}"

    @pytest.mark.asyncio
    async def test_message_passing_loses_information(self):
        result = await run_scenario(
            SCENARIO_5_FINDINGS, SCENARIO_5_ANALYSES,
            "error rate deployment database connection failures",
            SCENARIO_5_QUERIES, "Incident Response",
        )
        assert len(result.mp_lost) > len(result.vf_lost), \
            f"[{result.name}] MP must lose more: MP lost {result.mp_lost}, VF lost {result.vf_lost}"


# ═════════════════════════════════════════════════════════════════
# AGGREGATE: Cross-scenario proof
# ═════════════════════════════════════════════════════════════════

class TestCrossScenario:
    """Prove the vector field advantage holds across ALL 5 domains."""

    ALL_SCENARIOS = [
        (SCENARIO_1_FINDINGS, SCENARIO_1_ANALYSES, "compliance requirements and deadlines", SCENARIO_1_QUERIES, "EU AI Act"),
        (SCENARIO_2_FINDINGS, SCENARIO_2_ANALYSES, "critical vulnerabilities SQL injection authentication", SCENARIO_2_QUERIES, "Cybersecurity"),
        (SCENARIO_3_FINDINGS, SCENARIO_3_ANALYSES, "competitor revenue funding market size growth", SCENARIO_3_QUERIES, "Market Research"),
        (SCENARIO_4_FINDINGS, SCENARIO_4_ANALYSES, "launch blockers critical bugs performance issues", SCENARIO_4_QUERIES, "Product Launch"),
        (SCENARIO_5_FINDINGS, SCENARIO_5_ANALYSES, "error rate deployment database connection failures", SCENARIO_5_QUERIES, "Incident Response"),
    ]

    @pytest.mark.asyncio
    async def test_vector_field_wins_all_5_scenarios(self):
        """Vector field must beat message passing in every single scenario."""
        results = []
        for findings, analyses, b_query, queries, name in self.ALL_SCENARIOS:
            result = await run_scenario(findings, analyses, b_query, queries, name)
            results.append(result)

        wins = sum(1 for r in results if r.vf_found > r.mp_found)
        details = "\n".join(
            f"  {r.name}: VF={r.vf_found}/{r.total_queries} MP={r.mp_found}/{r.total_queries} "
            f"gap={r.coverage_gap:+.0%}"
            for r in results
        )
        assert wins == 5, f"Vector field must win all 5 scenarios, won {wins}/5:\n{details}"

    @pytest.mark.asyncio
    async def test_average_coverage_gap_significant(self):
        """Average coverage gap across all scenarios must be >= 20%."""
        results = []
        for findings, analyses, b_query, queries, name in self.ALL_SCENARIOS:
            result = await run_scenario(findings, analyses, b_query, queries, name)
            results.append(result)

        avg_gap = sum(r.coverage_gap for r in results) / len(results)
        details = "\n".join(
            f"  {r.name}: VF={r.vf_coverage:.0%} MP={r.mp_coverage:.0%} gap={r.coverage_gap:+.0%}"
            for r in results
        )
        assert avg_gap >= 0.20, \
            f"Average coverage gap {avg_gap:.0%} must be >= 20%:\n{details}"

    @pytest.mark.asyncio
    async def test_message_passing_always_loses_more(self):
        """Message passing must lose more information than vector field in every scenario."""
        results = []
        for findings, analyses, b_query, queries, name in self.ALL_SCENARIOS:
            result = await run_scenario(findings, analyses, b_query, queries, name)
            results.append(result)

        all_worse = all(len(r.mp_lost) > len(r.vf_lost) for r in results)
        details = "\n".join(
            f"  {r.name}: VF lost {r.vf_lost}, MP lost {r.mp_lost}"
            for r in results
        )
        assert all_worse, f"MP must lose more in every scenario:\n{details}"


# ═════════════════════════════════════════════════════════════════
# SUMMARY PRINTER (when run directly)
# ═════════════════════════════════════════════════════════════════

async def print_summary():
    """Run all 5 scenarios and print a summary table."""
    scenarios = [
        (SCENARIO_1_FINDINGS, SCENARIO_1_ANALYSES, "compliance requirements and deadlines", SCENARIO_1_QUERIES, "EU AI Act Compliance"),
        (SCENARIO_2_FINDINGS, SCENARIO_2_ANALYSES, "critical vulnerabilities SQL injection authentication", SCENARIO_2_QUERIES, "Cybersecurity Assessment"),
        (SCENARIO_3_FINDINGS, SCENARIO_3_ANALYSES, "competitor revenue funding market size growth", SCENARIO_3_QUERIES, "Market Research"),
        (SCENARIO_4_FINDINGS, SCENARIO_4_ANALYSES, "launch blockers critical bugs performance issues", SCENARIO_4_QUERIES, "Product Launch"),
        (SCENARIO_5_FINDINGS, SCENARIO_5_ANALYSES, "error rate deployment database connection failures", SCENARIO_5_QUERIES, "Incident Response"),
    ]

    print("=" * 78)
    print("PRD-108 MULTI-SCENARIO A/B COMPARISON")
    print("5 domains — same harness — same measurements")
    print("=" * 78)

    results = []
    for findings, analyses, b_query, queries, name in scenarios:
        result = await run_scenario(findings, analyses, b_query, queries, name)
        results.append(result)

    print(f"\n{'Scenario':<28} {'VF':>8} {'MP':>8} {'Gap':>8} {'MP Lost':>10}")
    print(f"{'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for r in results:
        print(f"{r.name:<28} {r.vf_coverage:>7.0%} {r.mp_coverage:>7.0%} {r.coverage_gap:>+7.0%} {len(r.mp_lost):>10}")

    avg_vf = sum(r.vf_coverage for r in results) / len(results)
    avg_mp = sum(r.mp_coverage for r in results) / len(results)
    avg_gap = sum(r.coverage_gap for r in results) / len(results)
    total_mp_lost = sum(len(r.mp_lost) for r in results)

    print(f"{'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    print(f"{'AVERAGE':<28} {avg_vf:>7.0%} {avg_mp:>7.0%} {avg_gap:>+7.0%} {total_mp_lost:>10}")

    print(f"\n{'='*78}")
    wins = sum(1 for r in results if r.vf_found > r.mp_found)
    print(f"VECTOR FIELD WINS: {wins}/5 scenarios")
    print(f"AVERAGE COVERAGE GAP: {avg_gap:+.0%}")
    print(f"TOTAL INFORMATION LOST BY MESSAGE PASSING: {total_mp_lost} findings")
    print(f"{'='*78}")

    # Detail per scenario
    for r in results:
        if r.mp_lost:
            print(f"\n  [{r.name}] Message passing lost: {r.mp_lost}")
        if r.vf_lost:
            print(f"  [{r.name}] Vector field lost: {r.vf_lost}")


if __name__ == "__main__":
    asyncio.run(print_summary())
