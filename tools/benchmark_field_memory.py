#!/usr/bin/env python3
"""
Field Memory Benchmark — Honest A/B test of shared semantic fields
vs message-passing for multi-agent context coverage.

Standalone script. Calls the platform API externally.
Real agents, real LLM calls, real Qdrant. No synthetic embeddings.

Usage:
  # 1. Run with vector_field backend (default)
  python tools/benchmark_field_memory.py --trials 5 --mode parallel

  # 2. Switch backend on Railway:
  #    railway variables set SHARED_CONTEXT_BACKEND=redis
  #    Wait for redeploy

  # 3. Run with redis backend
  python tools/benchmark_field_memory.py --trials 5 --mode parallel --label redis

  # 4. Compare the two JSON result files
  python tools/compare_benchmarks.py tools/benchmark_results/

Modes:
  sequential  — 3-phase pipeline: Research → Analysis → Synthesis (original)
  parallel    — 4 parallel research agents + synthesis (stresses shared memory)

Environment variables:
  AUTOMATOS_API_URL    - Platform API URL (default: http://localhost:8000)
  AUTOMATOS_AUTH_TOKEN - API key or Clerk Bearer token
  AUTOMATOS_WORKSPACE  - Workspace UUID
  OPENROUTER_API_KEY   - For LLM judge scoring (optional, falls back to keyword)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Seed facts — specific, verifiable, varied difficulty
# ---------------------------------------------------------------------------
# Easy: high keyword overlap with likely queries
# Medium: partial overlap, requires some inference
# Hard: semantic-only, no keyword overlap with obvious queries

SEED_FACTS = [
    # --- EU AI Act (domain 1) ---
    {
        "id": "eu1",
        "difficulty": "easy",
        "domain": "EU AI Act",
        "fact": "The EU AI Act classifies AI systems into four risk tiers: unacceptable, high, limited, and minimal risk.",
    },
    {
        "id": "eu2",
        "difficulty": "medium",
        "domain": "EU AI Act",
        "fact": "High-risk AI systems must maintain detailed technical documentation and undergo conformity assessments before deployment in the EU market.",
    },
    {
        "id": "eu3",
        "difficulty": "hard",
        "domain": "EU AI Act",
        "fact": "Real-time biometric identification in public spaces is prohibited except for three narrow law enforcement exceptions including missing children and imminent terrorist threats.",
    },
    {
        "id": "eu4",
        "difficulty": "medium",
        "domain": "EU AI Act",
        "fact": "The EU AI Act imposes fines of up to 35 million euros or 7% of global annual turnover for violations involving prohibited AI practices.",
    },
    {
        "id": "eu5",
        "difficulty": "hard",
        "domain": "EU AI Act",
        "fact": "AI-generated deepfakes must be clearly labeled as artificially generated content under the EU AI Act transparency obligations, with specific technical standards defined by ENISA.",
    },
    # --- Cybersecurity (domain 2) ---
    {
        "id": "cyber1",
        "difficulty": "easy",
        "domain": "Cybersecurity",
        "fact": "Zero-trust architecture requires continuous verification of every user and device, eliminating implicit trust based on network location.",
    },
    {
        "id": "cyber2",
        "difficulty": "medium",
        "domain": "Cybersecurity",
        "fact": "The MITRE ATT&CK framework catalogs 14 tactical categories of adversary behavior from initial access through exfiltration and impact.",
    },
    {
        "id": "cyber3",
        "difficulty": "hard",
        "domain": "Cybersecurity",
        "fact": "Supply chain attacks increased 742% between 2019 and 2022, with the SolarWinds Orion compromise affecting over 18,000 organizations including US federal agencies.",
    },
    {
        "id": "cyber4",
        "difficulty": "easy",
        "domain": "Cybersecurity",
        "fact": "The NIST Cybersecurity Framework 2.0 added a sixth function called 'Govern' alongside the original five: Identify, Protect, Detect, Respond, and Recover.",
    },
    {
        "id": "cyber5",
        "difficulty": "hard",
        "domain": "Cybersecurity",
        "fact": "Ransomware payments reached $1.1 billion globally in 2023, with the average enterprise ransom demand exceeding $5.3 million according to Chainalysis.",
    },
    # --- Market Research (domain 3) ---
    {
        "id": "mkt1",
        "difficulty": "easy",
        "domain": "Market Research",
        "fact": "The global AI market is projected to reach $1.8 trillion by 2030, growing at a CAGR of approximately 37% from 2023.",
    },
    {
        "id": "mkt2",
        "difficulty": "medium",
        "domain": "Market Research",
        "fact": "Enterprise adoption of multi-agent AI systems is concentrated in financial services, healthcare, and manufacturing, with financial services leading at 34% adoption.",
    },
    {
        "id": "mkt3",
        "difficulty": "hard",
        "domain": "Market Research",
        "fact": "Companies deploying AI agents report a median 23% reduction in operational costs but a 40% increase in infrastructure spending during the first 18 months of adoption.",
    },
    {
        "id": "mkt4",
        "difficulty": "medium",
        "domain": "Market Research",
        "fact": "McKinsey estimates generative AI could add $2.6 to $4.4 trillion annually in value across 63 enterprise use cases, with banking and retail capturing the largest share.",
    },
    {
        "id": "mkt5",
        "difficulty": "hard",
        "domain": "Market Research",
        "fact": "Only 11% of enterprises have moved beyond pilot stage with multi-agent AI deployments, with 67% citing integration complexity and 54% citing governance concerns as primary barriers.",
    },
    # --- Incident Response (domain 4) ---
    {
        "id": "ir1",
        "difficulty": "easy",
        "domain": "Incident Response",
        "fact": "The NIST incident response lifecycle consists of four phases: preparation, detection and analysis, containment eradication and recovery, and post-incident activity.",
    },
    {
        "id": "ir2",
        "difficulty": "medium",
        "domain": "Incident Response",
        "fact": "Mean time to identify a data breach in 2023 was 204 days, with an additional 73 days average to contain it, according to IBM's Cost of a Data Breach Report.",
    },
    {
        "id": "ir3",
        "difficulty": "hard",
        "domain": "Incident Response",
        "fact": "Organizations with an incident response team and regularly tested IR plans saved an average of $2.66 million per breach compared to those without either measure.",
    },
    {
        "id": "ir4",
        "difficulty": "easy",
        "domain": "Incident Response",
        "fact": "The average total cost of a data breach in 2023 was $4.45 million globally, with healthcare breaches averaging $10.93 million, the highest of any industry.",
    },
    {
        "id": "ir5",
        "difficulty": "hard",
        "domain": "Incident Response",
        "fact": "AI and automation in incident response reduced breach costs by an average of $1.76 million and shortened the breach lifecycle by 108 days compared to organizations without these capabilities.",
    },
    # --- Enterprise AI Governance (domain 5 — noise domain) ---
    {
        "id": "gov1",
        "difficulty": "easy",
        "domain": "AI Governance",
        "fact": "The OECD AI Principles, adopted by over 46 countries, establish five core values: inclusive growth, human-centered values, transparency, robustness, and accountability.",
    },
    {
        "id": "gov2",
        "difficulty": "medium",
        "domain": "AI Governance",
        "fact": "ISO/IEC 42001 is the first international standard for AI management systems, providing a framework for responsible AI development and deployment in enterprise settings.",
    },
    {
        "id": "gov3",
        "difficulty": "hard",
        "domain": "AI Governance",
        "fact": "The Singapore Model AI Governance Framework recommends a tiered testing approach where high-impact AI decisions require human-in-the-loop validation with documented escalation paths.",
    },
    # --- Operational Efficiency (domain 6 — noise domain) ---
    {
        "id": "ops1",
        "difficulty": "medium",
        "domain": "Operational Efficiency",
        "fact": "Gartner predicts that by 2028, 33% of enterprise software applications will include agentic AI, up from less than 1% in 2024.",
    },
    {
        "id": "ops2",
        "difficulty": "hard",
        "domain": "Operational Efficiency",
        "fact": "Infosys reports that enterprises using AI-driven process automation achieve 35-45% faster cycle times in procurement, with the highest gains in vendor onboarding and invoice reconciliation.",
    },
]


# ---------------------------------------------------------------------------
# Mission goals — two modes
# ---------------------------------------------------------------------------

MISSION_GOAL_SEQUENTIAL = """BENCHMARK MISSION: Multi-domain research synthesis

You have been provided with a research briefing containing findings across multiple domains.

RESEARCH BRIEFING:
{facts_text}

INSTRUCTIONS:
Phase 1 (Research): Thoroughly review and capture ALL findings from the briefing above. Every detail matters — specific numbers, percentages, named frameworks, and concrete examples. Store your complete findings in the shared field for other agents. Keep your output under 800 words — focus on extracting facts, not commentary.

Phase 2 (Analysis): Analyze the research findings. Identify cross-domain patterns, contradictions, and strategic implications. Query the shared field to ensure you have complete context from the research phase. Keep your output under 600 words.

Phase 3 (Synthesis): Produce a comprehensive executive summary that references specific findings from ALL domains. Query the shared field to recover any details the analyst may not have forwarded. Include specific numbers, framework names, and concrete data points from the original briefing. Keep your output under 1000 words.

CRITICAL: The final synthesis MUST reference specific data points from every domain. Vague summaries are not acceptable — cite the actual numbers and frameworks from the briefing.
"""

MISSION_GOAL_PARALLEL = """BENCHMARK MISSION: Parallel multi-domain research with cross-domain synthesis

You have been provided with a comprehensive research briefing spanning six domains. This mission requires PARALLEL execution — multiple research agents must work simultaneously on different domains, then a synthesis agent must combine ALL findings.

RESEARCH BRIEFING:
{facts_text}

INSTRUCTIONS:

PHASE 1 — PARALLEL RESEARCH (run these simultaneously, NOT sequentially):
- Domain A: Research and capture ALL EU AI Act findings. Store every specific regulation, threshold, fine amount, and exception in the shared field. Output under 400 words.
- Domain B: Research and capture ALL Cybersecurity findings. Store every framework name, percentage, dollar amount, and timeline in the shared field. Output under 400 words.
- Domain C: Research and capture ALL Market Research findings. Store every projection, adoption rate, cost figure, and percentage in the shared field. Output under 400 words.
- Domain D: Research and capture ALL Incident Response, AI Governance, and Operational Efficiency findings. Store every statistic, cost figure, and framework in the shared field. Output under 400 words.

PHASE 2 — CROSS-DOMAIN SYNTHESIS (after all research is complete):
Query the shared field to retrieve findings from ALL research agents. Produce an executive summary that:
1. References specific data points from EVERY domain (EU AI Act, Cybersecurity, Market Research, Incident Response, AI Governance, Operational Efficiency)
2. Identifies at least 3 cross-domain connections (e.g., how AI governance affects market adoption rates)
3. Includes concrete numbers: dollar amounts, percentages, timelines, and framework names
4. Keeps output under 1200 words — density over length

CRITICAL REQUIREMENTS:
- Research agents MUST run in parallel, not sequentially
- The synthesis agent MUST query the shared field to retrieve all domain findings
- Every specific number, percentage, and framework name from the briefing should appear in the final output
- Vague summaries fail the benchmark — cite actual data points
"""


def format_facts_for_briefing(facts: list[dict]) -> str:
    """Format seed facts as a natural briefing document."""
    by_domain: dict[str, list[str]] = {}
    for f in facts:
        by_domain.setdefault(f["domain"], []).append(f["fact"])

    sections = []
    for domain, items in by_domain.items():
        bullets = "\n".join(f"  - {item}" for item in items)
        sections.append(f"{domain}:\n{bullets}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

class PlatformClient:
    def __init__(self, api_url: str, auth_token: str, workspace_id: str):
        self.api_url = api_url.rstrip("/")
        self.workspace_id = workspace_id
        self.session = requests.Session()
        headers = {
            "X-Workspace-ID": workspace_id,
            "Content-Type": "application/json",
        }
        # Support both Clerk Bearer token and platform API key
        if auth_token.startswith("ey"):
            headers["Authorization"] = f"Bearer {auth_token}"
        else:
            headers["X-Api-Key"] = auth_token
        self.session.headers.update(headers)

    def create_mission(self, goal: str, token_budget: int = 30000) -> dict:
        for attempt in range(3):
            resp = self.session.post(
                f"{self.api_url}/api/missions",
                json={
                    "goal": goal,
                    "config": {
                        "auto_approve": True,
                        "token_budget": token_budget,
                        "skip_verification": True,
                    },
                },
                timeout=120,
            )
            if resp.status_code in (502, 422, 429) and attempt < 2:
                print(f"    {resp.status_code} on create_mission, retrying ({attempt + 1}/3)...")
                time.sleep(10)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return resp.json()

    def get_mission(self, mission_id: str) -> dict:
        # Cache-buster needed — Railway CDN caches GET responses
        resp = self.session.get(
            f"{self.api_url}/api/missions/{mission_id}",
            params={"_t": int(time.time() * 1000)},
        )
        if resp.status_code == 401:
            raise RuntimeError("Auth token expired — get a fresh Clerk session token")
        resp.raise_for_status()
        return resp.json()

    def get_field(self, mission_id: str) -> dict:
        resp = self.session.get(
            f"{self.api_url}/api/missions/{mission_id}/field",
            params={"_t": int(time.time() * 1000)},
        )
        resp.raise_for_status()
        return resp.json()

    def get_events(self, mission_id: str, event_type: str | None = None, limit: int = 200) -> list[dict]:
        """Fetch mission events for telemetry analysis."""
        params = {"limit": limit, "_t": int(time.time() * 1000)}
        if event_type:
            params["event_type"] = event_type
        resp = self.session.get(
            f"{self.api_url}/api/missions/{mission_id}/events",
            params=params,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("events", data) if isinstance(data, dict) else data

    def poll_until_terminal(
        self, mission_id: str, timeout: int = 1800, interval: int = 15
    ) -> dict:
        """Poll mission until it reaches a terminal state."""
        terminal = {"completed", "failed", "cancelled", "awaiting_human"}
        start = time.time()
        last_state = None
        last_task_progress = None

        while time.time() - start < timeout:
            mission = self.get_mission(mission_id)
            state = mission.get("state", "unknown")
            tasks = mission.get("tasks", [])
            done = sum(1 for t in tasks if t.get("state") in ("verified", "failed", "skipped"))
            task_progress = f"{done}/{len(tasks)}"

            if state != last_state or task_progress != last_task_progress:
                elapsed = int(time.time() - start)
                print(f"  [{elapsed:3d}s] state={state}  tasks={task_progress}")
                last_state = state
                last_task_progress = task_progress

            if state in terminal:
                return mission

            time.sleep(interval)

        raise TimeoutError(
            f"Mission {mission_id} did not complete within {timeout}s (last state: {last_state})"
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def collect_field_telemetry(client: PlatformClient, mission_id: str) -> dict:
    """Analyze mission events for field tool usage."""
    telemetry = {
        "field_queries": 0,
        "field_injects": 0,
        "field_stability_checks": 0,
        "total_tool_calls": 0,
        "agents_using_field": set(),
        "query_details": [],
    }

    try:
        events = client.get_events(mission_id, limit=200)
    except Exception as e:
        print(f"  Telemetry: failed to fetch events ({e!r})")
        return {k: (list(v) if isinstance(v, set) else v) for k, v in telemetry.items()}

    for event in events:
        event_data = event.get("data", {}) or {}
        event_type = event.get("event_type", "")

        # Look for tool call events
        tool_name = event_data.get("tool_name", "") or event_data.get("action", "") or ""

        if "field_query" in tool_name:
            telemetry["field_queries"] += 1
            agent_id = event_data.get("agent_id")
            if agent_id:
                telemetry["agents_using_field"].add(str(agent_id))
            telemetry["query_details"].append({
                "agent_id": agent_id,
                "query": (event_data.get("params", {}) or {}).get("query", "")[:80],
            })
        elif "field_inject" in tool_name:
            telemetry["field_injects"] += 1
            agent_id = event_data.get("agent_id")
            if agent_id:
                telemetry["agents_using_field"].add(str(agent_id))
        elif "field_stability" in tool_name:
            telemetry["field_stability_checks"] += 1

        if "tool" in event_type.lower() or tool_name:
            telemetry["total_tool_calls"] += 1

    # Convert set to list for JSON serialization
    telemetry["agents_using_field"] = list(telemetry["agents_using_field"])
    return telemetry


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_with_llm_judge(
    facts: list[dict], output_text: str, api_key: str
) -> dict:
    """Use an LLM to judge which facts appear in the output."""

    facts_list = "\n".join(
        f"{i+1}. [{f['id']}] ({f['difficulty']}) {f['fact']}"
        for i, f in enumerate(facts)
    )

    prompt = f"""You are a scoring judge for a multi-agent coordination benchmark.

Below is a list of specific facts that were provided to research agents in a multi-agent pipeline. Your job is to determine which of these facts were successfully captured in the final synthesis output.

FACTS TO CHECK:
{facts_list}

FINAL AGENT OUTPUT:
{output_text}

For each fact, determine if the final output contains the key information (not necessarily word-for-word, but the specific data point, number, or claim must be present).

Respond with a JSON object:
{{
  "scores": {{
    "eu1": {{"found": true/false, "evidence": "quote or explanation"}},
    "eu2": {{"found": true/false, "evidence": "..."}},
    ...
  }},
  "total_found": <number>,
  "total_facts": {len(facts)},
  "coverage": <float 0-1>
}}

Be strict: vague references don't count. The specific data point must be present."""

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "anthropic/claude-sonnet-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # Extract JSON from response
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])

    raise ValueError(f"Could not parse judge response: {content[:200]}")


def score_with_keywords(facts: list[dict], output_text: str) -> dict:
    """Fallback keyword scoring — less accurate but doesn't need API key."""
    output_lower = output_text.lower()
    scores = {}

    # Key phrases that indicate each fact is present
    indicators = {
        "eu1": ["four risk tiers", "unacceptable", "high, limited, and minimal"],
        "eu2": ["conformity assessment", "technical documentation"],
        "eu3": ["biometric identification", "missing children", "terrorist"],
        "eu4": ["35 million", "7% of global", "7 percent"],
        "eu5": ["deepfake", "artificially generated", "enisa"],
        "cyber1": ["zero-trust", "zero trust", "continuous verification"],
        "cyber2": ["mitre att&ck", "mitre attack", "14 tactical"],
        "cyber3": ["742%", "solarwinds", "18,000"],
        "cyber4": ["govern", "nist cybersecurity framework 2.0", "sixth function"],
        "cyber5": ["1.1 billion", "$1.1", "5.3 million", "chainalysis"],
        "mkt1": ["1.8 trillion", "$1.8", "37%"],
        "mkt2": ["34% adoption", "financial services leading", "multi-agent"],
        "mkt3": ["23% reduction", "40% increase", "18 months"],
        "mkt4": ["2.6 to 4.4 trillion", "$2.6", "$4.4", "63 enterprise"],
        "mkt5": ["11% of enterprises", "67% citing", "54% citing"],
        "ir1": ["nist", "four phases", "preparation, detection"],
        "ir2": ["204 days", "73 days", "mean time to identify"],
        "ir3": ["$2.66 million", "2.66 million", "incident response team"],
        "ir4": ["$4.45 million", "4.45 million", "$10.93", "10.93 million"],
        "ir5": ["$1.76 million", "1.76 million", "108 days"],
        "gov1": ["oecd ai principles", "46 countries", "inclusive growth"],
        "gov2": ["iso/iec 42001", "iso 42001", "ai management system"],
        "gov3": ["singapore model", "tiered testing", "human-in-the-loop"],
        "ops1": ["33% of enterprise", "agentic ai", "2028"],
        "ops2": ["35-45%", "35 to 45", "vendor onboarding", "invoice reconciliation"],
    }

    for fact in facts:
        fact_id = fact["id"]
        keywords = indicators.get(fact_id, [])
        found = any(kw.lower() in output_lower for kw in keywords)
        scores[fact_id] = {
            "found": found,
            "evidence": "keyword match" if found else "not found",
        }

    total_found = sum(1 for s in scores.values() if s["found"])
    return {
        "scores": scores,
        "total_found": total_found,
        "total_facts": len(facts),
        "coverage": total_found / len(facts) if facts else 0,
        "method": "keyword",
    }


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    client: PlatformClient,
    trial_num: int,
    facts: list[dict],
    judge_api_key: str | None,
    mode: str = "parallel",
) -> dict:
    """Run a single benchmark trial."""
    print(f"\n{'='*60}")
    print(f"Trial {trial_num} (mode={mode})")
    print(f"{'='*60}")

    # Build mission goal with seeded facts
    facts_text = format_facts_for_briefing(facts)
    if mode == "parallel":
        goal = MISSION_GOAL_PARALLEL.format(facts_text=facts_text)
    else:
        goal = MISSION_GOAL_SEQUENTIAL.format(facts_text=facts_text)

    # Create and run mission
    print("Creating mission (auto_approve=true)...")
    # Parallel mode needs more tokens (4+ concurrent agents)
    budget = 200000 if mode == "parallel" else 50000
    mission = client.create_mission(goal, token_budget=budget)
    mission_id = mission["id"]
    print(f"  Mission: {mission_id}")

    # Poll until complete
    print("Waiting for completion...")
    try:
        result = client.poll_until_terminal(mission_id, timeout=1800)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        return {
            "trial": trial_num,
            "mission_id": mission_id,
            "mode": mode,
            "status": "timeout",
            "coverage": 0,
        }

    state = result.get("state", "unknown")
    print(f"  Final state: {state}")

    if state not in ("completed", "awaiting_human"):
        print("  Mission did not complete successfully")
        return {
            "trial": trial_num,
            "mission_id": mission_id,
            "mode": mode,
            "status": state,
            "coverage": 0,
        }

    # Get task outputs
    tasks = result.get("tasks", [])
    tokens_used = result.get("tokens_used", 0)

    # Find the last task's output (synthesis step)
    verified_tasks = [t for t in tasks if t.get("state") == "verified"]
    if not verified_tasks:
        print("  No verified tasks found")
        return {
            "trial": trial_num,
            "mission_id": mission_id,
            "mode": mode,
            "status": "no_verified_tasks",
            "coverage": 0,
        }

    # Sort by sequence number, take the last one
    verified_tasks.sort(key=lambda t: t.get("sequence_number", 0))
    final_task = verified_tasks[-1]
    final_output = final_task.get("output", "") or ""

    print(f"  Final task: {final_task.get('title', 'unknown')}")
    print(f"  Output length: {len(final_output)} chars")
    print(f"  Tokens used: {tokens_used}")
    print(f"  Tasks: {len(verified_tasks)} verified / {len(tasks)} total")

    # Get field data
    try:
        field_data = client.get_field(mission_id)
        field_info = {
            "backend": field_data.get("backend"),
            "pattern_count": len(field_data.get("patterns", [])),
            "stability": field_data.get("stability", {}).get("stability"),
        }
    except Exception:
        field_info = {"backend": "unknown", "pattern_count": 0, "stability": None}

    # Collect telemetry — did agents actually use field tools?
    print("Collecting telemetry...")
    telemetry = collect_field_telemetry(client, mission_id)
    print(f"  Field queries: {telemetry['field_queries']}")
    print(f"  Field injects: {telemetry['field_injects']}")
    print(f"  Agents using field: {len(telemetry['agents_using_field'])}")

    # Score coverage
    print("Scoring...")
    if judge_api_key:
        try:
            scores = score_with_llm_judge(facts, final_output, judge_api_key)
            scores["method"] = "llm_judge"
            print(f"  LLM judge: {scores['coverage']:.0%} coverage")
        except Exception as e:
            print(f"  LLM judge failed ({e}), falling back to keyword")
            scores = score_with_keywords(facts, final_output)
    else:
        scores = score_with_keywords(facts, final_output)

    print(f"  Coverage: {scores['coverage']:.0%} ({scores['total_found']}/{scores['total_facts']})")

    # Per-difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        diff_facts = [f for f in facts if f["difficulty"] == diff]
        diff_found = sum(
            1 for f in diff_facts if scores["scores"].get(f["id"], {}).get("found")
        )
        print(f"    {diff}: {diff_found}/{len(diff_facts)}")

    # Per-domain breakdown
    domains = sorted(set(f["domain"] for f in facts))
    for domain in domains:
        domain_facts = [f for f in facts if f["domain"] == domain]
        domain_found = sum(
            1 for f in domain_facts if scores["scores"].get(f["id"], {}).get("found")
        )
        print(f"    {domain}: {domain_found}/{len(domain_facts)}")

    return {
        "trial": trial_num,
        "mission_id": mission_id,
        "mode": mode,
        "status": state,
        "coverage": scores["coverage"],
        "total_found": scores["total_found"],
        "total_facts": scores["total_facts"],
        "scoring_method": scores.get("method", "unknown"),
        "tokens_used": tokens_used,
        "task_count": len(tasks),
        "verified_count": len(verified_tasks),
        "final_output_length": len(final_output),
        "field": field_info,
        "telemetry": telemetry,
        "per_fact": scores["scores"],
        "per_difficulty": {
            diff: {
                "found": sum(
                    1
                    for f in facts
                    if f["difficulty"] == diff
                    and scores["scores"].get(f["id"], {}).get("found")
                ),
                "total": sum(1 for f in facts if f["difficulty"] == diff),
            }
            for diff in ("easy", "medium", "hard")
        },
        "per_domain": {
            domain: {
                "found": sum(
                    1
                    for f in facts
                    if f["domain"] == domain
                    and scores["scores"].get(f["id"], {}).get("found")
                ),
                "total": sum(1 for f in facts if f["domain"] == domain),
            }
            for domain in domains
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Field Memory Benchmark — honest A/B test"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of trials (default: 3)"
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="parallel",
        help="Mission structure: sequential (3-phase pipeline) or parallel (4 concurrent + synthesis)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Label for this run (e.g. 'vector_field' or 'redis'). "
        "Auto-detected from field API if not set.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("AUTOMATOS_API_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--auth-token",
        default=os.getenv("AUTOMATOS_AUTH_TOKEN"),
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("AUTOMATOS_WORKSPACE"),
    )
    parser.add_argument(
        "--judge-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key for LLM judge (falls back to keyword scoring)",
    )
    parser.add_argument(
        "--output-dir",
        default="tools/benchmark_results",
        help="Directory for result files",
    )
    args = parser.parse_args()

    if not args.auth_token:
        print("Error: set AUTOMATOS_AUTH_TOKEN or use --auth-token")
        sys.exit(1)
    if not args.workspace:
        print("Error: set AUTOMATOS_WORKSPACE or use --workspace")
        sys.exit(1)

    client = PlatformClient(args.api_url, args.auth_token, args.workspace)

    # Detect backend label
    label = args.label
    if not label:
        print("Detecting backend...")
        try:
            label = "unknown"
        except Exception:
            label = "unknown"

    print(f"\nField Memory Benchmark")
    print(f"  API:      {args.api_url}")
    print(f"  Label:    {label}")
    print(f"  Mode:     {args.mode}")
    print(f"  Trials:   {args.trials}")
    print(f"  Facts:    {len(SEED_FACTS)} ({len(set(f['domain'] for f in SEED_FACTS))} domains)")
    print(f"  Judge:    {'LLM (OpenRouter)' if args.judge_key else 'Keyword fallback'}")

    # Run trials
    results = []
    for i in range(1, args.trials + 1):
        try:
            trial_result = run_trial(client, i, SEED_FACTS, args.judge_key, mode=args.mode)
        except RuntimeError:
            raise  # Auth errors should abort immediately
        except Exception as e:
            print(f"  Trial {i} crashed: {e!r}")
            trial_result = {"trial": i, "status": "error", "coverage": 0, "error": str(e)}

        # Auto-detect label from first trial's field data
        if label == "unknown" and trial_result.get("field", {}).get("backend"):
            label = trial_result["field"]["backend"]
            print(f"  Detected backend: {label}")

        results.append(trial_result)

    # Summary
    successful = [r for r in results if r.get("coverage", 0) > 0]
    coverages = [r["coverage"] for r in successful]

    print(f"\n{'='*60}")
    print(f"RESULTS — {label} ({args.mode} mode)")
    print(f"{'='*60}")
    print(f"  Trials:     {len(results)} ({len(successful)} successful)")

    if coverages:
        avg = sum(coverages) / len(coverages)
        min_c = min(coverages)
        max_c = max(coverages)
        print(f"  Coverage:   {avg:.0%} avg  (min={min_c:.0%}, max={max_c:.0%})")

        # Per-difficulty averages
        for diff in ("easy", "medium", "hard"):
            diff_coverages = []
            for r in successful:
                d = r.get("per_difficulty", {}).get(diff, {})
                if d.get("total", 0) > 0:
                    diff_coverages.append(d["found"] / d["total"])
            if diff_coverages:
                diff_avg = sum(diff_coverages) / len(diff_coverages)
                print(f"    {diff:8s}: {diff_avg:.0%}")

        # Per-domain averages
        domains = sorted(set(f["domain"] for f in SEED_FACTS))
        print(f"  Per-domain:")
        for domain in domains:
            domain_coverages = []
            for r in successful:
                d = r.get("per_domain", {}).get(domain, {})
                if d.get("total", 0) > 0:
                    domain_coverages.append(d["found"] / d["total"])
            if domain_coverages:
                domain_avg = sum(domain_coverages) / len(domain_coverages)
                domain_total = sum(1 for f in SEED_FACTS if f["domain"] == domain)
                print(f"    {domain:25s}: {domain_avg:.0%} ({domain_total} facts)")

        avg_tokens = sum(r.get("tokens_used", 0) for r in successful) / len(successful)
        print(f"  Avg tokens: {avg_tokens:,.0f}")

        # Telemetry summary
        total_queries = sum(r.get("telemetry", {}).get("field_queries", 0) for r in successful)
        total_injects = sum(r.get("telemetry", {}).get("field_injects", 0) for r in successful)
        print(f"  Field queries (total): {total_queries}")
        print(f"  Field injects (total): {total_injects}")
    else:
        print("  No successful trials")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{label}_{args.mode}_{timestamp}.json"

    output_data = {
        "label": label,
        "mode": args.mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trials": args.trials,
            "facts_count": len(SEED_FACTS),
            "domains": len(set(f["domain"] for f in SEED_FACTS)),
            "api_url": args.api_url,
            "scoring_method": "llm_judge" if args.judge_key else "keyword",
        },
        "summary": {
            "total_trials": len(results),
            "successful_trials": len(successful),
            "avg_coverage": sum(coverages) / len(coverages) if coverages else 0,
            "min_coverage": min(coverages) if coverages else 0,
            "max_coverage": max(coverages) if coverages else 0,
            "per_difficulty": {
                diff: {
                    "avg_coverage": (
                        sum(
                            r["per_difficulty"][diff]["found"]
                            / r["per_difficulty"][diff]["total"]
                            for r in successful
                            if r.get("per_difficulty", {}).get(diff, {}).get("total", 0) > 0
                        )
                        / max(
                            1,
                            sum(
                                1
                                for r in successful
                                if r.get("per_difficulty", {}).get(diff, {}).get("total", 0) > 0
                            ),
                        )
                    )
                }
                for diff in ("easy", "medium", "hard")
            },
            "per_domain": {
                domain: {
                    "avg_coverage": (
                        sum(
                            r["per_domain"][domain]["found"]
                            / r["per_domain"][domain]["total"]
                            for r in successful
                            if r.get("per_domain", {}).get(domain, {}).get("total", 0) > 0
                        )
                        / max(
                            1,
                            sum(
                                1
                                for r in successful
                                if r.get("per_domain", {}).get(domain, {}).get("total", 0) > 0
                            ),
                        )
                    )
                }
                for domain in sorted(set(f["domain"] for f in SEED_FACTS))
            },
        },
        "trials": results,
    }

    output_file.write_text(json.dumps(output_data, indent=2))
    print(f"\n  Results saved: {output_file}")

    # If we have both result files, offer comparison
    existing = list(output_dir.glob("benchmark_*.json"))
    labels_seen = {json.loads(f.read_text())["label"] for f in existing}
    if len(labels_seen) >= 2:
        print(f"\n  Multiple backends detected: {labels_seen}")
        print(f"  Run: python tools/compare_benchmarks.py {output_dir}")


if __name__ == "__main__":
    main()
