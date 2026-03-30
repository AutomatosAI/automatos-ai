#!/usr/bin/env python3
"""
Field Memory Benchmark — Honest A/B test of shared semantic fields
vs message-passing for multi-agent context coverage.

Standalone script. Calls the platform API externally.
Real agents, real LLM calls, real Qdrant. No synthetic embeddings.

Usage:
  # 1. Run with vector_field backend (default)
  python tools/benchmark_field_memory.py --trials 3

  # 2. Switch backend on Railway:
  #    railway variables set SHARED_CONTEXT_BACKEND=redis
  #    Wait for redeploy

  # 3. Run with redis backend
  python tools/benchmark_field_memory.py --trials 3 --label redis

  # 4. Compare the two JSON result files

Environment variables:
  AUTOMATOS_API_URL    - Platform API URL (default: http://localhost:8000)
  AUTOMATOS_AUTH_TOKEN - Clerk Bearer token (grab from browser DevTools)
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
]

# ---------------------------------------------------------------------------
# Mission goal — instructs a 3-step sequential research pipeline
# ---------------------------------------------------------------------------

MISSION_GOAL = """BENCHMARK MISSION: Multi-domain research synthesis

You have been provided with a research briefing containing findings across four domains: EU AI Act compliance, Cybersecurity landscape, Market Research on AI adoption, and Incident Response best practices.

RESEARCH BRIEFING:
{facts_text}

INSTRUCTIONS:
Phase 1 (Research): Thoroughly review and capture ALL findings from the briefing above. Every detail matters — specific numbers, percentages, named frameworks, and concrete examples. Store your complete findings in the shared field for other agents.

Phase 2 (Analysis): Analyze the research findings. Identify cross-domain patterns, contradictions, and strategic implications. Query the shared field to ensure you have complete context from the research phase.

Phase 3 (Synthesis): Produce a comprehensive executive summary that references specific findings from ALL four domains. Query the shared field to recover any details the analyst may not have forwarded. Include specific numbers, framework names, and concrete data points from the original briefing.

CRITICAL: The final synthesis MUST reference specific data points from every domain. Vague summaries are not acceptable — cite the actual numbers and frameworks from the briefing.
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
                    },
                },
                timeout=120,
            )
            if resp.status_code == 502 and attempt < 2:
                print(f"    502 on create_mission, retrying ({attempt + 1}/3)...")
                time.sleep(5)
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
        resp.raise_for_status()
        return resp.json()

    def get_field(self, mission_id: str) -> dict:
        resp = self.session.get(f"{self.api_url}/api/missions/{mission_id}/field")
        resp.raise_for_status()
        return resp.json()

    def poll_until_terminal(
        self, mission_id: str, timeout: int = 900, interval: int = 15
    ) -> dict:
        """Poll mission until it reaches a terminal state."""
        terminal = {"completed", "failed", "cancelled", "awaiting_human"}
        start = time.time()
        last_state = None

        while time.time() - start < timeout:
            mission = self.get_mission(mission_id)
            state = mission.get("state", "unknown")

            if state != last_state:
                elapsed = int(time.time() - start)
                tasks = mission.get("tasks", [])
                done = sum(1 for t in tasks if t.get("state") in ("verified", "failed", "skipped"))
                print(f"  [{elapsed:3d}s] state={state}  tasks={done}/{len(tasks)}")
                last_state = state

            if state in terminal:
                return mission

            time.sleep(interval)

        raise TimeoutError(
            f"Mission {mission_id} did not complete within {timeout}s (last state: {last_state})"
        )


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

Below is a list of specific facts that were provided to the first agent in a 3-agent pipeline. Your job is to determine which of these facts were successfully captured in the final agent's output.

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
        "cyber1": ["zero-trust", "zero trust", "continuous verification"],
        "cyber2": ["mitre att&ck", "mitre attack", "14 tactical"],
        "cyber3": ["742%", "solarwinds", "18,000"],
        "mkt1": ["1.8 trillion", "$1.8", "37%"],
        "mkt2": ["34% adoption", "financial services leading", "multi-agent"],
        "mkt3": ["23% reduction", "40% increase", "18 months"],
        "ir1": ["nist", "four phases", "preparation, detection"],
        "ir2": ["204 days", "73 days", "mean time to identify"],
        "ir3": ["$2.66 million", "2.66 million", "incident response team"],
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
) -> dict:
    """Run a single benchmark trial."""
    print(f"\n{'='*60}")
    print(f"Trial {trial_num}")
    print(f"{'='*60}")

    # Build mission goal with seeded facts
    facts_text = format_facts_for_briefing(facts)
    goal = MISSION_GOAL.format(facts_text=facts_text)

    # Create and run mission
    print("Creating mission (auto_approve=true)...")
    mission = client.create_mission(goal, token_budget=30000)
    mission_id = mission["id"]
    print(f"  Mission: {mission_id}")

    # Poll until complete
    print("Waiting for completion...")
    try:
        result = client.poll_until_terminal(mission_id, timeout=900)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        return {
            "trial": trial_num,
            "mission_id": mission_id,
            "status": "timeout",
            "coverage": 0,
        }

    state = result.get("state", "unknown")
    print(f"  Final state: {state}")

    if state not in ("completed", "awaiting_human"):
        print(f"  Mission did not complete successfully")
        return {
            "trial": trial_num,
            "mission_id": mission_id,
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

    return {
        "trial": trial_num,
        "mission_id": mission_id,
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
        print("Error: set AUTOMATOS_AUTH_TOKEN (Clerk Bearer token from browser DevTools)")
        sys.exit(1)
    if not args.workspace:
        print("Error: set AUTOMATOS_WORKSPACE (workspace UUID)")
        sys.exit(1)

    client = PlatformClient(args.api_url, args.auth_token, args.workspace)

    # Detect backend label
    label = args.label
    if not label:
        print("Detecting backend...")
        try:
            # Create a throwaway check — we'll detect from the first trial instead
            label = "unknown"
        except Exception:
            label = "unknown"

    print(f"\nField Memory Benchmark")
    print(f"  API:      {args.api_url}")
    print(f"  Label:    {label}")
    print(f"  Trials:   {args.trials}")
    print(f"  Facts:    {len(SEED_FACTS)}")
    print(f"  Judge:    {'LLM (OpenRouter)' if args.judge_key else 'Keyword fallback'}")

    # Run trials
    results = []
    for i in range(1, args.trials + 1):
        trial_result = run_trial(client, i, SEED_FACTS, args.judge_key)

        # Auto-detect label from first trial's field data
        if label == "unknown" and trial_result.get("field", {}).get("backend"):
            label = trial_result["field"]["backend"]
            print(f"  Detected backend: {label}")

        results.append(trial_result)

    # Summary
    successful = [r for r in results if r["coverage"] > 0]
    coverages = [r["coverage"] for r in successful]

    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
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

        avg_tokens = sum(r.get("tokens_used", 0) for r in successful) / len(successful)
        print(f"  Avg tokens: {avg_tokens:,.0f}")
    else:
        print("  No successful trials")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{label}_{timestamp}.json"

    output_data = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trials": args.trials,
            "facts_count": len(SEED_FACTS),
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
