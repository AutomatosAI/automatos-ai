"""
Blog Pipeline Model Benchmark
==============================

Runs the daily-blog-pipeline with different model configurations,
collects metrics (duration, tokens, cost, quality), and outputs
a comparison table + markdown report.

Usage:
    python scripts/benchmark_blog_pipeline.py

Set env vars:
    API_URL=https://api.automatos.app  (default)
    API_KEY=your-api-key
    WORKSPACE_ID=your-workspace-id
"""

import json
import os
import sys
import time
import httpx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Config ───────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "https://api.automatos.app")
API_KEY = os.environ.get("API_KEY", os.environ.get("ORCHESTRATOR_API_KEY", ""))
WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "ae8320bc-95e1-4de1-bbe9-396bef19cbf8")
TEMPLATE_ID = "daily-blog-pipeline"
AGENT_IDS = {"QUILL": 191, "EDITOR": 305, "CANVAS": 303}

# Poll settings
POLL_INTERVAL = 15  # seconds
MAX_WAIT = 600      # 10 minutes max per run

# ── Model Configurations to Benchmark ─────────────────────────────────
# Each entry: (label, model_id, approx_cost_per_M_input)

MODEL_CONFIGS = [
    ("GPT-5.4 (premium)", "openai/gpt-5.4", 2.00),
    ("DeepSeek Chat", "deepseek/deepseek-chat", 0.14),
    ("Gemini 2.5 Flash", "google/gemini-2.5-flash", 0.15),
    ("Gemini 2.5 Pro", "google/gemini-2.5-pro", 1.25),
    ("Llama 4 Scout", "meta-llama/llama-4-scout", 0.08),
    ("Mistral Small 3.1", "mistralai/mistral-small-3.1-24b-instruct", 0.03),
    ("Qwen3 235B MoE", "qwen/qwen3-235b-a22b-2507", 0.07),
    ("Qwen3.5 Flash", "qwen/qwen3.5-flash-02-23", 0.07),
    ("GPT-4.1 Mini", "openai/gpt-4.1-mini", 0.40),
    ("Gemini 2.0 Flash Lite", "google/gemini-2.0-flash-lite-001", 0.07),
]

# ── Helpers ───────────────────────────────────────────────────────────

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    "X-Workspace-ID": WORKSPACE_ID,
}


def set_agent_models(model_id: str):
    """Update all 3 blog pipeline agents to use the given model."""
    from core.database.database import get_database_url
    from sqlalchemy import create_engine, text as sql_text

    engine = create_engine(get_database_url())
    with engine.connect() as db:
        for name, agent_id in AGENT_IDS.items():
            row = db.execute(
                sql_text("SELECT model_config FROM agents WHERE id = :id"),
                {"id": agent_id},
            ).fetchone()
            config = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            config["model_id"] = model_id
            config["max_tokens"] = 4000
            db.execute(
                sql_text("UPDATE agents SET model_config = :cfg WHERE id = :id"),
                {"cfg": json.dumps(config), "id": agent_id},
            )
        db.commit()


def trigger_pipeline(category: str = "AI & Automation") -> str:
    """Trigger the blog pipeline and return execution_id."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{API_URL}/api/workflow-recipes/{TEMPLATE_ID}/execute",
            headers=HEADERS,
            json={"inputs": {"category": category}},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["recipe_execution_id"]


def poll_execution(exec_id: str) -> dict:
    """Poll until execution completes or times out."""
    start = time.monotonic()
    with httpx.Client(timeout=30) as client:
        while time.monotonic() - start < MAX_WAIT:
            resp = client.get(
                f"{API_URL}/api/workflow-recipes/{TEMPLATE_ID}/executions/{exec_id}",
                headers=HEADERS,
            )
            data = resp.json()
            status = data.get("status", "unknown")
            current = data.get("current_step", "?")
            total = data.get("total_steps", "?")

            if status in ("completed", "failed"):
                return data

            elapsed = int(time.monotonic() - start)
            print(f"    [{elapsed}s] Step {current}/{total} - {status}", flush=True)
            time.sleep(POLL_INTERVAL)

    return {"status": "timeout", "error_message": f"Exceeded {MAX_WAIT}s"}


def extract_metrics(result: dict) -> dict:
    """Extract key metrics from execution result."""
    steps = result.get("step_results", [])

    total_tokens = 0
    total_duration_ms = 0
    step_details = []
    all_tools = []

    for s in steps:
        tokens = s.get("tokens_used", 0) or 0
        duration = s.get("duration_ms", 0) or 0
        total_tokens += tokens
        total_duration_ms += duration
        tools = s.get("tool_calls_summary", [])
        all_tools.extend(tools)

        step_details.append({
            "step": s.get("step_id"),
            "agent": s.get("agent_name"),
            "status": s.get("status"),
            "duration_ms": duration,
            "tokens": tokens,
            "tool_calls": len(tools),
            "error": s.get("error"),
        })

    # Count successful vs failed steps
    completed_steps = sum(1 for s in step_details if s["status"] == "completed")
    failed_steps = sum(1 for s in step_details if s["status"] not in ("completed", None))

    return {
        "status": result.get("status"),
        "total_duration_ms": total_duration_ms,
        "total_tokens": total_tokens,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "total_steps": len(steps),
        "tool_calls": len(all_tools),
        "successful_tools": sum(1 for t in all_tools if "success" in t.lower()),
        "step_details": step_details,
        "error": result.get("error_message"),
    }


def get_latest_draft() -> dict | None:
    """Get the most recent draft blog post."""
    from core.database.database import get_database_url
    from sqlalchemy import create_engine, text as sql_text

    engine = create_engine(get_database_url())
    with engine.connect() as db:
        row = db.execute(sql_text("""
            SELECT id, title, status, excerpt, content, cover_image_url,
                   seo_title, seo_description, tags, category,
                   LENGTH(content) as content_len
            FROM blog_posts
            WHERE workspace_id = :ws AND status = 'draft'
            ORDER BY created_at DESC LIMIT 1
        """), {"ws": WORKSPACE_ID}).fetchone()

        if not row:
            return None
        return {
            "post_id": str(row[0]),
            "title": row[1],
            "status": row[2],
            "has_excerpt": bool(row[3]),
            "content_length": row[10] or 0,
            "has_cover_image": bool(row[5]),
            "has_seo_title": bool(row[6]),
            "has_seo_description": bool(row[7]),
            "tag_count": len(row[8]) if row[8] else 0,
            "category": row[9],
        }


def quality_score(draft: dict | None, metrics: dict) -> int:
    """Score 0-100 based on completeness."""
    if not draft:
        return 0
    score = 0
    # Content quality (40 points)
    if draft["content_length"] > 500:
        score += 20
    if draft["content_length"] > 1500:
        score += 10
    if draft["content_length"] > 3000:
        score += 10
    # Metadata (30 points)
    if draft["has_excerpt"]:
        score += 10
    if draft["has_seo_title"]:
        score += 10
    if draft["has_seo_description"]:
        score += 10
    # Extras (20 points)
    if draft["has_cover_image"]:
        score += 10
    if draft["tag_count"] >= 3:
        score += 5
    if draft["category"]:
        score += 5
    # Pipeline completion (10 points)
    if metrics["completed_steps"] >= 4:
        score += 10
    elif metrics["completed_steps"] >= 3:
        score += 5
    return score


# ── Main Benchmark Loop ──────────────────────────────────────────────

def run_benchmark():
    if not API_KEY:
        print("ERROR: Set API_KEY or ORCHESTRATOR_API_KEY env var")
        sys.exit(1)

    results = []
    categories = [
        "AI & Automation", "Developer Tools", "Future of Work",
        "Cloud Infrastructure", "AI Agents", "Machine Learning",
        "Open Source", "Startup Engineering", "AI Safety",
        "Edge Computing",
    ]

    print("=" * 70)
    print("  BLOG PIPELINE MODEL BENCHMARK")
    print(f"  {len(MODEL_CONFIGS)} models x 1 run each")
    print("=" * 70)
    print()

    for i, (label, model_id, approx_cost) in enumerate(MODEL_CONFIGS):
        run_num = i + 1
        category = categories[i % len(categories)]

        print(f"[Run {run_num}/{len(MODEL_CONFIGS)}] {label} ({model_id})")
        print(f"  Category: {category}")

        # 1. Set models
        print("  Setting agent models...", flush=True)
        try:
            set_agent_models(model_id)
        except Exception as e:
            print(f"  ERROR setting models: {e}")
            results.append({
                "run": run_num, "label": label, "model": model_id,
                "status": "config_error", "error": str(e),
            })
            continue

        # 2. Trigger pipeline
        print("  Triggering pipeline...", flush=True)
        try:
            exec_id = trigger_pipeline(category)
            print(f"  Execution: {exec_id}")
        except Exception as e:
            print(f"  ERROR triggering: {e}")
            results.append({
                "run": run_num, "label": label, "model": model_id,
                "status": "trigger_error", "error": str(e),
            })
            continue

        # 3. Poll for completion
        print("  Waiting for completion...", flush=True)
        run_start = time.monotonic()
        execution = poll_execution(exec_id)
        wall_time = int(time.monotonic() - run_start)

        # 4. Extract metrics
        metrics = extract_metrics(execution)
        draft = get_latest_draft()
        q_score = quality_score(draft, metrics)

        # Estimate cost (rough)
        est_cost = (metrics["total_tokens"] / 1_000_000) * approx_cost

        result = {
            "run": run_num,
            "label": label,
            "model": model_id,
            "category": category,
            "exec_id": exec_id,
            "status": metrics["status"],
            "wall_time_s": wall_time,
            "total_duration_ms": metrics["total_duration_ms"],
            "total_tokens": metrics["total_tokens"],
            "completed_steps": metrics["completed_steps"],
            "failed_steps": metrics["failed_steps"],
            "tool_calls": metrics["tool_calls"],
            "successful_tools": metrics["successful_tools"],
            "quality_score": q_score,
            "est_cost_usd": round(est_cost, 4),
            "draft": draft,
            "step_details": metrics["step_details"],
            "error": metrics.get("error"),
        }
        results.append(result)

        # Print summary
        status_icon = "OK" if metrics["status"] == "completed" else "FAIL"
        print(f"  [{status_icon}] {wall_time}s wall | {metrics['total_tokens']} tokens | "
              f"{metrics['completed_steps']}/4 steps | quality={q_score}/100 | ~${est_cost:.4f}")
        if draft:
            print(f"  Draft: \"{draft['title'][:60]}\" ({draft['content_length']} chars)")
        if metrics.get("error"):
            print(f"  Error: {metrics['error'][:100]}")
        print()

        # Brief pause between runs
        if i < len(MODEL_CONFIGS) - 1:
            print("  Cooling down 5s...\n")
            time.sleep(5)

    # ── Output Results ────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  BENCHMARK RESULTS")
    print("=" * 90)

    # Summary table
    header = f"{'#':>2} {'Model':<28} {'Status':<9} {'Time':>6} {'Tokens':>8} {'Steps':>5} {'Tools':>5} {'Quality':>7} {'Cost':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        status = r.get("status", "error")[:8]
        time_s = r.get("wall_time_s", 0)
        tokens = r.get("total_tokens", 0)
        steps = f"{r.get('completed_steps', 0)}/4"
        tools = r.get("tool_calls", 0)
        quality = f"{r.get('quality_score', 0)}/100"
        cost = f"${r.get('est_cost_usd', 0):.4f}"

        print(f"{r['run']:>2} {r['label']:<28} {status:<9} {time_s:>5}s {tokens:>8} {steps:>5} {tools:>5} {quality:>7} {cost:>8}")

    # Save full results as JSON
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")

    # Generate markdown report
    md_path = os.path.join(os.path.dirname(__file__), "benchmark_report.md")
    generate_markdown_report(results, md_path)
    print(f"Markdown report saved to: {md_path}")

    # Restore to DeepSeek (cheap default)
    print("\nRestoring agents to deepseek/deepseek-chat...")
    set_agent_models("deepseek/deepseek-chat")
    print("Done!")


def generate_markdown_report(results: list, path: str):
    """Generate a publishable markdown benchmark report."""
    lines = [
        "# AI Model Benchmark: Blog Pipeline Showdown",
        "",
        "We ran the same 4-step autonomous blog pipeline (research, write, edit, design) "
        "across 10 different AI models to find the best balance of speed, quality, and cost.",
        "",
        "## The Pipeline",
        "",
        "| Step | Agent | Task |",
        "|------|-------|------|",
        "| 1 | QUILL | Research trending topic, write 800-1500 word draft |",
        "| 2 | EDITOR | Review draft, improve clarity/SEO/engagement |",
        "| 3 | CANVAS | Generate cover image via AI image gen |",
        "| 4 | QUILL | Create approval task on the board |",
        "",
        "## Results",
        "",
        "| # | Model | Status | Time | Tokens | Steps | Quality | Est. Cost |",
        "|---|-------|--------|------|--------|-------|---------|-----------|",
    ]

    for r in results:
        status = r.get("status", "error")
        status_icon = "pass" if status == "completed" else "fail"
        lines.append(
            f"| {r['run']} | {r['label']} | {status_icon} | "
            f"{r.get('wall_time_s', 0)}s | {r.get('total_tokens', 0):,} | "
            f"{r.get('completed_steps', 0)}/4 | {r.get('quality_score', 0)}/100 | "
            f"${r.get('est_cost_usd', 0):.4f} |"
        )

    # Find winners
    completed = [r for r in results if r.get("status") == "completed"]
    if completed:
        fastest = min(completed, key=lambda r: r.get("wall_time_s", 999))
        cheapest = min(completed, key=lambda r: r.get("est_cost_usd", 999))
        best_quality = max(completed, key=lambda r: r.get("quality_score", 0))
        most_efficient = min(completed, key=lambda r: r.get("total_tokens", 999999))

        lines.extend([
            "",
            "## Winners",
            "",
            f"- **Fastest**: {fastest['label']} ({fastest.get('wall_time_s')}s)",
            f"- **Cheapest**: {cheapest['label']} (${cheapest.get('est_cost_usd', 0):.4f})",
            f"- **Best Quality**: {best_quality['label']} ({best_quality.get('quality_score')}/100)",
            f"- **Most Token-Efficient**: {most_efficient['label']} ({most_efficient.get('total_tokens'):,} tokens)",
        ])

    # Value score
    if completed:
        lines.extend([
            "",
            "## Value Score (Quality / Cost)",
            "",
        ])
        scored = []
        for r in completed:
            cost = r.get("est_cost_usd", 0.0001) or 0.0001
            value = r.get("quality_score", 0) / cost
            scored.append((r["label"], value, r.get("quality_score", 0), cost))
        scored.sort(key=lambda x: x[1], reverse=True)
        for i, (label, value, quality, cost) in enumerate(scored, 1):
            lines.append(f"{i}. **{label}** — {value:.0f} quality/$ (Q={quality}, ${cost:.4f})")

    lines.extend([
        "",
        "---",
        "",
        "*Benchmark run by Automatos AI Platform. "
        "All models accessed via OpenRouter. "
        "Pipeline: daily-blog-pipeline (4 sequential steps).*",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_benchmark()
