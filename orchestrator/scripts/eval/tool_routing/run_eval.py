"""
Tool-routing eval runner.

Iterates the cartesian product of (model × mode × query), calls OpenRouter
with the appropriate tool schemas + system prompt, captures the chosen
action, and appends one row per call to `results/results.jsonl`.

Run from the orchestrator/ directory so PYTHONPATH resolves the
ActionRegistry import:

    cd orchestrator
    export OPENROUTER_API_KEY=sk-or-...
    python -m scripts.eval.tool_routing.run_eval

Resumability: rows already in `results.jsonl` are skipped on re-run.
You can interrupt at any time and rerun — only the unfinished cells
will be filled in.

Output schema (one JSON object per line):

    {
      "ts":             "2026-05-02T12:34:56Z",
      "model":          "openai/gpt-5-mini",
      "mode":           "filtered" | "full",
      "query_id":       "q017",
      "query":          "...",
      "correct_actions": ["platform_get_llm_usage", ...],
      "chosen_action":  "platform_get_llm_usage" | null,
      "chosen_via":     "platform_execute" | "direct" | null,
      "surfaced":       ["...", ...],          # action names the prompt actually showed
      "prompt_tokens":  1234,
      "completion_tokens": 42,
      "total_tokens":   1276,
      "latency_ms":     920,
      "raw_finish":     "tool_calls" | "stop" | "length" | ...,
      "error":          null | "<message>"
    }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_runner")

HERE = Path(__file__).resolve().parent
MODELS_YAML = HERE / "models.yaml"
EVAL_JSONL = HERE / "eval_set.jsonl"
RESULTS_DIR = HERE / "results"
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"

# Load env from both top-level repo and orchestrator/.env (the production
# EmbeddingManager + Redis CacheService used by ActionSemanticIndex read these
# at runtime). Existing exported env vars take precedence (load_dotenv default).
_ORCH_ROOT = HERE.parents[2]                         # orchestrator/
_REPO_ROOT = _ORCH_ROOT.parent                       # automatos-ai/
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_ORCH_ROOT / ".env")


# ──────────────────────────────────────────────────────────────────
# Top-level tool schemas
# ──────────────────────────────────────────────────────────────────
#
# `platform_execute` is the meta-tool: the LLM picks an action from the
# catalog injected into the system prompt and passes it as `action`.
#
# The other tools below are first-class top-level tools that bypass the
# action catalog. They mirror the production tool surface so the eval is
# a faithful test of "given the same tool schemas the agent sees in prod,
# does the LLM still pick the right one?"

PLATFORM_EXECUTE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "platform_execute",
        "description": (
            "Execute a platform action by name. The action MUST be one of the "
            "actions listed in the 'Available Platform Actions' section of the "
            "system prompt. Pass action-specific parameters in `params`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Exact action name from the catalog above.",
                },
                "params": {
                    "type": "object",
                    "description": "Action-specific parameters.",
                    "additionalProperties": True,
                },
            },
            "required": ["action"],
        },
    },
}


def _simple_tool(name: str, description: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Tool-specific input."}
                },
                "additionalProperties": True,
            },
        },
    }


TOP_LEVEL_TOOLS: List[Dict[str, Any]] = [
    PLATFORM_EXECUTE_TOOL,
    _simple_tool(
        "composio_execute",
        "Execute an external integration via Composio (Gmail, Slack, GitHub, "
        "Notion, Jira, etc.). Use this for sending emails, posting messages, "
        "creating issues, and other third-party SaaS actions.",
    ),
    _simple_tool(
        "search_knowledge",
        "Semantic search over the workspace's uploaded documents and knowledge "
        "base. Returns relevant document chunks for a natural-language query.",
    ),
    _simple_tool(
        "workspace_read_file",
        "Read a file from the workspace's connected code repository.",
    ),
    _simple_tool(
        "workspace_write_file",
        "Write or overwrite a file in the workspace's connected code repository.",
    ),
    _simple_tool(
        "workspace_list_dir",
        "List the contents of a directory in the workspace's connected repo.",
    ),
    _simple_tool(
        "workspace_grep",
        "Search for a regex pattern across files in the workspace's connected repo.",
    ),
    _simple_tool(
        "workspace_exec",
        "Execute a shell command (build, test, lint) in the workspace's repo sandbox.",
    ),
    _simple_tool(
        "workspace_git",
        "Run a git command (status, diff, log) in the workspace's connected repo.",
    ),
]

SYSTEM_PROMPT_PREAMBLE = (
    "You are an AI assistant for a multi-agent platform. The user will ask "
    "for an action. You MUST respond with exactly one tool call — pick the "
    "single most appropriate tool from the available tools and the platform "
    "action catalog below. Do not explain, do not chain calls, do not ask "
    "clarifying questions. If the user's request is best served by a "
    "platform action, use `platform_execute` with the chosen action name.\n\n"
)


# ──────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text())


def _load_eval_set() -> List[Dict[str, Any]]:
    if not EVAL_JSONL.exists():
        raise FileNotFoundError(
            f"{EVAL_JSONL} missing — run `python -m scripts.eval.tool_routing.seed_eval_set` first"
        )
    return [json.loads(line) for line in EVAL_JSONL.read_text().splitlines() if line.strip()]


def _existing_keys() -> Set[Tuple[str, str, str]]:
    """Return the set of (model, mode, query_id) already in results.jsonl."""
    keys: Set[Tuple[str, str, str]] = set()
    if not RESULTS_JSONL.exists():
        return keys
    for line in RESULTS_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((row.get("model", ""), row.get("mode", ""), row.get("query_id", "")))
    return keys


def _append_row(row: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────
# OpenRouter call
# ──────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _call_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_query: str,
    *,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    tools: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Call OpenRouter once. Returns (parsed_result, error).
    The parsed_result keys map directly into the output row.
    """
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            tools=tools,
            tool_choice="required",  # force a tool call so we always get a chosen action
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return (
            {
                "chosen_action": None,
                "chosen_via": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_ms": latency_ms,
                "raw_finish": None,
            },
            f"{type(exc).__name__}: {exc}",
        )

    latency_ms = int((time.perf_counter() - started) * 1000)

    chosen_action: Optional[str] = None
    chosen_via: Optional[str] = None
    raw_finish = None

    try:
        choice = resp.choices[0]
        raw_finish = getattr(choice, "finish_reason", None)
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.function
            fn_name = fn.name
            if fn_name == "platform_execute":
                # The chosen "action" is inside the JSON args.
                args = json.loads(fn.arguments or "{}")
                chosen_action = args.get("action")
                chosen_via = "platform_execute"
            else:
                chosen_action = fn_name
                chosen_via = "direct"
    except Exception as exc:  # noqa: BLE001
        return (
            {
                "chosen_action": None,
                "chosen_via": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "latency_ms": latency_ms,
                "raw_finish": raw_finish,
            },
            f"parse_error: {type(exc).__name__}: {exc}",
        )

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return (
        {
            "chosen_action": chosen_action,
            "chosen_via": chosen_via,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "raw_finish": raw_finish,
        },
        None,
    )


# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "filtered", "both"],
        default="both",
        help="Which prompt mode(s) to run. Default: both.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model IDs (overrides models.yaml selection).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of (model, mode, query) cells. Useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and print plan; don't call any LLM.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    cfg = _load_yaml(MODELS_YAML)
    models_cfg: List[Dict[str, Any]] = cfg.get("models") or []
    # Embedding model is now driven by the production EmbeddingManager
    # (config: EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_DIMENSION).
    # The `embedding:` block in models.yaml is informational only.
    top_k = int(cfg.get("top_k", 15))
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 256))
    request_timeout = int(cfg.get("request_timeout", 60))

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models_cfg = [m for m in models_cfg if m["id"] in wanted]
    if not models_cfg:
        print("No models selected", file=sys.stderr)
        return 2

    modes: List[str]
    if args.mode == "both":
        modes = ["full", "filtered"]
    else:
        modes = [args.mode]

    queries = _load_eval_set()
    logger.info(
        f"Loaded {len(queries)} queries × {len(models_cfg)} models × {len(modes)} modes "
        f"= {len(queries) * len(models_cfg) * len(modes)} cells"
    )

    # ── Action registry + prompt builder ────────────────────────
    # Imported lazily so `--dry-run` and `--help` don't require the orchestrator's deps.
    # NOTE: bootstrap MUST run before prompt_builder is used in filtered mode,
    # because filtered mode imports `modules.tools.discovery.action_semantic_index`
    # which depends on the same stubbed package chain.
    from scripts.eval.tool_routing._registry_bootstrap import load_registry
    from scripts.eval.tool_routing.prompt_builder import PromptBuilder

    registry = load_registry()
    actions = registry.get_all()
    logger.info(f"Loaded {len(actions)} actions from live registry")

    if "filtered" in modes and not api_key and not args.dry_run:
        # The production ActionSemanticIndex routes embeddings through
        # EmbeddingManager, which (per .env: EMBEDDING_PROVIDER=openrouter)
        # also requires OPENROUTER_API_KEY at request time.
        print("filtered mode needs OPENROUTER_API_KEY", file=sys.stderr)
        return 2

    builder = PromptBuilder(actions=actions)

    # OpenAI-compatible client pointed at OpenRouter
    if args.dry_run:
        client = None
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("openai package missing — `pip install openai`", file=sys.stderr)
            return 2
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    done = _existing_keys()
    if done:
        logger.info(f"Skipping {len(done)} cells already in results.jsonl")

    # Cache prompts per mode/query so we don't rebuild N_models times.
    prompt_cache: Dict[Tuple[str, str], Tuple[str, List[str]]] = {}

    def get_prompt(mode: str, query: str) -> Tuple[str, List[str]]:
        key = (mode, query)
        if key in prompt_cache:
            return prompt_cache[key]
        catalog, surfaced = builder.build(query, mode=mode, top_k=top_k)
        prompt = SYSTEM_PROMPT_PREAMBLE + catalog
        prompt_cache[key] = (prompt, surfaced)
        return prompt_cache[key]

    cells_run = 0
    for model_cfg in models_cfg:
        model_id = model_cfg["id"]
        for mode in modes:
            for q in queries:
                if args.limit is not None and cells_run >= args.limit:
                    break
                key = (model_id, mode, q["query_id"])
                if key in done:
                    continue

                prompt, surfaced = get_prompt(mode, q["query"])

                if args.dry_run:
                    logger.info(
                        f"[dry-run] {model_id} | {mode} | {q['query_id']} | "
                        f"prompt={len(prompt)} chars, surfaced={len(surfaced)}"
                    )
                    cells_run += 1
                    continue

                logger.info(f"→ {model_id} | {mode} | {q['query_id']} | {q['query'][:60]!r}")
                result, err = _call_model(
                    client,
                    model=model_id,
                    system_prompt=prompt,
                    user_query=q["query"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    request_timeout=request_timeout,
                    tools=TOP_LEVEL_TOOLS,
                )
                row = {
                    "ts": _now_iso(),
                    "model": model_id,
                    "mode": mode,
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "correct_actions": q["correct_actions"],
                    "category": q.get("category"),
                    "difficulty": q.get("difficulty"),
                    **result,
                    "surfaced": surfaced,
                    "error": err,
                }
                _append_row(row)
                cells_run += 1

    if args.dry_run:
        logger.info(f"Done (dry-run). Would have run {cells_run} cells.")
    else:
        logger.info(f"Done. Wrote {cells_run} new rows → {RESULTS_JSONL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
