"""
PRD-139 US-006: Bootstrap synthetic telemetry for cold-start pipeline validation.

Generates synthetic ToolExecutionLog rows from the 47 eval training queries,
mapping each to known-correct actions with realistic agent bias, timing noise,
and multi-action turn grouping. Idempotent — clears prior synthetic data before
inserting fresh rows.

Usage:
    cd orchestrator
    python -m scripts.eval.tool_routing.seed_telemetry [--verify]

Options:
    --verify    After seeding, call build_edges() and print edge summary
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Synthetic agent IDs — outside real agent ID range to avoid FK conflicts
AGENT_SENTINEL = 9001  # Heavy on reports + memory
AGENT_SCOUT = 9002     # Heavy on workspace + documents
AGENT_OPS = 9003       # Balanced, slightly favors agents + missions

SYNTHETIC_AGENTS = [AGENT_SENTINEL, AGENT_SCOUT, AGENT_OPS]

# Repetitions per eval query (target: ~40 per query => ~1880 base rows)
REPETITIONS_PER_QUERY = 42

# Success/failure ratio
SUCCESS_RATE = 0.80

# Workspace UUID for synthetic data (deterministic from seed)
SYNTHETIC_WORKSPACE_ID = uuid.UUID("00000000-dead-beef-cafe-000000139006")

# Time window: spread rows over 30 days ending at script run time
TIME_WINDOW_DAYS = 30

# Telemetry source marker for idempotent cleanup
TELEMETRY_SOURCE = "synthetic"

# Random seed for reproducibility
RANDOM_SEED = 139006

# Category-to-agent bias weights (higher = more likely assigned to that agent)
# agent_id -> category -> relative weight
AGENT_CATEGORY_BIAS: Dict[int, Dict[str, float]] = {
    AGENT_SENTINEL: {
        "reports": 5.0,
        "memory": 5.0,
        "agents": 2.0,
        "analytics": 2.0,
        "missions": 1.5,
        "playbooks": 1.0,
        "documents": 0.5,
        "workspace": 0.3,
        "workspace_code": 0.3,
        "marketplace": 1.0,
        "external": 0.5,
        "cross": 2.0,
    },
    AGENT_SCOUT: {
        "reports": 0.5,
        "memory": 1.0,
        "agents": 1.0,
        "analytics": 1.0,
        "missions": 0.5,
        "playbooks": 1.0,
        "documents": 5.0,
        "workspace": 4.0,
        "workspace_code": 5.0,
        "marketplace": 2.0,
        "external": 2.0,
        "cross": 1.5,
    },
    AGENT_OPS: {
        "reports": 1.5,
        "memory": 1.5,
        "agents": 4.0,
        "analytics": 3.0,
        "missions": 4.0,
        "playbooks": 3.0,
        "documents": 1.5,
        "workspace": 1.5,
        "workspace_code": 1.5,
        "marketplace": 2.0,
        "external": 2.0,
        "cross": 3.0,
    },
}

# Multi-action turn pairings: when an action in the key set is selected,
# we sometimes pair it with the follow-up action to create used_after signals
MULTI_ACTION_PAIRS: Dict[str, List[str]] = {
    "platform_get_latest_report": ["platform_submit_report"],
    "platform_search_memory": ["platform_store_memory"],
    "workspace_read_file": ["workspace_write_file"],
    "platform_list_agents": ["platform_get_agent", "platform_create_agent"],
    "platform_list_missions": ["platform_get_mission", "platform_create_mission"],
    "platform_list_documents": ["search_knowledge"],
    "platform_list_playbooks": ["platform_execute_playbook"],
    "platform_get_llm_usage": ["platform_get_cost_breakdown"],
    "platform_workspace_stats": ["platform_get_success_rate"],
}

# Probability of pairing a multi-action follow-up in the same turn
PAIR_PROBABILITY = 0.45


# ---------------------------------------------------------------------------
# Eval set loader
# ---------------------------------------------------------------------------


def _load_eval_set() -> List[Dict[str, Any]]:
    """Load eval_set.jsonl from the same directory."""
    eval_path = Path(__file__).parent / "eval_set.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval_set.jsonl not found at {eval_path}. "
            "Run seed_eval_set.py first to generate it."
        )
    rows = []
    with open(eval_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _select_agent(category: str, rng: random.Random) -> int:
    """Select an agent_id weighted by category bias."""
    weights = [
        AGENT_CATEGORY_BIAS[agent].get(category, 1.0)
        for agent in SYNTHETIC_AGENTS
    ]
    return rng.choices(SYNTHETIC_AGENTS, weights=weights, k=1)[0]


def _generate_timestamp(
    base_time: datetime, day_offset: int, rng: random.Random
) -> datetime:
    """Generate a realistic timestamp within a given day."""
    hour = rng.choices(
        range(24),
        weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0,
                 3.0, 2.5, 2.0, 2.5, 3.0, 3.0, 2.5, 2.0, 1.5, 1.0,
                 0.8, 0.5, 0.3, 0.2],
        k=1,
    )[0]
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return base_time + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


def _generate_execution_time_ms(is_success: bool, rng: random.Random) -> int:
    """Generate realistic execution time."""
    if is_success:
        # Success: 50-3000ms, biased toward 200-800ms
        return int(rng.gauss(500, 300))
    else:
        # Failures tend to be slower (timeouts) or very fast (validation errors)
        if rng.random() < 0.3:
            return rng.randint(10, 100)  # Fast validation failure
        return int(rng.gauss(2000, 1000))  # Slow timeout/error


def _make_turn_id(query_id: str, repetition: int, agent_id: int) -> str:
    """Generate a deterministic turn_id for grouping."""
    raw = f"{query_id}:{repetition}:{agent_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def generate_synthetic_rows(
    eval_set: List[Dict[str, Any]],
    base_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic telemetry rows from eval set.

    Args:
        eval_set: Loaded eval_set.jsonl entries.
        base_time: Start of the 30-day window. Defaults to utcnow() - 30 days.
            Pass a fixed value for deterministic output in tests.

    Returns list of dicts ready for bulk insert into tool_execution_logs.
    """
    rng = random.Random(RANDOM_SEED)
    rows: List[Dict[str, Any]] = []

    if base_time is None:
        base_time = datetime.utcnow() - timedelta(days=TIME_WINDOW_DAYS)

    for eval_entry in eval_set:
        query_id = eval_entry["query_id"]
        query_text = eval_entry["query"]
        correct_actions = eval_entry["correct_actions"]
        category = eval_entry["category"]

        for rep in range(REPETITIONS_PER_QUERY):
            # Select agent based on category bias
            agent_id = _select_agent(category, rng)

            # Determine success/failure
            is_success = rng.random() < SUCCESS_RATE

            # Pick a primary action (first correct action, or random from list)
            if len(correct_actions) == 1:
                primary_action = correct_actions[0]
            else:
                primary_action = rng.choice(correct_actions)

            # Generate timing
            day_offset = rng.randint(0, TIME_WINDOW_DAYS - 1)
            executed_at = _generate_timestamp(base_time, day_offset, rng)
            execution_time_ms = max(10, _generate_execution_time_ms(is_success, rng))

            # Turn ID for grouping (used_after edges need this)
            turn_id = _make_turn_id(query_id, rep, agent_id)

            # Determine app_name from action prefix
            app_name = _action_to_app_name(primary_action)

            # Primary row
            row = _build_row(
                agent_id=agent_id,
                action_name=primary_action,
                app_name=app_name,
                user_query=query_text,
                status="success" if is_success else "error",
                execution_time_ms=execution_time_ms,
                executed_at=executed_at,
                turn_id=turn_id,
                workspace_id=SYNTHETIC_WORKSPACE_ID,
            )
            rows.append(row)

            # Multi-action pairing: add a follow-up action in the same turn
            if primary_action in MULTI_ACTION_PAIRS and rng.random() < PAIR_PROBABILITY:
                follow_ups = MULTI_ACTION_PAIRS[primary_action]
                follow_up = rng.choice(follow_ups)
                # Follow-up comes 200-2000ms after primary
                follow_delay_ms = rng.randint(200, 2000)
                follow_executed_at = executed_at + timedelta(milliseconds=follow_delay_ms)
                follow_success = rng.random() < SUCCESS_RATE
                follow_exec_time = max(10, _generate_execution_time_ms(follow_success, rng))

                follow_row = _build_row(
                    agent_id=agent_id,
                    action_name=follow_up,
                    app_name=_action_to_app_name(follow_up),
                    user_query=query_text,
                    status="success" if follow_success else "error",
                    execution_time_ms=follow_exec_time,
                    executed_at=follow_executed_at,
                    turn_id=turn_id,
                    workspace_id=SYNTHETIC_WORKSPACE_ID,
                )
                rows.append(follow_row)

    return rows


def _action_to_app_name(action_name: str) -> str:
    """Derive app_name from action name prefix."""
    if action_name.startswith("platform_"):
        return "PLATFORM"
    elif action_name.startswith("workspace_"):
        return "WORKSPACE"
    elif action_name.startswith("composio_"):
        return "COMPOSIO"
    elif action_name == "search_knowledge":
        return "PLATFORM"
    else:
        prefix = action_name.split("_")[0].upper() if "_" in action_name else action_name.upper()
        return prefix


def _build_row(
    *,
    agent_id: int,
    action_name: str,
    app_name: str,
    user_query: str,
    status: str,
    execution_time_ms: int,
    executed_at: datetime,
    turn_id: str,
    workspace_id: uuid.UUID,
) -> Dict[str, Any]:
    """Build a row dict for insertion."""
    return {
        "agent_id": agent_id,
        "app_name": app_name,
        "action_name": action_name,
        "user_query": user_query,
        "workspace_id": workspace_id,
        "status": status,
        "execution_time_ms": execution_time_ms,
        "executed_at": executed_at,
        "telemetry_source": TELEMETRY_SOURCE,
        "routing_source": "keyword",  # Synthetic assumes keyword routing
        "router_decision": {"turn_id": turn_id},
        "input_parameters": {"keys": ["query"]},
        "cache_hit": False,
    }


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def seed_telemetry(db: Session, rows: List[Dict[str, Any]]) -> int:
    """Write synthetic telemetry rows to tool_execution_logs.

    Idempotent: deletes all rows with telemetry_source='synthetic' before inserting.

    Args:
        db: SQLAlchemy session
        rows: List of row dicts from generate_synthetic_rows()

    Returns:
        Number of rows inserted
    """
    from core.models.composio_cache import ToolExecutionLog

    # Step 1: Idempotent cleanup — remove prior synthetic data
    deleted = db.query(ToolExecutionLog).filter(
        ToolExecutionLog.telemetry_source == TELEMETRY_SOURCE
    ).delete(synchronize_session="fetch")

    if deleted:
        logger.info(f"Cleaned up {deleted} prior synthetic rows")

    db.flush()

    # Step 2: Bulk insert new rows
    for row_data in rows:
        entry = ToolExecutionLog(
            agent_id=row_data["agent_id"],
            app_name=row_data["app_name"],
            action_name=row_data["action_name"],
            user_query=row_data["user_query"],
            workspace_id=row_data["workspace_id"],
            status=row_data["status"],
            execution_time_ms=row_data["execution_time_ms"],
            executed_at=row_data["executed_at"],
            telemetry_source=row_data["telemetry_source"],
            routing_source=row_data["routing_source"],
            router_decision=row_data["router_decision"],
            input_parameters=row_data["input_parameters"],
            cache_hit=row_data["cache_hit"],
        )
        db.add(entry)

    db.flush()
    db.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# Verification (optional --verify flag)
# ---------------------------------------------------------------------------


async def verify_edge_build() -> None:
    """Run build_edges() after seeding to verify end-to-end pipeline."""
    from core.services.edge_builder import build_edges

    logger.info("Running build_edges() to verify pipeline...")
    summary = await build_edges(window=timedelta(days=TIME_WINDOW_DAYS + 1))

    print(f"\n{'='*60}")
    print("Edge Build Verification Summary")
    print(f"{'='*60}")
    print(f"  Logs processed:    {summary.logs_processed}")
    print(f"  Edges built:       {summary.edges_built}")
    print(f"  Affinities built:  {summary.affinities_built}")
    print(f"  Intent clusters:   {summary.intent_clusters}")
    print(f"  Duration (ms):     {summary.duration_ms}")
    print(f"{'='*60}")

    if summary.edges_built == 0:
        print("\n  WARNING: No edges built — check sample_floor or turn grouping")
    if summary.affinities_built == 0:
        print("\n  WARNING: No affinities built — check cluster assignment")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for python -m scripts.eval.tool_routing.seed_telemetry."""
    parser = argparse.ArgumentParser(
        description="Seed synthetic telemetry for cold-start pipeline validation"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After seeding, run build_edges() and print summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate rows but don't write to DB — print stats only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load eval set
    eval_set = _load_eval_set()
    logger.info(f"Loaded {len(eval_set)} eval queries from eval_set.jsonl")

    # Generate synthetic rows
    rows = generate_synthetic_rows(eval_set)
    logger.info(f"Generated {len(rows)} synthetic telemetry rows")

    # Print generation stats
    agent_counts = {}
    status_counts = {"success": 0, "error": 0}
    for row in rows:
        agent_counts[row["agent_id"]] = agent_counts.get(row["agent_id"], 0) + 1
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(f"\nGeneration Summary:")
    print(f"  Total rows: {len(rows)}")
    print(f"  Queries:    {len(eval_set)}")
    print(f"  Agent distribution:")
    for agent_id, count in sorted(agent_counts.items()):
        print(f"    Agent {agent_id}: {count} rows ({100*count/len(rows):.1f}%)")
    print(f"  Status distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"    {status}: {count} ({100*count/len(rows):.1f}%)")

    # Count multi-action turns
    turn_ids = [row["router_decision"]["turn_id"] for row in rows]
    from collections import Counter
    turn_counts = Counter(turn_ids)
    multi_turns = sum(1 for c in turn_counts.values() if c > 1)
    print(f"  Multi-action turns: {multi_turns}")

    if args.dry_run:
        print("\n  [DRY RUN] No rows written to database.")
        return

    # Write to DB
    from core.database.database import get_db_session

    with get_db_session() as db:
        inserted = seed_telemetry(db, rows)
        logger.info(f"Inserted {inserted} synthetic telemetry rows")

    if args.verify:
        asyncio.run(verify_edge_build())

    print(f"\nDone. {len(rows)} synthetic rows seeded with telemetry_source='{TELEMETRY_SOURCE}'")


if __name__ == "__main__":
    main()
