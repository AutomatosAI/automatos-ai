"""The cost row, from ``llm_usage`` (PRD-247 S0.4, D7).

The platform stamps every model call with an ``execution_id`` it chooses
(``board_task:<id>`` for a ticket, others for chat and heartbeats); the runner
cannot inject a campaign id. So attribution is by the ids the run created plus
the workspace window: everything the sim workspace spent between the run's
first and last call, which also catches what the scenarios did not ask for
(heartbeats, watchers, retries) — those are the interesting rows.

Rows are read with ``psql`` inside the postgres container; the credentials stay
in that container's environment and never pass through this process.
"""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .config import Settings

FIELD_SEP = "\x1f"
COLUMNS = ("execution_id", "request_type", "model_id", "provider", "agent_id", "input_tokens",
           "output_tokens", "cache_read_tokens", "total_cost", "latency_ms", "status", "created_at")
NUMERIC = {"input_tokens": int, "output_tokens": int, "cache_read_tokens": int, "latency_ms": int,
           "total_cost": float, "agent_id": int}
QUERY_TIMEOUT_S = 60


class CostError(RuntimeError):
    """Usage rows could not be read; the run records the reason and scores cost as unknown."""


def _sql(workspace_id: str, since: str) -> str:
    # Both values are validated before they are interpolated: a UUID and an ISO timestamp.
    ws = str(uuid.UUID(workspace_id))
    ts = datetime.fromisoformat(since).isoformat()
    # created_at comes back as naive UTC in ISO form whatever the column type:
    # AT TIME ZONE 'UTC' normalises timestamptz and timestamp alike, PGTZ=UTC
    # (set on the psql session below) pins the rendering.
    cols = ", ".join(f"to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS')" if c == "created_at" else c
                     for c in COLUMNS)
    return (f"SELECT {cols} FROM llm_usage WHERE workspace_id = '{ws}' "
            f"AND created_at >= '{ts}' ORDER BY created_at")


def usage_rows(settings: Settings, workspace_id: str, since_iso: str) -> tuple[dict[str, Any], ...]:
    """Every ``llm_usage`` row for the workspace since ``since_iso`` (ISO 8601)."""
    sql = _sql(workspace_id, since_iso)
    cmd = [settings.docker, "exec", "-e", f"SIM_SQL={sql}", "-e", "PGTZ=UTC", settings.postgres_container, "sh", "-c",
           'psql -X -q -A -t -F "$(printf \'\\037\')" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "$SIM_SQL"']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=QUERY_TIMEOUT_S, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise CostError(f"psql via docker failed: {exc}") from exc
    if proc.returncode != 0:
        raise CostError(f"psql exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
    return tuple(parse_rows(proc.stdout))


def parse_rows(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        values = line.split(FIELD_SEP)
        if len(values) != len(COLUMNS):
            continue
        row: dict[str, Any] = dict(zip(COLUMNS, values))
        for key, caster in NUMERIC.items():
            try:
                row[key] = caster(row[key]) if row[key] != "" else None
            except ValueError:
                row[key] = None
        rows.append(row)
    return rows


def summarise(rows: Iterable[Mapping[str, Any]], execution_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Totals over ``rows``, narrowed to ``execution_ids`` when given."""
    wanted = set(execution_ids or ())
    picked = [r for r in rows if not wanted or r.get("execution_id") in wanted]
    by_model: dict[str, dict[str, Any]] = {}
    for row in picked:
        slot = by_model.setdefault(str(row.get("model_id") or "?"), {"calls": 0, "cost_usd": 0.0, "tokens": 0})
        slot["calls"] += 1
        slot["cost_usd"] += float(row.get("total_cost") or 0.0)
        slot["tokens"] += int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
    return {
        "calls": len(picked),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in picked),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in picked),
        "cache_read_tokens": sum(int(r.get("cache_read_tokens") or 0) for r in picked),
        "cost_usd": round(sum(float(r.get("total_cost") or 0.0) for r in picked), 6),
        "errors": sum(1 for r in picked if str(r.get("status") or "").lower() not in ("", "success", "ok")),
        "latency_ms_max": max((int(r.get("latency_ms") or 0) for r in picked), default=0),
        "by_model": by_model,
    }


def workspace_total(rows: Iterable[Mapping[str, Any]]) -> float:
    return round(sum(float(r.get("total_cost") or 0.0) for r in rows), 6)


def models_seen(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(r.get("model_id") or "?") for r in rows}))
