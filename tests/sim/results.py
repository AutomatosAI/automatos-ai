"""Records the drivers produce and the scorer reads.

A ``ScenarioResult`` is the whole evidence for one scenario: what was asked,
what the platform did (status timeline, chat turns, tool calls), what came out
(deliverables, reports), what the customer was asked, and the checks that were
run against the pack's expectations. Findings are derived from these, never
from memory of what "should" have happened (PRD-247 D10).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .api import Api, Trace
from .config import Settings

TEXT_KEYS = ("content", "summary", "body", "result", "text", "description", "excerpt")
TEXT_CAP = 20_000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ScenarioResult:
    id: str
    kind: str
    started_at: str
    ended_at: str
    outcome: str
    ok: bool
    task: Mapping[str, Any] | None = None
    timeline: tuple[Mapping[str, Any], ...] = ()
    chat: tuple[Mapping[str, Any], ...] = ()
    steps: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    effects: Mapping[str, Any] = field(default_factory=dict)
    asks: tuple[Mapping[str, Any], ...] = ()
    checks: tuple[Check, ...] = ()
    errors: tuple[str, ...] = ()
    execution_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    evidence_text: str = ""
    expect: str = ""
    brief: str = ""

    @property
    def duration_s(self) -> float:
        start, end = parse_iso(self.started_at), parse_iso(self.ended_at)
        return round((end - start).total_seconds(), 1) if start and end else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        return data


@dataclass
class RunContext:
    settings: Settings
    api: Api
    trace: Trace
    workspace_id: str
    agents: Mapping[str, Mapping[str, Any]]
    persona: Mapping[str, Any]
    run_started: str


def text_of(obj: Any, cap: int = TEXT_CAP) -> str:
    """The readable text inside an API object — the fields a customer would read."""
    parts: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 3 or sum(len(p) for p in parts) > cap:
            return
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, Mapping):
            for key in TEXT_KEYS:
                if isinstance(node.get(key), str):
                    parts.append(node[key])
            for key, value in node.items():
                if key not in TEXT_KEYS and isinstance(value, (Mapping, list)):
                    walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(obj, 0)
    joined = "\n".join(p for p in parts if p)
    return joined[:cap]


def must_contain_checks(must_contain: Sequence[str], text: str) -> tuple[Check, ...]:
    lowered = text.lower()
    return tuple(Check(f"contains:{needle}", needle.lower() in lowered,
                       "" if needle.lower() in lowered else "not found in the output")
                 for needle in must_contain)


def effect_checks(expect: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[Check, ...]:
    """``*_min`` and ``*_created`` keys are floors; ``no_errors`` and ``status`` are exact."""
    checks = []
    for key, wanted in expect.items():
        seen = observed.get(key)
        if key == "no_errors":
            ok = (observed.get("errors", 0) == 0) == bool(wanted)
            checks.append(Check("effect:no_errors", ok, f"errors={observed.get('errors', 0)}"))
        elif key == "status":
            ok = str(seen) == str(wanted)
            checks.append(Check("effect:status", ok, f"wanted {wanted}, got {seen}"))
        else:
            ok = isinstance(seen, (int, float)) and seen >= float(wanted)
            checks.append(Check(f"effect:{key}", ok, f"wanted >= {wanted}, got {seen}"))
    return tuple(checks)


def slug(value: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:limit] or "x"
