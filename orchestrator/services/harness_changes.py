"""PRD-263 US-049/US-050: the change-kind registry, one vocabulary for PRESCRIBE and the apply action.

At 697ca7d6e PRESCRIBE emitted four change kinds and the apply helper knew ten, so a
kind could be proposed that nothing applied, or applied that nothing proposed (the
4-vs-10 mismatch), and two of the four proposed a placeholder in place of a value.
Every kind now has one entry in ``KINDS`` (FR-13):

* **Apply kinds** may become an owner's Approvals card in a ``propose`` workspace
  (US-054) and are applied only by ``platform_harness_apply`` (US-052): each names its
  detector, inner platform action(s), params builder, risk, predicted metric, guard
  metrics, prior-value reader and rollback builder. They are ``heartbeat_tune``,
  ``tool_assignment_remove`` and ``rollback``.
* **Report kinds** are lines of the audit report and nothing else: ``reconnect_app``,
  ``model``, ``description``, ``temperature``, ``tags``, ``routing`` and ``power_mode``.
  The kinds that were dead or placeholders stay report lines (§11-Q9), with no
  detector until real evidence is wired for one.

:func:`prescribe` runs every kind's detector over the run's evidence and validates each
suggestion (ORG-6, ORG-7). An invalid one (an unknown kind, no target agent, the
workspace's Auto as target, or an apply kind with no prior value or a proposed value
its params builder refuses) is dropped, and its reason is kept on the run's row
(``harness_runs.prescriptions``). Nothing is queued with a placeholder.

The detectors: ``heartbeat_tune`` (US-050); ``reconnect_app`` (DIAGNOSE's tool issues,
US-019); ``tool_assignment_remove`` behind its gate (US-057, below); and ``rollback``,
whose suggestions the post-change check (services.harness_post_change, US-056) puts in
DIAGNOSE's result.

**The tool-removal gate (US-057, section 11-Q17 default).** A removal is suggested only
for an app assigned to the agent whose judged week's production calls (asked and refused
rows are never read) number at least HARNESS_MIN_SAMPLES and fail, not counting failures
``classify_tool_error`` puts down to the app's sign-in or connection (those are "reconnect
<app>" lines), with a Wilson lower bound of at least HARNESS_TOOL_FAIL_WILSON_MIN. It is an
apply kind only while no Playbook or active Mission that routes to the agent needs the app
(harness_metrics' app links); otherwise it stays a report line naming what needs the app.

No write, no platform action, no model call: the prior-value readers only read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import and_
from sqlalchemy.orm import Session

from core.stats import harness_verdicts as hv
from services.harness_diagnose import PLATFORM_APPS, RECONNECT_APP, TOOL_FAILURE
from services.harness_metrics import (
    LINK_ASSIGNED, LINK_MISSION, LINK_PLAYBOOK, HarnessMetrics, heartbeat_interval, heartbeat_setting,
)

# The apply kinds (FR-13).
HEARTBEAT_TUNE = "heartbeat_tune"
TOOL_ASSIGNMENT_REMOVE = "tool_assignment_remove"
ROLLBACK = "rollback"

# The platform actions platform_harness_apply runs as the workspace's Auto (US-052).
CONFIGURE_HEARTBEAT = "platform_configure_agent_heartbeat"
UNASSIGN_TOOL = "platform_unassign_tool_from_agent"
ASSIGN_TOOL = "platform_assign_tool_to_agent"

# What a change predicts will move, by services.harness_diagnose metric name; a tool
# removal predicts its own agent-app failure rate, which the removal takes to n=0, so
# its harm shows in the guard metrics (design ORG-10).
TASK_SUCCESS = "task_success"
TOOL_FAILURE_RATE = "tool_failure"
HEARTBEAT_OBJECTIVE = "heartbeat_objective"
APP_TOOL_FAILURE = "app_tool_failure"
# Judged after every applied change, beside its predicted metric (US-056).
GUARD_METRICS = (TASK_SUCCESS, TOOL_FAILURE_RATE)

# configure_agent_heartbeat clamps every interval it writes to 5-1440 minutes
# (modules/tools/discovery/handlers_assignments.py); heartbeat_tune doubles within it.
HEARTBEAT_MIN_INTERVAL_MINUTES = 5
HEARTBEAT_MAX_INTERVAL_MINUTES = 1440
HEARTBEAT_INTERVAL_FACTOR = 2

# The keys of current and proposed values, and of a rollback's proposed value.
INTERVAL_KEY = "interval_minutes"
APP_KEY = "app_name"
ASSIGNED_KEY = "assigned"
ROLLBACK_OF_KEY = "kind"
ROLLBACK_VALUE_KEY = "value"
# The rx id of the change a rollback undoes (US-056): its ledger row moves to rolled_back.
ROLLBACK_RX_KEY = "rx_id"
# A suggestion of an apply kind its gate keeps a report line carries why under this key.
REPORT_ONLY_KEY = "report_only"
# What needs an app, as a removal kept a report line names it (US-057).
NEEDED_BY = {LINK_PLAYBOOK: 'the Playbook "{source}"', LINK_MISSION: 'the Mission "{source}"'}

TARGET_AGENT = "agent"
# A dropped suggestion's entry on the run's row carries its reason under this key.
DROPPED_KEY = "dropped"
DROP_UNKNOWN_KIND = "unknown change kind"
DROP_NO_TARGET = "no target agent"
DROP_AUTO = "targets the workspace's Auto"
DROP_NO_PRIOR = "no prior value"
DROP_NO_CHANGE = "proposes no change"


@dataclass(frozen=True)
class ChangeSettings:
    """The detectors' thresholds (config.py, PRD-263 section 9)."""

    heartbeat_min_runs: int
    heartbeat_max_met: float
    min_samples: int
    tool_fail_wilson_min: float


@dataclass(frozen=True)
class Evidence:
    """What a run's detectors read: COLLECT's counts, DIAGNOSE's result and the thresholds."""

    metrics: HarnessMetrics
    diagnosis: Mapping[str, Any]
    settings: ChangeSettings


@dataclass(frozen=True)
class Proposal:
    """One change a detector suggests for one agent, before it has an id."""

    target_id: int
    target_name: str
    current_value: Dict[str, Any]
    proposed_value: Dict[str, Any]
    rationale: str
    expected_improvement: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Set when the kind's gate keeps this suggestion a report line (US-057): why.
    report_only: Optional[str] = None


@dataclass(frozen=True)
class InnerCall:
    """The platform action, and its params, that makes (or undoes) one change."""

    action: str
    params: Dict[str, Any]


Detector = Callable[[Evidence], Tuple[Proposal, ...]]
# (target agent id, proposed value) -> the call that makes the change.
ParamsBuilder = Callable[[int, Mapping[str, Any]], InnerCall]
# (db, workspace id, target agent id, the change's value) -> the setting as it reads now,
# shaped like the prescription's current_value; None when the agent is not in the workspace.
PriorValueReader = Callable[[Session, Union[UUID, str], int, Mapping[str, Any]], Optional[Dict[str, Any]]]
# (target agent id, prior value) -> the call that restores the prior value.
RollbackBuilder = Callable[[int, Mapping[str, Any]], InnerCall]


@dataclass(frozen=True)
class ChangeKind:
    """One change kind. A report kind has only its detector and risk; an apply kind has
    all eight fields. ``risk`` (1-5) is printed with each suggestion of the kind."""

    name: str
    detector: Detector
    risk: int
    inner_actions: Tuple[str, ...] = ()
    params_builder: Optional[ParamsBuilder] = None
    predicted_metric: Optional[str] = None
    guard_metrics: Tuple[str, ...] = ()
    prior_value_reader: Optional[PriorValueReader] = None
    rollback_builder: Optional[RollbackBuilder] = None

    @property
    def applies(self) -> bool:
        """Whether the kind can be applied (an apply kind), rather than only reported."""
        return bool(self.inner_actions)


@dataclass(frozen=True)
class Prescribed:
    """PRESCRIBE's result: the valid suggestions, and the dropped ones with their reasons."""

    valid: Tuple[Dict[str, Any], ...]
    dropped: Tuple[Dict[str, Any], ...]

    def recorded(self) -> List[Dict[str, Any]]:
        """What the run's row keeps in ``prescriptions``: every valid suggestion, then each
        dropped one with its reason."""
        return [*self.valid, *self.dropped]


def settings_from_config() -> ChangeSettings:
    """The detectors' HARNESS_* thresholds from config.py."""
    from config import config

    return ChangeSettings(
        heartbeat_min_runs=config.HARNESS_HEARTBEAT_MIN_RUNS,
        heartbeat_max_met=config.HARNESS_HEARTBEAT_MAX_MET,
        min_samples=config.HARNESS_MIN_SAMPLES,
        tool_fail_wilson_min=config.HARNESS_TOOL_FAIL_WILSON_MIN,
    )


# ------------------------------------------------------------------ detectors


def no_detector(_evidence: Evidence) -> Tuple[Proposal, ...]:
    """The detector of a kind PRESCRIBE does not detect (see the module docstring)."""
    return ()


def doubled_interval(minutes: int) -> int:
    """Twice ``minutes``, clamped as configure_agent_heartbeat clamps it."""
    doubled = minutes * HEARTBEAT_INTERVAL_FACTOR
    return max(HEARTBEAT_MIN_INTERVAL_MINUTES, min(HEARTBEAT_MAX_INTERVAL_MINUTES, doubled))


def _heartbeat_proposal(row: Mapping[str, Any], settings: ChangeSettings) -> Optional[Proposal]:
    """US-050: an enabled heartbeat whose judged week has at least
    HARNESS_HEARTBEAT_MIN_RUNS runs with an outcome (a None outcome is left out of the
    count) and a Wilson upper bound of the share that met their objective below
    HARNESS_HEARTBEAT_MAX_MET gets its interval doubled. Fewer runs is INSUFFICIENT_DATA:
    nothing is proposed. Nor is a doubling the clamp would not lengthen."""
    if not row["heartbeat_enabled"]:
        return None
    met = row["heartbeat_objective_met"]
    judged = met + row["heartbeat_objective_missed"]
    if judged < settings.heartbeat_min_runs:
        return None
    bound = hv.wilson_upper_bound(met, judged)
    current = row["heartbeat_interval_minutes"]
    proposed = doubled_interval(current)
    if bound >= settings.heartbeat_max_met or proposed <= current:
        return None
    unknown = row["heartbeat_objective_unknown"]
    return Proposal(
        target_id=int(row["agent_id"]),
        target_name=row["agent_name"],
        current_value={INTERVAL_KEY: current},
        proposed_value={INTERVAL_KEY: proposed},
        rationale=(
            f"Heartbeat every {current} minutes → every {proposed} minutes: {met} of {judged} heartbeats "
            f"this week met their objective; even allowing for chance, fewer than "
            f"{settings.heartbeat_max_met:.0%} would. {unknown} reported nothing to judge and are not counted."
        ),
        expected_improvement=f"{row['agent_name']} runs half as many heartbeats",
        evidence={"window": row["iso_week"], "met": met, "judged": judged, "unknown": unknown,
                  "met_upper_bound": bound},
    )


def detect_heartbeat_tune(evidence: Evidence) -> Tuple[Proposal, ...]:
    """A heartbeat_tune for each agent whose heartbeat rarely met its objective in the
    judged week. The roster COLLECT reads holds no system agent, and the validator drops
    any suggestion for the workspace's Auto."""
    metrics = evidence.metrics
    rows = (row for row in metrics.agent_rows if row["iso_week"] == metrics.window.cur_week)
    proposals = (_heartbeat_proposal(row, evidence.settings) for row in rows)
    return tuple(proposal for proposal in proposals if proposal is not None)


def detect_reconnects(evidence: Evidence) -> Tuple[Proposal, ...]:
    """A "reconnect <app>" for each reconnect_app issue DIAGNOSE found (US-019): the
    owner reconnects the app, so no agent setting changes."""
    return tuple(
        Proposal(
            target_id=int(issue["agent_id"]),
            target_name=issue["agent_name"],
            current_value={APP_KEY: issue["app_name"]},
            proposed_value={APP_KEY: issue["app_name"]},
            rationale=issue["detail"],
            expected_improvement=f"{issue['agent_name']}'s {issue['app_name']} calls stop failing",
            evidence={"calls": issue["calls"], "failed_calls": issue["failed_calls"],
                      "reconnect_calls": issue["reconnect_calls"]},
        )
        for issue in evidence.diagnosis.get("issues", [])
        if issue.get("root_cause") == RECONNECT_APP
    )


@dataclass(frozen=True)
class AppLinks:
    """From COLLECT's app links (US-057): the (agent id, app) pairs assigned, and for each
    pair what needs the app, as the report names it."""

    assigned: FrozenSet[Tuple[int, str]]
    needed: Mapping[Tuple[int, str], Tuple[str, ...]]


def app_links(metrics: HarnessMetrics) -> AppLinks:
    """The run's :class:`AppLinks`."""
    assigned = frozenset((link["agent_id"], link["app_name"]) for link in metrics.app_links
                         if link["link"] == LINK_ASSIGNED)
    needed: Dict[Tuple[int, str], List[str]] = {}
    for link in metrics.app_links:
        if link["link"] in NEEDED_BY:
            name = NEEDED_BY[link["link"]].format(source=link["source"])
            needed.setdefault((link["agent_id"], link["app_name"]), []).append(name)
    return AppLinks(assigned, {pair: tuple(names) for pair, names in needed.items()})


def _kept_because(needs: Tuple[str, ...], agent: str, app: str) -> Optional[str]:
    """Why a removal stays a report line: what needs the app (None when nothing does)."""
    if not needs:
        return None
    return f"{app} is needed by {' and '.join(needs)}, which {'routes' if len(needs) == 1 else 'route'} to {agent}"


def _removal(issue: Mapping[str, Any], links: AppLinks, evidence: Evidence) -> Optional[Proposal]:
    """US-057: a removal of a failing app assigned to the agent, when its calls number at least
    HARNESS_MIN_SAMPLES and the failures that are not the app's sign-in or connection have a
    Wilson lower bound of at least HARNESS_TOOL_FAIL_WILSON_MIN; kept a report line while a
    Playbook or active Mission that routes to the agent needs the app."""
    agent_id, agent, app = int(issue["agent_id"]), issue["agent_name"], str(issue["app_name"]).upper()
    if app in PLATFORM_APPS or (agent_id, app) not in links.assigned:
        return None
    calls, failed = issue["calls"], issue["failed_calls"] - issue["reconnect_calls"]
    bound = hv.wilson_lower_bound(failed, calls)
    if calls < evidence.settings.min_samples or bound < evidence.settings.tool_fail_wilson_min:
        return None
    needs = links.needed.get((agent_id, app), ())
    kept = _kept_because(needs, agent, app)
    rationale = (f"{failed} of {agent}'s {calls} {app} calls this week failed, not counting failures of "
                 f"{app}'s sign-in or connection; even allowing for chance, at least {bound:.0%} fail.")
    return Proposal(
        target_id=agent_id, target_name=agent,
        current_value={APP_KEY: app, ASSIGNED_KEY: True}, proposed_value={APP_KEY: app, ASSIGNED_KEY: False},
        rationale=f"{rationale} Reported only: {kept}." if kept else rationale,
        expected_improvement=f"{agent} stops calling {app}, which fails",
        evidence={"window": evidence.metrics.window.cur_week, "calls": calls, "failed_calls": failed,
                  "reconnect_calls": issue["reconnect_calls"], "failure_lower_bound": bound, "needed_by": list(needs)},
        report_only=kept,
    )


def detect_tool_removals(evidence: Evidence) -> Tuple[Proposal, ...]:
    """A tool_assignment_remove for each of DIAGNOSE's tool failures (never a reconnect
    line) that passes the gate (:func:`_removal`)."""
    links = app_links(evidence.metrics)
    issues = (issue for issue in evidence.diagnosis.get("issues", []) if issue.get("root_cause") == TOOL_FAILURE)
    proposals = (_removal(issue, links, evidence) for issue in issues)
    return tuple(proposal for proposal in proposals if proposal is not None)


def detect_rollbacks(evidence: Evidence) -> Tuple[Proposal, ...]:
    """The rollbacks the post-change check suggests this run (US-056), from DIAGNOSE's result."""
    from services.harness_post_change import rollbacks_of

    return rollbacks_of(evidence.diagnosis)


# ------------------------------------------------------------------ params and rollback builders


def _interval(value: Mapping[str, Any]) -> int:
    """A value's heartbeat interval: whole minutes inside the handler's clamp, else ValueError."""
    minutes = value.get(INTERVAL_KEY) if isinstance(value, Mapping) else None
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError(f"{INTERVAL_KEY} must be whole minutes, got {minutes!r}")
    if not HEARTBEAT_MIN_INTERVAL_MINUTES <= minutes <= HEARTBEAT_MAX_INTERVAL_MINUTES:
        raise ValueError(f"{INTERVAL_KEY} must be {HEARTBEAT_MIN_INTERVAL_MINUTES}-"
                         f"{HEARTBEAT_MAX_INTERVAL_MINUTES}, got {minutes}")
    return minutes


def _app(value: Mapping[str, Any]) -> str:
    """A value's app name, upper-cased as the assignment handlers store it, else ValueError."""
    app = value.get(APP_KEY) if isinstance(value, Mapping) else None
    if not isinstance(app, str) or not app.strip():
        raise ValueError(f"{APP_KEY} must name an app, got {app!r}")
    return app.strip().upper()


def heartbeat_params(agent_id: int, value: Mapping[str, Any]) -> InnerCall:
    """Set the agent's heartbeat interval: the change, and the rollback to its prior value."""
    return InnerCall(CONFIGURE_HEARTBEAT, {"agent_id": agent_id, INTERVAL_KEY: _interval(value)})


def unassign_params(agent_id: int, value: Mapping[str, Any]) -> InnerCall:
    """Take the app from the agent; the handler deactivates the assignment, keeping the row."""
    return InnerCall(UNASSIGN_TOOL, {"agent_id": agent_id, APP_KEY: _app(value)})


def assign_params(agent_id: int, value: Mapping[str, Any]) -> InnerCall:
    """Give the app back to the agent: the handler re-activates the deactivated assignment."""
    return InnerCall(ASSIGN_TOOL, {"agent_id": agent_id, APP_KEY: _app(value)})


def _rolled_back(value: Mapping[str, Any]) -> Tuple[ChangeKind, Mapping[str, Any]]:
    """A rollback's proposed value: the apply kind it undoes and the prior value it restores."""
    name = value.get(ROLLBACK_OF_KEY) if isinstance(value, Mapping) else None
    kind = KINDS.get(name) if isinstance(name, str) else None
    if kind is None or not kind.applies or kind.name == ROLLBACK:
        raise ValueError(f"a rollback undoes a change of another apply kind, got {name!r}")
    prior = value.get(ROLLBACK_VALUE_KEY)
    if not isinstance(prior, Mapping):
        raise ValueError(f"a rollback of {name} needs the prior value to restore")
    return kind, prior


def rollback_params(agent_id: int, value: Mapping[str, Any]) -> InnerCall:
    """The undo call of the change the rollback reverses."""
    kind, prior = _rolled_back(value)
    return kind.rollback_builder(agent_id, prior)


def never_rolled_back(_agent_id: int, _value: Mapping[str, Any]) -> InnerCall:
    """A rollback is never rolled back (FR-16)."""
    raise ValueError("a rollback is never rolled back")


# ------------------------------------------------------------------ prior-value readers


def _workspace(workspace_id: Union[UUID, str]) -> UUID:
    return UUID(str(workspace_id))


def read_heartbeat_interval(db: Session, workspace_id: Union[UUID, str], agent_id: int,
                            _value: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The agent's ``configuration.heartbeat.interval_minutes`` as COLLECT and the heartbeat
    service read it (60 when unset); None when the agent is not in this workspace."""
    from core.models import Agent

    row = (db.query(Agent.configuration)
           .filter(Agent.workspace_id == _workspace(workspace_id), Agent.id == agent_id)
           .first())
    if row is None:
        return None
    configuration = row[0] if isinstance(row[0], dict) else {}
    heartbeat = heartbeat_setting(configuration.get("heartbeat"), agent_id)
    return {INTERVAL_KEY: heartbeat_interval(heartbeat, agent_id)}


def read_tool_assignment(db: Session, workspace_id: Union[UUID, str], agent_id: int,
                         value: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Whether the agent holds an active assignment of the value's app; None when the agent
    is not in this workspace."""
    from core.models import Agent
    from core.models.composio_cache import AgentAppAssignment

    app = _app(value)
    row = (db.query(Agent.id, AgentAppAssignment.is_active)
           .outerjoin(AgentAppAssignment,
                      and_(AgentAppAssignment.agent_id == Agent.id, AgentAppAssignment.app_name == app))
           .filter(Agent.workspace_id == _workspace(workspace_id), Agent.id == agent_id)
           .first())
    if row is None:
        return None
    return {APP_KEY: app, ASSIGNED_KEY: row[1] is True}


def read_rolled_back(db: Session, workspace_id: Union[UUID, str], agent_id: int,
                     value: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The setting a rollback restores, read by the reader of the kind it undoes."""
    kind, prior = _rolled_back(value)
    return kind.prior_value_reader(db, workspace_id, agent_id, prior)


# ------------------------------------------------------------------ the registry

KINDS: Dict[str, ChangeKind] = {kind.name: kind for kind in (
    ChangeKind(HEARTBEAT_TUNE, detect_heartbeat_tune, risk=2, inner_actions=(CONFIGURE_HEARTBEAT,),
               params_builder=heartbeat_params, predicted_metric=HEARTBEAT_OBJECTIVE,
               guard_metrics=GUARD_METRICS, prior_value_reader=read_heartbeat_interval,
               rollback_builder=heartbeat_params),
    ChangeKind(TOOL_ASSIGNMENT_REMOVE, detect_tool_removals, risk=3, inner_actions=(UNASSIGN_TOOL,),
               params_builder=unassign_params, predicted_metric=APP_TOOL_FAILURE,
               guard_metrics=GUARD_METRICS, prior_value_reader=read_tool_assignment,
               rollback_builder=assign_params),
    # A rollback runs the undo call of the kind it reverses, so its actions are theirs. It
    # restores the agent as it was; its card names the metric that fell (US-056).
    ChangeKind(ROLLBACK, detect_rollbacks, risk=1, inner_actions=(CONFIGURE_HEARTBEAT, ASSIGN_TOOL),
               params_builder=rollback_params, predicted_metric=TASK_SUCCESS,
               guard_metrics=GUARD_METRICS, prior_value_reader=read_rolled_back,
               rollback_builder=never_rolled_back),
    # Changes no agent setting: the owner reconnects the app.
    ChangeKind(RECONNECT_APP, detect_reconnects, risk=1),
    ChangeKind("model", no_detector, risk=3),
    ChangeKind("description", no_detector, risk=1),
    ChangeKind("temperature", no_detector, risk=2),
    ChangeKind("tags", no_detector, risk=1),
    # Routing rules and power mode are admin-only settings.
    ChangeKind("routing", no_detector, risk=3),
    ChangeKind("power_mode", no_detector, risk=3),
)}
APPLY_KINDS: Tuple[str, ...] = tuple(name for name, kind in KINDS.items() if kind.applies)
REPORT_KINDS: Tuple[str, ...] = tuple(name for name, kind in KINDS.items() if not kind.applies)


# ------------------------------------------------------------------ PRESCRIBE


def _apply_drop_reason(kind: ChangeKind, rx: Mapping[str, Any]) -> Optional[str]:
    """Why an apply-kind suggestion cannot be raised: no prior value, no change, or a
    proposed value its params builder refuses (a placeholder never passes)."""
    current, proposed = rx.get("current_value"), rx.get("proposed_value")
    if not current:
        return DROP_NO_PRIOR
    if proposed == current:
        return DROP_NO_CHANGE
    try:
        kind.params_builder(rx["target_id"], proposed)
    except ValueError as exc:
        return f"invalid proposed value: {exc}"
    return None


def drop_reason(rx: Mapping[str, Any], auto_agent_id: Optional[int]) -> Optional[str]:
    """ORG-7: why a suggestion is invalid, or None when it is valid. The registry never
    targets the workspace's Auto (FR-13)."""
    kind = KINDS.get(rx.get("change_type"))
    if kind is None:
        return DROP_UNKNOWN_KIND
    target = rx.get("target_id")
    if isinstance(target, bool) or not isinstance(target, int):
        return DROP_NO_TARGET
    if auto_agent_id is not None and target == auto_agent_id:
        return DROP_AUTO
    return _apply_drop_reason(kind, rx) if kind.applies else None


def _prescription(kind: ChangeKind, proposal: Proposal, rx_id: str) -> Dict[str, Any]:
    """A proposal with its id and its kind's risk and metrics: one entry of the run's row. A
    suggestion its kind's gate keeps a report line is no apply kind, and says why."""
    return {
        "prescription_id": rx_id,
        "target_type": TARGET_AGENT,
        "target_id": proposal.target_id,
        "target_name": proposal.target_name,
        "change_type": kind.name,
        "apply_kind": kind.applies and proposal.report_only is None,
        **({REPORT_ONLY_KEY: proposal.report_only} if proposal.report_only else {}),
        "current_value": dict(proposal.current_value),
        "proposed_value": dict(proposal.proposed_value),
        "risk_score": kind.risk,
        "predicted_metric": kind.predicted_metric,
        "guard_metrics": list(kind.guard_metrics),
        "evidence": dict(proposal.evidence),
        "expected_improvement": proposal.expected_improvement,
        "rationale": proposal.rationale,
    }


def _dropped(rx: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """A dropped suggestion as the run's row keeps it: its id, kind, target and reason."""
    keep = ("prescription_id", "change_type", "target_id", "target_name")
    return {**{key: rx[key] for key in keep}, DROPPED_KEY: reason}


def prescribe(evidence: Evidence, run_id: int, auto_agent_id: Optional[int]) -> Prescribed:
    """ORG-6 and ORG-7: every kind's detector over the run's evidence, then the validator.

    Ids are ``rx-<run_id>-<seq>`` in registry order, from the run's harness_runs id, so
    two runs on one day stay apart; a dropped suggestion keeps its id with its reason.
    ``auto_agent_id`` is the workspace's Auto (None when it has none).
    """
    found = [(kind, proposal) for kind in KINDS.values() for proposal in kind.detector(evidence)]
    valid: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for seq, (kind, proposal) in enumerate(found, 1):
        rx = _prescription(kind, proposal, f"rx-{run_id}-{seq:03d}")
        reason = drop_reason(rx, auto_agent_id)
        if reason is None:
            valid.append(rx)
        else:
            dropped.append(_dropped(rx, reason))
    return Prescribed(valid=tuple(valid), dropped=tuple(dropped))


__all__ = [
    "APPLY_KINDS",
    "AppLinks",
    "ChangeKind",
    "ChangeSettings",
    "Evidence",
    "InnerCall",
    "KINDS",
    "Prescribed",
    "Proposal",
    "REPORT_KINDS",
    "app_links",
    "detect_rollbacks",
    "detect_tool_removals",
    "doubled_interval",
    "drop_reason",
    "prescribe",
    "settings_from_config",
]
