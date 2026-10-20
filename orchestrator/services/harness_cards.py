"""PRD-263 US-054: what the owner reads of each change HARNESS proposes (section 8).

:func:`question` is an Approvals card's text (services.harness_grants raises the card):
five parts, in order, in plain words, with no tool or parameter names: what changes;
why (the counts behind it, and how many agents HARNESS judged); what approving does;
how a change that goes wrong is undone; and when the card expires, true because the
apply action refuses a lapsed grant (US-052) and the daily sweep closes it (US-053).
:func:`evidence_ticket` is the run's one board card: each proposal, its numbers and its
card. Text only: no write, no platform action, no model call (FR-12).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from services.harness_changes import APP_KEY, HEARTBEAT_TUNE, INTERVAL_KEY, ROLLBACK, TOOL_ASSIGNMENT_REMOVE

SECONDS_PER_HOUR = 3600
SECONDS_PER_MINUTE = 60

CHANGE = "Change: {change}."
WHY = "Why: {why}"
APPROVE = "If you approve: {approve}"
UNDO = "If it goes wrong: {undo}"
EXPIRES = ("Expires: {ttl} from now. After that, approving does nothing, and the card closes "
           "within a day.")

HEARTBEAT_CHANGE = "{agent}'s heartbeat, every {before} minutes → every {after} minutes"
HEARTBEAT_APPROVE = "only this setting changes, now, and only if it still reads every {before} minutes."
HEARTBEAT_WHY = ("{met} of {judged} heartbeats this week met their objective. Even allowing for chance, "
                 "fewer than {share} would.")
HEARTBEAT_UNKNOWN = "{unknown} heartbeats reported nothing to judge and are not counted."
REMOVE_CHANGE = "take {app} away from {agent}"
REMOVE_APPROVE = "only this assignment changes, now, and only if {agent} still has {app}."
ROLLBACK_CHANGE = "undo the change HARNESS made to {agent}"
ROLLBACK_APPROVE = "only that setting goes back, now, and only if it still reads as HARNESS left it."
ROLLBACK_UNDO = "this card undoes an earlier HARNESS change and is not itself undone."
WATCHED_UNDO = ("HARNESS compares this agent's next weeks with the {weeks} weeks before the change, "
                "and sends a separate card to undo it if its Tasks or tool errors get worse.")
COVERAGE = "HARNESS judged {judged} of {active} agents this week."

EVIDENCE_TITLE = "HARNESS proposes {count} change{plural} for {week}"
EVIDENCE_OPENING = ("HARNESS judged {week} and proposes the changes below. Each has its own Approvals "
                    "card, on this ticket and in Command Center > Governance > Approvals, and nothing "
                    "changes until you approve one. Each card expires {ttl} after it was raised.")
EVIDENCE_LINE = "- **{agent}**: {change}. {why} {card}"
CARD_RAISED = "(Approvals card #{grant_id})"
CARD_NOT_RAISED = "(its Approvals card could not be raised; HARNESS proposes it again next week if it still holds)"
SETTING_MOVED = "(no card: the setting changed while HARNESS ran)"
EVIDENCE_CLOSING = ("Approving or cancelling this ticket on the board only closes it: it changes no "
                    "setting. Dragging it to Blocked stops its cards: approving one then changes nothing.")


@dataclass(frozen=True)
class CardSettings:
    """What the card text reads from config.py: the heartbeat detector's share, the
    reference the post-change check compares with, and how long a grant lives."""

    heartbeat_max_met: float
    reference_weeks: int
    ttl_seconds: int


@dataclass(frozen=True)
class Outcome:
    """One proposal on the evidence ticket: its prescription, and its card's grant id
    (None when the card could not be raised) or why it got none."""

    prescription: Mapping[str, Any]
    grant_id: Optional[int] = None
    no_card: Optional[str] = None


def settings_from_config() -> CardSettings:
    """The card text's values: HARNESS_HEARTBEAT_MAX_MET, HARNESS_REF_DAYS in weeks, and the
    TTL every approval grant is created with (APPROVAL_GRANT_TTL_SECONDS)."""
    from config import config
    from core.services.approval_grants import DEFAULT_TTL_SECONDS
    from services.harness_diagnose import reference_weeks

    return CardSettings(config.HARNESS_HEARTBEAT_MAX_MET, reference_weeks(), DEFAULT_TTL_SECONDS)


def ttl_words(seconds: int) -> str:
    """'24 hours', '1 hour' or '90 minutes'."""
    if seconds % SECONDS_PER_HOUR == 0:
        hours = seconds // SECONDS_PER_HOUR
        return f"{hours} hour{'' if hours == 1 else 's'}"
    return f"{max(1, seconds // SECONDS_PER_MINUTE)} minutes"


def _value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def change_and_approve(rx: Mapping[str, Any]) -> Tuple[str, str]:
    """What changes, and what approving does, for a prescription of an apply kind."""
    agent, kind = rx["target_name"], rx["change_type"]
    before, after = rx.get("current_value"), rx.get("proposed_value")
    if kind == HEARTBEAT_TUNE:
        minutes = _value(before, INTERVAL_KEY)
        return (HEARTBEAT_CHANGE.format(agent=agent, before=minutes, after=_value(after, INTERVAL_KEY)),
                HEARTBEAT_APPROVE.format(before=minutes))
    if kind == TOOL_ASSIGNMENT_REMOVE:
        app = _value(before, APP_KEY) or _value(after, APP_KEY)
        return REMOVE_CHANGE.format(app=app, agent=agent), REMOVE_APPROVE.format(agent=agent, app=app)
    if kind == ROLLBACK:
        return ROLLBACK_CHANGE.format(agent=agent), ROLLBACK_APPROVE
    raise ValueError(f"{kind!r} is not a change HARNESS raises a card for")


def counts_behind(rx: Mapping[str, Any], settings: CardSettings) -> str:
    """The counts behind one change: a heartbeat's judged runs and how many met their
    objective; for any other kind, the reason its detector gave."""
    evidence = rx.get("evidence") if isinstance(rx.get("evidence"), Mapping) else {}
    if rx["change_type"] != HEARTBEAT_TUNE or not {"met", "judged"} <= set(evidence):
        return str(rx.get("rationale") or "").strip()
    said = HEARTBEAT_WHY.format(met=evidence["met"], judged=evidence["judged"],
                                share=f"{settings.heartbeat_max_met:.0%}")
    if evidence.get("unknown"):
        said = f"{said} {HEARTBEAT_UNKNOWN.format(unknown=evidence['unknown'])}"
    return said


def coverage(diagnosis: Mapping[str, Any]) -> str:
    """How many agents HARNESS judged this week; '' when it had none to judge."""
    counts = diagnosis.get("counts") if isinstance(diagnosis.get("counts"), Mapping) else {}
    if not counts.get("active_agents"):
        return ""
    return COVERAGE.format(judged=counts.get("judged_agents", 0), active=counts["active_agents"])


def why(rx: Mapping[str, Any], diagnosis: Mapping[str, Any], settings: CardSettings) -> str:
    """The counts behind the change, then how many agents HARNESS judged this week."""
    return " ".join(part for part in (counts_behind(rx, settings), coverage(diagnosis)) if part)


def question(rx: Mapping[str, Any], diagnosis: Mapping[str, Any], settings: CardSettings) -> str:
    """The Approvals card's text, its five parts one to a line (the card keeps line breaks)."""
    change, approve = change_and_approve(rx)
    undo = ROLLBACK_UNDO if rx["change_type"] == ROLLBACK else WATCHED_UNDO.format(weeks=settings.reference_weeks)
    return "\n".join((
        CHANGE.format(change=change),
        WHY.format(why=why(rx, diagnosis, settings)),
        APPROVE.format(approve=approve),
        UNDO.format(undo=undo),
        EXPIRES.format(ttl=ttl_words(settings.ttl_seconds)),
    ))


def evidence_title(week: str, count: int) -> str:
    """The evidence ticket's title: how many changes HARNESS proposes, and for which week."""
    return EVIDENCE_TITLE.format(count=count, plural="" if count == 1 else "s", week=week)


def _evidence_line(outcome: Outcome, settings: CardSettings) -> str:
    rx = outcome.prescription
    if outcome.grant_id is not None:
        card = CARD_RAISED.format(grant_id=outcome.grant_id)
    else:
        card = outcome.no_card or CARD_NOT_RAISED
    change, _ = change_and_approve(rx)
    return EVIDENCE_LINE.format(agent=rx["target_name"], change=change, why=counts_behind(rx, settings), card=card)


def evidence_ticket(week: str, outcomes: Sequence[Outcome], diagnosis: Mapping[str, Any],
                    settings: CardSettings) -> str:
    """The evidence ticket's description: what it is, how many agents HARNESS judged, each
    proposal with its numbers and its card, and what the ticket's own Approve does."""
    opening = EVIDENCE_OPENING.format(week=week, ttl=ttl_words(settings.ttl_seconds))
    return "\n\n".join((
        " ".join(part for part in (opening, coverage(diagnosis)) if part),
        "\n".join(_evidence_line(outcome, settings) for outcome in outcomes),
        EVIDENCE_CLOSING,
    ))


__all__ = ["CardSettings", "Outcome", "SETTING_MOVED", "change_and_approve", "counts_behind", "coverage",
           "evidence_ticket", "evidence_title", "question", "settings_from_config", "ttl_words"]
