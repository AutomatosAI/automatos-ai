"""PRD-248 S5 — four more decision points, judged beside the platform (shadow only).

Each hook is one async function that takes plain values (never an ORM row),
asks the engine its questions, and writes one shadow row with what the
platform decided next to what the engine would have. Call sites stay tiny:
they check the dial, gather values, and hand the coroutine to
``engine.shadow()``. Nothing here changes any decision.

* ``ticket_assign``  — which agent gets a mission task (beside AgentMatcher.rank)
* ``session_end``    — is a Claude Code session's work finished (beside apply_result)
* ``hold_risk``      — how risky is a held command or ask (beside create_grant)
* ``report_triage``  — does a report or heartbeat need the owner today (beside dispatch)

Question wording follows the vendor's own failure-mode list: direct, no
negations, no arithmetic, the state trimmed to the fields the question needs.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .questions import CHOICE_MAX_OPTIONS, Choice, DecisionResult, Noul, Question, Score

PURPOSE_ASSIGN = "ticket_assign"
PURPOSE_SESSION_END = "session_end"
PURPOSE_HOLD = "hold_risk"
PURPOSE_REPORT = "report_triage"

PREVIEW_CHARS = 200
TEXT_MAX_CHARS = 3000
DESCRIPTION_MAX_CHARS = 120
NONE_OPTION = "none"

RISK_LEVELS: Sequence[str] = (
    "Reads only its own files or lists things; nothing changes.",
    "Writes files inside its own working folder for this ticket.",
    "Reads or writes files outside its own working folder.",
    "Reaches the network or a connected app (email, calendar, shop, payments).",
    "Destructive or irreversible: deletes, publishes, sends, pays, changes accounts or the system.",
)

INTENT_CRITERIA: Dict[str, str] = {
    "inspect_files": "Look at, list, search or diff files; no change.",
    "edit_own_files": "Create or change files inside the ticket's own folder.",
    "run_or_build": "Run tests, a build, a script or a local program.",
    "fetch_from_network": "Download, call an API, or read from a connected app.",
    "change_system_or_data": "Install, configure, delete, publish, send, pay, or change shared data.",
    "unclear": "The text does not say enough to tell.",
}

SEVERITY_CRITERIA: Dict[str, str] = {
    "fyi": "Routine; nothing for the owner to do.",
    "worth_a_look": "Useful to read this week; no decision needed.",
    "needs_a_decision": "The owner has to decide or approve something.",
    "urgent": "Something is broken, losing money, or time-critical today.",
}


def _preview(text: Any, n: int = PREVIEW_CHARS) -> str:
    return (str(text or ""))[:n]


def _receipt(row: Dict[str, Any], result: Optional[DecisionResult], started: float) -> None:
    if result is None:
        row["error"] = "no_result"
    else:
        row.update(
            provider=result.provider,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            answers={k: a.to_dict() for k, a in result.answers.items()},
        )
    row["shadow_ms"] = int((time.monotonic() - started) * 1000)


async def _ask_and_record(
    engine: Any,
    *,
    purpose: str,
    row: Dict[str, Any],
    state: Any,
    questions: Mapping[str, Question],
    workspace_id: Any,
    finish: Any = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    try:
        result = await engine.decide(
            state=state, questions=questions, workspace_id=workspace_id, purpose=purpose
        )
        _receipt(row, result, started)
        if result is not None and finish is not None:
            finish(row, result)
    except Exception as exc:  # noqa: BLE001 — a shadow never surfaces
        row["error"] = f"{exc!r}"[:200]
    try:
        engine.record_shadow(row)
    except Exception:  # noqa: BLE001
        pass
    return row


# ---------------------------------------------------------------------------
# ticket_assign — which agent gets a mission task
# ---------------------------------------------------------------------------


def assignment_options(candidates: Sequence[Tuple[str, str]]) -> Dict[str, Optional[str]]:
    seen: Dict[str, Optional[str]] = {}
    for name, description in candidates:
        key = (name or "").strip()
        if key and key.lower() != NONE_OPTION and key not in seen:
            seen[key] = (description or "").strip()[:DESCRIPTION_MAX_CHARS] or None
        if len(seen) >= CHOICE_MAX_OPTIONS - 1:
            break
    return seen


def assignment_questions(candidates: Sequence[Tuple[str, str]]) -> Dict[str, Question]:
    options = assignment_options(candidates)
    if len(options) < 1:
        raise ValueError("an assignment needs at least one candidate")
    options[NONE_OPTION] = "No listed agent fits this work."
    return {
        "assignee": Choice(
            "Which agent should do this task? Pick by the task's needs and the agent's role.",
            options,
        )
    }


def assignment_state(
    *, title: str, description: str, role: Optional[str], required_tools: Sequence[str]
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"task": _preview(title), "details": _preview(description, TEXT_MAX_CHARS)}
    if role:
        state["role_wanted"] = str(role)
    if required_tools:
        state["tools_needed"] = [str(t) for t in required_tools][:20]
    return state


async def shadow_assignment(
    engine: Any,
    *,
    workspace_id: Any,
    task_id: Any,
    title: str,
    description: str,
    role: Optional[str],
    required_tools: Sequence[str],
    candidates: Sequence[Tuple[str, str]],
    platform_ranked: Sequence[str],
) -> Dict[str, Any]:
    ranked = [str(n) for n in platform_ranked if n]
    row: Dict[str, Any] = {
        "purpose": PURPOSE_ASSIGN,
        "workspace_id": workspace_id,
        "task_id": task_id,
        "title": _preview(title),
        "role": role,
        "candidates": len(assignment_options(candidates)),
        "platform_top": ranked[0] if ranked else None,
        "platform_ranked": ranked[:5],
    }
    try:
        questions = assignment_questions(candidates)
    except ValueError as exc:
        row["error"] = str(exc)
        engine.record_shadow(row)
        return row

    def finish(r: Dict[str, Any], result: DecisionResult) -> None:
        answer = result.get("assignee")
        if answer is None or not answer.choice:
            r["agree"] = None
            return
        pick = answer.choice
        r["jev_pick"] = pick
        r["jev_confidence"] = round(answer.certainty, 4)
        lower = [n.lower() for n in ranked]
        r["jev_pick_platform_rank"] = (lower.index(pick.lower()) + 1) if pick.lower() in lower else None
        r["agree"] = (bool(ranked) and pick.lower() == ranked[0].lower()) if pick != NONE_OPTION else (not ranked)

    return await _ask_and_record(
        engine, purpose=PURPOSE_ASSIGN, row=row,
        state=assignment_state(title=title, description=description, role=role, required_tools=required_tools),
        questions=questions, workspace_id=workspace_id, finish=finish,
    )


# ---------------------------------------------------------------------------
# session_end — is a Claude Code session's work finished
# ---------------------------------------------------------------------------


def session_end_questions() -> Dict[str, Question]:
    return {
        "work_complete": Noul("The final message reports the task's work as finished."),
        "nothing_done": Noul("The final message says the work was already done before, or that no action was taken."),
        "needs_owner": Noul("The final message asks the owner for a decision or information before it can continue."),
    }


def session_end_state(
    *,
    title: str,
    description: str,
    final_text: str,
    attempt: Any,
    exit_reason: Optional[str],
    files_touched: int,
    denials: int,
) -> Dict[str, Any]:
    return {
        "task": _preview(title),
        "brief": _preview(description, 1000),
        "final_message": _preview(final_text, TEXT_MAX_CHARS),
        "attempt": attempt,
        "ended_because": exit_reason or "",
        "files_written": int(files_touched or 0),
        "commands_refused": int(denials or 0),
    }


def session_end_verdict(result: DecisionResult) -> Optional[str]:
    complete = result.get("work_complete")
    nothing = result.get("nothing_done")
    owner = result.get("needs_owner")
    if owner is not None and owner.yes:
        return "needs_owner"
    if nothing is not None and nothing.yes:
        return "nothing_done"
    if complete is not None and complete.yes:
        return "complete"
    if complete is not None:
        return "incomplete"
    return None


async def shadow_session_end(
    engine: Any,
    *,
    workspace_id: Any,
    task_id: Any,
    attempt: Any,
    title: str,
    description: str,
    final_text: str,
    exit_reason: Optional[str],
    files_touched: int,
    denials: int,
    platform_status: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "purpose": PURPOSE_SESSION_END,
        "workspace_id": workspace_id,
        "task_id": task_id,
        "attempt": attempt,
        "title": _preview(title),
        "final_preview": _preview(final_text),
        "exit_reason": exit_reason,
        "files_touched": int(files_touched or 0),
        "denials": int(denials or 0),
        "platform_status": platform_status,
    }

    def finish(r: Dict[str, Any], result: DecisionResult) -> None:
        r["jev_verdict"] = session_end_verdict(result)

    return await _ask_and_record(
        engine, purpose=PURPOSE_SESSION_END, row=row,
        state=session_end_state(
            title=title, description=description, final_text=final_text, attempt=attempt,
            exit_reason=exit_reason, files_touched=files_touched, denials=denials,
        ),
        questions=session_end_questions(), workspace_id=workspace_id, finish=finish,
    )


# ---------------------------------------------------------------------------
# hold_risk — how risky is a held command or an ask
# ---------------------------------------------------------------------------


def hold_questions() -> Dict[str, Question]:
    return {
        "risk": Score("How far does what is being asked for reach?", list(RISK_LEVELS)),
        "intent": Choice("What is the agent trying to do?", INTENT_CRITERIA),
        "owner_can_judge": Noul(
            "A business owner with no technical knowledge could decide this from the text alone."
        ),
    }


def hold_state(
    *,
    kind: str,
    subject_type: str,
    tool_name: Optional[str],
    question_md: Optional[str],
    options: Optional[Sequence[Any]],
    reason: Optional[str],
) -> Dict[str, Any]:
    state: Dict[str, Any] = {"kind": kind, "about": subject_type, "asked": _preview(question_md, TEXT_MAX_CHARS)}
    if tool_name:
        state["tool"] = str(tool_name)
    if reason:
        state["why_it_was_raised"] = _preview(reason, 500)
    if options:
        state["answers_offered"] = [str(o) for o in list(options)[:10]]
    return state


async def shadow_hold(
    engine: Any,
    *,
    workspace_id: Any,
    grant_id: Any,
    kind: str,
    subject_type: str,
    subject_id: Any,
    tool_name: Optional[str],
    risk_tier: Optional[str],
    question_md: Optional[str],
    options: Optional[Sequence[Any]],
    reason: Optional[str],
    agent_id: Any = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "purpose": PURPOSE_HOLD,
        "workspace_id": workspace_id,
        "grant_id": grant_id,
        "kind": kind,
        "subject_type": subject_type,
        "subject_id": str(subject_id) if subject_id is not None else None,
        "agent_id": agent_id,
        "tool_name": tool_name,
        "platform_risk_tier": risk_tier,
        "asked_preview": _preview(question_md),
        "platform_decision": "ask",
    }

    def finish(r: Dict[str, Any], result: DecisionResult) -> None:
        risk = result.get("risk")
        if risk is not None and risk.score is not None:
            r["jev_risk_level"] = round(float(risk.score) + 1.0, 2)  # 1..5, level index + 1
            r["jev_risk_confidence"] = round(risk.certainty, 4)
        intent = result.get("intent")
        if intent is not None:
            r["jev_intent"] = intent.choice
        judge = result.get("owner_can_judge")
        if judge is not None and judge.noul is not None:
            r["jev_owner_can_judge"] = round(judge.noul, 4)

    return await _ask_and_record(
        engine, purpose=PURPOSE_HOLD, row=row,
        state=hold_state(
            kind=kind, subject_type=subject_type, tool_name=tool_name,
            question_md=question_md, options=options, reason=reason,
        ),
        questions=hold_questions(), workspace_id=workspace_id, finish=finish,
    )


# ---------------------------------------------------------------------------
# report_triage — does a report or heartbeat need the owner today
# ---------------------------------------------------------------------------


def report_questions() -> Dict[str, Question]:
    return {
        "needs_attention": Noul("The owner should act on this today."),
        "severity": Choice("How should the owner treat this?", SEVERITY_CRITERIA),
    }


def report_state(
    *,
    kind: str,
    title: str,
    summary: str,
    status: Optional[str],
    agent_name: Optional[str],
    report_type: Optional[str],
    action_items: int = 0,
    recommendations: int = 0,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "kind": kind,
        "from": agent_name or "",
        "title": _preview(title),
        "summary": _preview(summary, TEXT_MAX_CHARS),
        "status": status or "",
    }
    if report_type:
        state["report_type"] = str(report_type)
    if action_items:
        state["action_items"] = int(action_items)
    if recommendations:
        state["recommendations"] = int(recommendations)
    return state


async def shadow_report_triage(
    engine: Any,
    *,
    workspace_id: Any,
    kind: str,
    subject_id: Any,
    title: str,
    summary: str,
    status: Optional[str],
    agent_name: Optional[str],
    report_type: Optional[str],
    platform_action: str,
    action_items: int = 0,
    recommendations: int = 0,
    agent_id: Any = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "purpose": PURPOSE_REPORT,
        "workspace_id": workspace_id,
        "kind": kind,
        "subject_id": str(subject_id) if subject_id is not None else None,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "title": _preview(title),
        "status": status,
        "platform_action": platform_action,
    }

    def finish(r: Dict[str, Any], result: DecisionResult) -> None:
        attention = result.get("needs_attention")
        if attention is not None and attention.noul is not None:
            r["jev_needs_attention"] = round(attention.noul, 4)
        severity = result.get("severity")
        if severity is not None:
            r["jev_severity"] = severity.choice
            r["jev_severity_confidence"] = round(severity.certainty, 4)

    return await _ask_and_record(
        engine, purpose=PURPOSE_REPORT, row=row,
        state=report_state(
            kind=kind, title=title, summary=summary, status=status, agent_name=agent_name,
            report_type=report_type, action_items=action_items, recommendations=recommendations,
        ),
        questions=report_questions(), workspace_id=workspace_id, finish=finish,
    )
