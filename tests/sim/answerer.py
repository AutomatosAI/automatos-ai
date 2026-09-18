"""The human at the other end (PRD-247 D6 / S0.5).

Agents ask questions and request approvals through PRD-225's grants
(``GET /api/v1/approval-grants?status=pending``). The answerer plays the
customer: it answers ``kind=question`` grants from a persona's script and, in a
sim workspace only, grants the rest — a customer approves their own asks.
Every decision is returned to the caller and lands in the run record.

The client sends the sim workspace's id on every call and the grants routes
filter on it server-side; the runner stops before filing anything if that
scoping is not honoured (``workspace.assert_scoped``). Nothing here can reach
a workspace the run did not create by any path other than that header.

Personas live in ``~/.automatos-sim/personas.toml`` (local, never committed);
``tests/sim/personas.example.toml`` shows the shape. Without one, the default
persona below answers like a careful small-business owner.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import Api, ApiError, items_of
from .config import PERSONAS_FILE

QUESTION_KIND = "question"
QUESTION_FIELDS = ("question", "prompt", "summary", "reason", "description", "title")

DEFAULT_PERSONA: dict[str, Any] = {
    "name": "default",
    "default_answer": "Use your best judgement and carry on. Keep it simple, and don't spend money "
                      "or contact anyone outside the company.",
    "answers": [
        {"match": ["budget", "spend", "cost", "price", "pay"],
         "answer": "Keep it under 50 pounds and check with me before spending anything."},
        {"match": ["deadline", "when do you need", "by when", "due"],
         "answer": "End of this week is fine."},
        {"match": ["format", "file", "document", "deliver", "pdf", "slides"],
         "answer": "A short markdown document is fine. Save it as a deliverable."},
        {"match": ["email", "send", "contact", "message", "post", "publish"],
         "answer": "Don't send or publish anything. Draft it and I'll review it first."},
        {"match": ["which", "prefer", "option", "choose", "or "],
         "answer": "Go with the simplest option that gets the job done."},
        {"match": ["access", "credential", "login", "password", "api key", "connect"],
         "answer": "You don't have access to that. Work with what you can reach and note the gap."},
    ],
}


def load_persona(name: str | None, path: Path = PERSONAS_FILE) -> dict[str, Any]:
    """``[personas.<name>]`` from the local file; the default persona when absent."""
    if not name or name == "default" or not path.exists():
        return DEFAULT_PERSONA
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    persona = (raw.get("personas") or {}).get(name)
    if not isinstance(persona, Mapping):
        return DEFAULT_PERSONA
    return {"name": name, "default_answer": str(persona.get("default_answer") or DEFAULT_PERSONA["default_answer"]),
            "answers": list(persona.get("answers") or [])}


def question_text(grant: Mapping[str, Any]) -> str:
    """Whatever field the grant used to carry the question, joined for matching."""
    parts = [str(grant.get(f)) for f in QUESTION_FIELDS if grant.get(f)]
    context = grant.get("context") or grant.get("metadata")
    if isinstance(context, Mapping):
        parts.extend(str(context.get(f)) for f in QUESTION_FIELDS if context.get(f))
    return " ".join(parts)


def pick_answer(persona: Mapping[str, Any], question: str, options: Sequence[str] | None = None) -> dict[str, str]:
    """An offered option that matches a script line wins; then keyword lines; then the default."""
    lowered = question.lower()
    for line in persona.get("answers") or []:
        keys = [str(k).lower() for k in line.get("match") or []]
        if not any(k in lowered for k in keys):
            continue
        answer = str(line.get("answer") or "")
        for option in options or ():
            if any(k in str(option).lower() for k in keys):
                return {"option": str(option)}
        return {"answer_text": answer}
    if options:
        return {"option": str(options[0])}
    return {"answer_text": str(persona.get("default_answer") or DEFAULT_PERSONA["default_answer"])}


def answer_pending(api: Api, persona: Mapping[str, Any], *, auto_grant: bool = True,
                   label: str = "answer") -> tuple[dict[str, Any], ...]:
    """One pass over pending grants; returns one record per decision (errors included)."""
    try:
        payload = api.get("/api/v1/approval-grants", params={"status": "pending"}, label=f"{label}:list")
    except ApiError as exc:
        return ({"error": f"list grants failed: {exc.status} {exc.body[:200]}"},)
    decisions = []
    for grant in items_of(payload, "grants"):
        if not isinstance(grant, Mapping) or "id" not in grant:
            continue
        decisions.append(_decide(api, persona, grant, auto_grant, label))
    return tuple(decisions)


def _decide(api: Api, persona: Mapping[str, Any], grant: Mapping[str, Any], auto_grant: bool,
            label: str) -> dict[str, Any]:
    kind = str(grant.get("kind") or "")
    question = question_text(grant)
    options = grant.get("options") if isinstance(grant.get("options"), list) else None
    record: dict[str, Any] = {"grant_id": grant["id"], "kind": kind, "question": question[:500], "options": options}
    try:
        if kind == QUESTION_KIND:
            answer = pick_answer(persona, question, options)
            api.post(f"/api/v1/approval-grants/{grant['id']}/answer", answer, label=f"{label}:answer")
            return {**record, "action": "answered", **answer}
        if auto_grant:
            api.post(f"/api/v1/approval-grants/{grant['id']}/grant", {}, label=f"{label}:grant")
            return {**record, "action": "granted"}
        return {**record, "action": "left"}
    except ApiError as exc:
        return {**record, "action": "error", "status": exc.status, "body": exc.body[:300]}
