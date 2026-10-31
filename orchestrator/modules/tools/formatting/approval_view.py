"""PRD-268 §6.11.4, K10 — ``ApprovalView@1``: the exact action, its target, its preview and the rule that asked.

```ts
type ApprovalView = {
  grant_id: string
  asked_by: { kind: 'auto' | 'agent'; name: string }
  action: { verb: 'send' | 'pay' | 'sign' | 'delete' | 'publish' | 'run' | string; summary: string }
  target?: { label: string; detail?: string }
  preview?: { kind: 'text' | 'markdown' | 'diff'; body: string }
  rule: { id: string; sentence: string }
  answers: Array<'refuse' | 'allow_once'>          // never 'always_allow' (K10, §12-Q17)
  expires_at?: string
}
```

Built from a card payload (the chat's ``tool_approval`` and ``mission_approval``, by
``carries_the_approval_view`` on ``ToolResultFormatter.format_for_frontend``) or from a grant row
(``services/approval_views.py`` behind ``GET /api/v1/approval-grants/{grant_id}/view``). The
target and the preview are read from the call's arguments through the card's own digest
(``card_digest.params_digest``: the ``_`` plumbing dropped, each value cut at ``DIGEST_CHARS``),
never from the raw params, so the view shows nothing the card does not. The rule's sentence is
``modules.policy.rule_sentences``'s, imported where it is read: that table reads the approval
policies from ``core.services``, which the formatter's own import must not load. Always allow is
not built: ``answers`` cannot hold it.

A Mission plan has no grant row: its view's ``grant_id`` is the Mission's id. A card's view is
in English (the formatter knows no language); the route's is in the caller's.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from modules.policy.policy_document import RISK_DESTRUCTIVE, RISK_EXTERNAL, RISK_PUBLISH
from modules.proactive.locale import DEFAULT_LOCALE
from modules.tools.formatting.card_digest import QUESTION, params_digest, shown_on_the_card

logger = logging.getLogger(__name__)

__all__ = [
    "ANSWERS", "ApprovalView", "ViewParts", "build_view", "carries_the_approval_view", "mission_card_view",
    "ViewTarget", "one_line", "shown", "target_of", "tool_card_view", "verb_for",
]

VIEW = "approval_view"
TOOL_CARD = "tool_approval"
MISSION_CARD = "mission_approval"
# What a person may answer today (K10): Always allow is gated on §12-Q17.
ANSWERS = ("refuse", "allow_once")
AUTO = "auto"
AGENT = "agent"
AUTO_NAME = "Auto"
RUN = "run"
# The verb a risk class reads as; any other class runs. Strict on purpose: voice never allows these.
VERB_BY_RISK = {RISK_DESTRUCTIVE: "delete", RISK_PUBLISH: "publish", RISK_EXTERNAL: "send"}
# The digest's keys that name what a call acts on, first match wins.
TARGET_KEYS = ("to", "recipient", "recipient_email", "recipients", "email", "channel", "chat_id", "agent_name",
               "agent_id", "document_id", "task_id", "mission_id", "playbook_id", "name", "title", "path", "url")
_LINE = "{key}: {value}"
_STEP = "{n}. {title}"

Answer = Literal["refuse", "allow_once"]


class _Shape(BaseModel):
    """Held still, unknown fields ignored, an id given as a number read as its string."""

    model_config = ConfigDict(extra="ignore", frozen=True, coerce_numbers_to_str=True)


class AskedBy(_Shape):
    kind: Literal["auto", "agent"]
    name: str


class ViewAction(_Shape):
    verb: str
    summary: str


class ViewTarget(_Shape):
    label: str
    detail: Optional[str] = None


class ViewPreview(_Shape):
    kind: Literal["text", "markdown", "diff"]
    body: str


class ViewRule(_Shape):
    id: str
    sentence: str


class ApprovalView(_Shape):
    """``ApprovalView@1`` (§6.11.4)."""

    grant_id: str
    asked_by: AskedBy
    action: ViewAction
    target: Optional[ViewTarget] = None
    preview: Optional[ViewPreview] = None
    rule: ViewRule
    answers: Tuple[Answer, ...] = ANSWERS
    expires_at: Optional[str] = None


class ViewParts(NamedTuple):
    """What a view is made of, already read from a card or a grant. ``digest`` is the card's digest."""

    grant_id: Any
    agent_name: Optional[str]
    verb: str
    summary: str
    digest: Dict[str, Any]
    rule_id: str
    target: Optional[ViewTarget] = None
    preview: Optional[ViewPreview] = None
    expires_at: Optional[str] = None
    answers: Tuple[str, ...] = ANSWERS


def build_view(parts: ViewParts, language: str = DEFAULT_LOCALE) -> Dict[str, Any]:
    """The view as JSON: the target and the preview from the digest unless ``parts`` gives them. A rule
    id the table doesn't hold reads as ``FALLBACK``."""
    from modules.policy.rule_sentences import known_rule, rule_sentence

    rule_id = known_rule(parts.rule_id)
    view = ApprovalView(
        grant_id=parts.grant_id,
        asked_by=_asked_by(parts.agent_name),
        action=ViewAction(verb=parts.verb, summary=shown(parts.summary)),
        target=parts.target or target_of(parts.digest),
        preview=parts.preview or _preview_of(parts.digest),
        rule=ViewRule(id=rule_id, sentence=rule_sentence(rule_id, language)),
        answers=parts.answers,
        expires_at=parts.expires_at,
    )
    return view.model_dump(mode="json", exclude_none=True)


def one_line(value: Any) -> str:
    """``value`` on one line: a newline in a model-written value never forges a line of its own."""
    return " ".join(str(value if value is not None else "").split())


def shown(value: Any) -> str:
    """``value`` as the view shows it: on one line, and no longer than the card's digest shows it."""
    return shown_on_the_card(one_line(value))


def verb_for(risk_class: Any) -> str:
    """The action's verb for ``risk_class``: delete, publish or send, else run."""
    return VERB_BY_RISK.get(str(risk_class), RUN)


def _asked_by(agent_name: Optional[str]) -> AskedBy:
    """Auto, unless an agent other than Auto asked."""
    name = one_line(agent_name)
    if not name or name == AUTO_NAME:
        return AskedBy(kind=AUTO, name=AUTO_NAME)
    return AskedBy(kind=AGENT, name=shown(name))


def target_of(digest: Dict[str, Any]) -> Optional[ViewTarget]:
    """What the call acts on: the first digest entry that names a target."""
    key = next((k for k in TARGET_KEYS if digest.get(k) not in (None, "")), None)
    return ViewTarget(label=shown(digest[key]), detail=key) if key else None


def _preview_of(digest: Dict[str, Any]) -> Optional[ViewPreview]:
    """The call exactly as the card shows it, one ``key: value`` line per argument."""
    lines = [_LINE.format(key=shown(key), value=shown(value)) for key, value in digest.items()]
    return ViewPreview(kind="text", body="\n".join(lines)) if lines else None


def _described(action: str) -> str:
    """The action's registered description, or nothing for an action the registry doesn't hold."""
    from modules.tools.discovery.action_registry import get_action_registry

    return getattr(get_action_registry().get(action), "description", "") or ""


def tool_card_view(card: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """The view of a confirmation-gated call's card (its digest already numbered by the card)."""
    from modules.policy.rule_sentences import confirm_rule

    params = result.get("params") if isinstance(result.get("params"), dict) else {}
    action = str(card.get("action") or "")
    risk = card.get("risk_class")
    summary = card.get(QUESTION) or _described(action) or card.get("message") or action
    digest = card.get("params") if isinstance(card.get("params"), dict) else params_digest(params)
    return build_view(ViewParts(grant_id=card.get("grant_id"), agent_name=params.get("_agent_name"),
                                verb=verb_for(risk), summary=str(summary), digest=digest,
                                rule_id=confirm_rule(risk)))


def mission_card_view(card: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """The view of a Mission plan's card: the goal, its number, its steps and the policy that held it."""
    from modules.policy.rule_sentences import FALLBACK, approval_policy_rule

    created = result if result.get("mission_id") else (result.get("raw_result") or {})
    steps = [step.get("title") for step in card.get("tasks") or [] if isinstance(step, dict)]
    body = "\n".join(_STEP.format(n=n, title=shown(title))
                     for n, title in enumerate(steps, start=1) if title)
    number = created.get("number")
    policy = created.get("approval_policy")
    return build_view(ViewParts(
        grant_id=card.get("mission_id"), agent_name=None, verb=RUN, summary=str(card.get("goal") or ""),
        digest={}, rule_id=approval_policy_rule(policy) if policy else FALLBACK,
        target=ViewTarget(label=shown(number)) if number else None,
        preview=ViewPreview(kind="text", body=body) if body else None,
    ))


def _with_view(data: Dict[str, Any], key: str, build: Callable[..., Dict[str, Any]],
               result: Dict[str, Any]) -> Dict[str, Any]:
    """``data`` with its ``key`` card carrying the view (a new dict); today's card when it can't be built."""
    card = data.get(key)
    if not isinstance(card, dict):
        return data
    try:
        return {**data, key: {**card, VIEW: build(card, result)}}
    except Exception:
        logger.exception("[approval_view] could not build the %s view; the card goes without it", key)
        return data


def carries_the_approval_view(format_for_frontend: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap ToolResultFormatter.format_for_frontend: the ``tool_approval`` and ``mission_approval``
    cards carry ``approval_view`` (§6.11.4), read from the card the inner formatter made."""
    @functools.wraps(format_for_frontend)
    def wrapped(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        data = format_for_frontend(result, tool_name)
        if not isinstance(data, dict) or not isinstance(result, dict):
            return data
        data = _with_view(data, TOOL_CARD, tool_card_view, result)
        return _with_view(data, MISSION_CARD, mission_card_view, result)
    return wrapped
