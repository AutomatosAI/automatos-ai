"""
PRD-251 S3.5 (D14b) — one way out: agents cannot post directly when Socials is on
=================================================================================

With Socials on for a workspace, a post leaves Automatos one way: an agent drafts
it (``platform_create_social_post``), a person approves it in the Socials tab and
the platform publishes it (Wave 3). An agent's direct Composio call to a social
PUBLISH action is refused before any Composio call, with::

    Socials is on for this workspace: draft the post with platform_create_social_post;
    a person approves it in the Socials tab and the platform publishes it.

The publish actions are DATA, never a code constant: the ``socials.post_actions``
system setting, a JSON list of action slugs, seeded by the ``prd251_wave1``
migration with the actions that publish a post on Instagram, X and LinkedIn, and
edited by the super-admin in Settings → System Settings. Slugs match
case-insensitively. Reads (``LINKEDIN_GET_MY_INFO``) and uploads
(``TWITTER_UPLOAD_MEDIA``, an Instagram media container) are not on it and run as
before. A workspace with Socials off (either switch, D1) runs every action exactly
as before.

Where it runs: on the agent paths, right after the Wave 0 deny list
(``core/composio/deny_list.py``), which is checked first and always wins:

* ``ComposioToolExecutor.execute``, on the name asked for and again on the name
  validation resolved it to, before the entity lookup, file uploads, the LinkedIn
  image workaround (which never passes through the Composio client) and the SDK.
  Every agent tool call, a Playbook step's spine call and an approved grant reach
  Composio here. The platform's own callers (the Socials voice and footage
  recipes, web search, cloud sync) pass through it too; none of them calls a
  posting action;
* a Playbook step's Composio branch (``api/recipe_executor._execute_step``),
  before the dedup cache, the LinkedIn image workaround and the spine.

The way through: the platform's own publisher (Wave 3,
``modules/socials/publisher.py``) publishes a post a person approved, so it will
pass ``way_through=PLATFORM_PUBLISHER`` and the gate lets it by. Nothing passes it
yet. The Wave 0 deny list applies to it all the same.

Fails closed for a Socials-on workspace (the Wave 0 RVW-1 lesson: a gate that
can't decide must deny):

* a stored value that is not a JSON list of strings → every Composio call these
  paths make in a Socials-on workspace is refused, with the reason, until a
  super-admin fixes it; so is every call while the list cannot be read and
  nothing has been read in this process (a failed read keeps the last list read,
  logged at WARNING);
* a Socials switch that cannot be read → a listed action is refused;
* no row, or an empty list → nothing is listed (a stack without the seed), as
  with the deny list.

The list is cached per process for ``config.SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS``
(30 s by default), so an edit applies within one TTL, with no restart or deploy. A
warm cache answers from memory; a stale one answers from memory too and starts ONE
background refresh; a cold one is read in a worker thread, never on the event loop
(F105). The two switches are read, in a worker thread, only for a listed action or
while the list cannot be decided, so a switch flipped off applies on the next call.
The strict reads only: never ``get_system_setting`` or ``socials_master_enabled``,
which turn "could not read" into a default or into Socials OFF — and OFF lets an
agent post.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, FrozenSet, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

KEY_POST_ACTIONS = "post_actions"  # in SettingCategory.SOCIALS
DRAFT_TOOL = "platform_create_social_post"
ERROR_TYPE_POST_GATE = "socials_post_gate"

_DRAFT = (
    "draft the post with platform_create_social_post; a person approves it in the "
    "Socials tab and the platform publishes it."
)
SOCIALS_ON_REFUSAL = "Socials is on for this workspace: " + _DRAFT
SWITCH_UNREAD_REFUSAL = (
    "Automatos could not read whether Socials is on for this workspace, so {slug} did "
    "not run: " + _DRAFT
)
_LIST_UNREADABLE = (
    "its list of posting actions (system setting socials.post_actions) is not a JSON "
    "list of action slugs, so no Composio action runs here until a super-admin fixes "
    "it in Settings → System Settings"
)
_LIST_READ_FAILED = (
    "its list of posting actions (system setting socials.post_actions) could not be "
    "read, so no Composio action runs here until it can be"
)
LIST_REFUSAL = "Socials is on for this workspace and {problem}. To post, " + _DRAFT
LIST_AND_SWITCH_REFUSAL = (
    "Automatos could not read whether Socials is on for this workspace, and {problem}. "
    "To post, " + _DRAFT
)


class _WayThrough:
    """The gate's way through, held by one caller only."""

    __slots__ = ("holder",)

    def __init__(self, holder: str):
        self.holder = holder

    def __repr__(self) -> str:
        return f"<Socials post gate way through: {self.holder}>"


# Only the platform's own publisher passes it (Wave 3), for a post a person
# approved. Anything else — True, a string, another object — is no way through.
PLATFORM_PUBLISHER = _WayThrough("the platform publisher (PRD-251 Wave 3)")


def parse_post_actions(raw: Optional[str]) -> FrozenSet[str]:
    """The setting's value → the listed slugs, upper-cased.

    No value (``None``, blank, or not text — the column only ever holds text) →
    nothing listed. Raises ``ValueError`` unless the value is a JSON list of
    strings.
    """
    if not isinstance(raw, str) or not raw.strip():
        return frozenset()
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(slug, str) for slug in value):
        raise ValueError("socials.post_actions must be a JSON list of action slugs")
    return frozenset(slug.strip().upper() for slug in value if slug.strip())


def _read_post_actions() -> Optional[str]:
    """The setting's stored value, ``None`` when there is no row. Raises when the
    read cannot complete."""
    from core.llm.manager import read_system_setting
    from core.models.system_settings import SettingCategory

    return read_system_setting(SettingCategory.SOCIALS.value, KEY_POST_ACTIONS)


# The last COMPLETED read: (outcome, expires_at on the monotonic clock). The outcome
# is the listed slugs, or _UNREADABLE when the stored value is not a JSON list. A
# failed read is never cached: with nothing cached, the next call reads again.
_UNREADABLE = object()
_MISSING = object()
_cache: Optional[Tuple[object, float]] = None
_generation = 0  # bumped by reset_cache(), so a refresh that started before it never lands after it
_refresh_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def _ttl_seconds() -> float:
    from config import config

    return float(config.SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS)


def reset_cache() -> None:
    """Forget the cached list, so the next check reads it again (tests)."""
    global _cache, _generation
    _generation += 1
    _cache = None


def _outcome_of(raw: Optional[str]) -> object:
    try:
        return parse_post_actions(raw)
    except ValueError:
        logger.error(
            "[SocialsPostGate] socials.post_actions is unreadable (%r); Composio calls in "
            "Socials-on workspaces are refused until it is fixed",
            raw,
        )
        return _UNREADABLE


def _refresh() -> Optional[object]:
    """Read the list once (single-flight) and cache the outcome.

    Returns the outcome. When the read cannot complete, the cached outcome is kept
    (WARNING) and returned; with nothing cached, returns ``None`` (ERROR). Blocks
    the calling thread, so it runs off the event loop.
    """
    global _cache
    with _refresh_lock:
        generation = _generation
        cached = _cache
        now = _now()
        if cached is not None and now < cached[1]:
            return cached[0]  # another thread refreshed while this one waited
        try:
            raw = _read_post_actions()
        except Exception:  # noqa: BLE001 — any read that did not complete
            if cached is not None:
                logger.warning(
                    "[SocialsPostGate] socials.post_actions could not be refreshed; keeping the cached list",
                    exc_info=True,
                )
                if generation == _generation:
                    _cache = (cached[0], now + _ttl_seconds())  # retry after one TTL, not on every call
                return cached[0]
            logger.error(
                "[SocialsPostGate] socials.post_actions could not be read and nothing is cached; "
                "Composio calls in Socials-on workspaces are refused until it can be",
                exc_info=True,
            )
            return None
        outcome = _outcome_of(raw)
        if generation == _generation:
            _cache = (outcome, now + _ttl_seconds())
        return outcome


def _start_background_refresh() -> None:
    """Refresh in a daemon thread, at most one at a time; never blocks the caller."""
    if _refresh_lock.locked():
        return
    threading.Thread(target=_refresh, name="socials-post-gate-refresh", daemon=True).start()


def _cached_outcome() -> object:
    """The cached outcome without a database read. Fresh → it. Stale → it, with a
    background refresh started. Cold → ``_MISSING``."""
    cached = _cache
    if cached is None:
        return _MISSING
    if _now() >= cached[1]:
        _start_background_refresh()
        cached = _cache or cached  # an inline refresh (tests) has already landed
    return cached[0]


async def _post_actions() -> Optional[object]:
    """The list's outcome: warm from memory, cold read in a worker thread."""
    outcome = _cached_outcome()
    if outcome is _MISSING:
        outcome = await asyncio.to_thread(_refresh)
    return outcome


def _socials_on(workspace_id: Any) -> Optional[bool]:
    """Both Socials switches for the workspace (D1): True when both are on, False
    when either is off (or there is no such workspace), None when a switch could
    not be read. Reads the database, so it runs in a worker thread."""
    try:
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace
        from modules.socials.settings import parse_workspace_socials, socials_master_switch

        if not socials_master_switch():
            return False
        workspace_uuid = workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))
        db = SessionLocal()
        try:
            row = db.query(Workspace.settings).filter(Workspace.id == workspace_uuid).first()
        finally:
            db.close()
    except Exception:  # noqa: BLE001 — a switch that cannot be read fails closed
        logger.error(
            "[SocialsPostGate] could not read whether Socials is on for workspace %s; refusing",
            workspace_id,
            exc_info=True,
        )
        return None
    return row is not None and parse_workspace_socials(row[0]).enabled


def _refusal(slug: str, outcome: Optional[object], socials_on: Optional[bool]) -> str:
    if isinstance(outcome, frozenset):  # the action is listed
        return SOCIALS_ON_REFUSAL if socials_on else SWITCH_UNREAD_REFUSAL.format(slug=slug)
    problem = _LIST_UNREADABLE if outcome is _UNREADABLE else _LIST_READ_FAILED
    return (LIST_REFUSAL if socials_on else LIST_AND_SWITCH_REFUSAL).format(problem=problem)


async def post_action_refusal(action: Any, workspace_id: Any, *, way_through: Any = None) -> Optional[str]:
    """Why an agent's Composio call to ``action`` may not run in ``workspace_id``
    (D14b), else ``None``. Call it after the Wave 0 deny list. Never raises.

    Only ``way_through=PLATFORM_PUBLISHER`` passes a listed action in a Socials-on
    workspace. The switches are read only for a listed action, or while the list
    cannot be decided.
    """
    if way_through is PLATFORM_PUBLISHER:
        return None
    slug = str(action or "").strip().upper()
    outcome = await _post_actions()
    if isinstance(outcome, frozenset) and slug not in outcome:
        return None
    socials_on = await asyncio.to_thread(_socials_on, workspace_id)
    if socials_on is False:
        return None
    refusal = _refusal(slug, outcome, socials_on)
    logger.warning("[SocialsPostGate] refused %s in workspace %s: %s", slug, workspace_id, refusal)
    return refusal


def refused_result(refusal: str) -> Dict[str, Any]:
    """The standard failed-execution envelope for a call the gate refused: the
    reason, and the tool an agent drafts the post with instead."""
    return {
        "success": False,
        "data": None,
        "error": refusal,
        "error_type": ERROR_TYPE_POST_GATE,
        "use_tool": DRAFT_TOOL,
    }
