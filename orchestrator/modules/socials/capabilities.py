"""PRD-251 D16: the media capability registry — what a workspace's connected
Composio tools may do for Socials.

Footage, stills and premium voice come from the workspace's own Composio
toolkits (D11, D12, D15): Automatos holds no provider key and writes no provider
client. This registry decides which of those toolkits' actions Socials may call,
and what for. An action is offered only when all four hold:

1. its toolkit is connected in the workspace (``EntityManager.get_connected_apps``);
2. the allowlist lists it for that toolkit. The allowlist is DATA, never a code
   constant: the ``socials.media_actions`` system setting, a JSON object
   ``{toolkit: {capability: [action slug, ...]}}``, seeded by the
   ``prd251_wave1`` migration for fal.ai, Kie.ai, Higgsfield MCP, Fish Audio and
   ElevenLabs. An unknown toolkit offers nothing until a super-admin adds it
   there. Toolkits and slugs match case-insensitively;
3. the toolkit's cached action schemas (``composio_actions_cache``) hold it. An
   allowlisted slug the cache does not hold is not offered, so a toolkit whose
   actions were never synced shows as unavailable. An offered action carries its
   cached input schema, which the recipes build their calls from (US-111, US-114);
4. the Wave 0 deny list (``core/composio/deny_list.py``) lets it run. The deny
   list always wins: a denied slug is never offered, even when allowlisted, and a
   deny list that cannot be read offers nothing. ``ComposioToolExecutor`` checks
   it again when the action is called.

Capabilities: ``generate_video`` and ``generate_image`` (footage and stills),
``tts`` (speech, one call per script line), ``voices`` (the voice catalogue a tts
recipe picks from), ``estimate`` (a price before any spend, D13), ``status`` (a
job's state or result: always submit, then poll), ``upload`` (a file the
provider reads) and ``balance`` (credits read before and after a job, D13). One
action can serve several: fal's queue submit runs video and image models alike.

Fails closed, like every guard on real money: no row offers nothing, and so does
a value that is not the shape above or a read that cannot complete. Those two say
why (``MediaCapabilities.problem``) and are logged at ERROR. The read is strict
(``read_system_setting``), never ``get_system_setting``, whose catch-all would
turn "could not read" into a default.

``media_capabilities`` reads the database synchronously, so code on the event
loop runs it in a worker thread. ``get_connected_apps`` may commit the session
(it upgrades a pending connection whose OAuth has completed): call the registry
before staging writes of your own.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.composio.deny_list import composio_action_denial
from core.composio.entity_manager import EntityManager
from core.llm.manager import read_system_setting
from core.models.composio_cache import ComposioActionCache
from modules.socials.settings import SOCIALS_SETTINGS_CATEGORY

logger = logging.getLogger(__name__)

KEY_MEDIA_ACTIONS = "media_actions"  # in the socials settings category

# The capabilities: the vocabulary the recipes and the composer ask for. Which
# action serves which capability is the allowlist's business (data).
GENERATE_VIDEO = "generate_video"
GENERATE_IMAGE = "generate_image"
TTS = "tts"
VOICES = "voices"
ESTIMATE = "estimate"
STATUS = "status"
UPLOAD = "upload"
BALANCE = "balance"
CAPABILITIES = (GENERATE_VIDEO, GENERATE_IMAGE, TTS, VOICES, ESTIMATE, STATUS, UPLOAD, BALANCE)

_SETTING = f"{SOCIALS_SETTINGS_CATEGORY}.{KEY_MEDIA_ACTIONS}"
READ_FAILED_PROBLEM = (
    f"The Socials media allowlist (system setting {_SETTING}) could not be read, so no "
    "connected media tool is offered until it can be."
)
UNUSABLE_PROBLEM = (
    f"The Socials media allowlist (system setting {_SETTING}) is not usable ({{why}}), so no "
    "connected media tool is offered until a super-admin fixes it."
)

Allowlist = Dict[str, Dict[str, FrozenSet[str]]]  # toolkit → SLUG → capabilities


def parse_media_actions(raw: Optional[str]) -> Allowlist:
    """The setting's value → ``{toolkit: {SLUG: capabilities}}``, toolkits in
    lower case and slugs in upper case.

    No value (``None`` or blank) → ``{}``: nothing is offered. Raises
    ``ValueError`` naming the problem unless the value is a JSON object of
    toolkit → ``{capability: [slug, ...]}`` whose capabilities are known and
    whose slugs are non-blank strings.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc.msg}") from None
    if not isinstance(value, dict):
        raise ValueError("not a JSON object of toolkits")
    allowlist: Allowlist = {}
    for toolkit, listed in value.items():
        name = toolkit.strip().lower()
        if not name:
            raise ValueError("a toolkit name is blank")
        if not isinstance(listed, dict):
            raise ValueError(f"{toolkit!r} is not an object of capabilities")
        actions = allowlist.setdefault(name, {})
        for capability, slugs in listed.items():
            if capability not in CAPABILITIES:
                raise ValueError(f"{toolkit!r} lists the unknown capability {capability!r}")
            if not isinstance(slugs, list) or not all(isinstance(s, str) and s.strip() for s in slugs):
                raise ValueError(f"{toolkit!r} {capability} is not a list of action slugs")
            for slug in slugs:
                key = slug.strip().upper()
                actions[key] = actions.get(key, frozenset()) | {capability}
    return allowlist


def read_media_actions() -> Tuple[Allowlist, Optional[str]]:
    """The allowlist, and why it cannot be used when it cannot. Never raises: a
    read that cannot complete, or a value that is not the shape, gives
    ``({}, problem)`` and is logged at ERROR. No row gives ``({}, None)``."""
    try:
        raw = read_system_setting(SOCIALS_SETTINGS_CATEGORY, KEY_MEDIA_ACTIONS)
    except Exception:  # noqa: BLE001 — any read that did not complete fails closed
        logger.error(
            "[SocialsMedia] system setting %s could not be read; no media tool is offered until it can be",
            _SETTING,
            exc_info=True,
        )
        return {}, READ_FAILED_PROBLEM
    try:
        return parse_media_actions(raw), None
    except ValueError as exc:
        logger.error("[SocialsMedia] system setting %s is not usable (%s): %r", _SETTING, exc, raw)
        return {}, UNUSABLE_PROBLEM.format(why=exc)


def _known(capability: str) -> str:
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown media capability {capability!r}; expected one of {CAPABILITIES}")
    return capability


def _toolkit(name: Any) -> str:
    return str(name or "").strip().lower()


@dataclass(frozen=True)
class OfferedAction:
    """One action Socials may call, with the cached schema its recipe builds the call from."""

    toolkit: str  # the Composio toolkit, lower case
    slug: str  # the action slug, upper case
    capabilities: FrozenSet[str]
    display_name: str
    parameters: Mapping[str, Any]  # the cached input schema; {} until the sync fills it


@dataclass(frozen=True)
class MediaCapabilities:
    """What one workspace's connected Composio tools may do for Socials."""

    # Connected toolkit → SLUG → the action; only toolkits that offer something.
    offered: Mapping[str, Mapping[str, OfferedAction]]
    # Every toolkit the allowlist knows → its capabilities, connected or not.
    allowlisted: Mapping[str, FrozenSet[str]]
    # Every toolkit connected in the workspace, lower case.
    connected: FrozenSet[str]
    # SLUG → the deny list's reason, for allowlisted actions it withheld.
    withheld: Mapping[str, str]
    # Why nothing is offered, when the allowlist itself cannot be used.
    problem: Optional[str] = None

    def toolkits(self, capability: str) -> Tuple[str, ...]:
        """The connected toolkits that offer ``capability``, in name order."""
        _known(capability)
        return tuple(
            toolkit
            for toolkit, actions in self.offered.items()
            if any(capability in action.capabilities for action in actions.values())
        )

    def actions(self, toolkit: str, capability: Optional[str] = None) -> Tuple[OfferedAction, ...]:
        """The toolkit's offered actions, all of them or those serving ``capability``."""
        if capability is not None:
            _known(capability)
        offered = self.offered.get(_toolkit(toolkit), {})
        return tuple(
            action for action in offered.values() if capability is None or capability in action.capabilities
        )

    def action(self, toolkit: str, slug: str) -> Optional[OfferedAction]:
        """The offered action, or ``None`` when Socials may not call it."""
        return self.offered.get(_toolkit(toolkit), {}).get(str(slug or "").strip().upper())

    def connectable(self, capability: str) -> Tuple[str, ...]:
        """The allowlisted toolkits that would bring ``capability`` once connected
        (the composer's links to the Composio connect flow)."""
        _known(capability)
        return tuple(
            toolkit
            for toolkit, capabilities in self.allowlisted.items()
            if capability in capabilities and toolkit not in self.connected
        )


def _workspace(workspace_id: Any) -> UUID:
    return workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))


def _cached_actions(db: Session, wanted: Allowlist) -> Dict[Tuple[str, str], Any]:
    """``{(toolkit, SLUG): cached row}`` for the wanted actions the cache holds."""
    apps = sorted(toolkit.upper() for toolkit, actions in wanted.items() if actions)
    slugs = sorted({slug for actions in wanted.values() for slug in actions})
    if not apps:
        return {}
    rows = (
        db.query(
            ComposioActionCache.app_name,
            ComposioActionCache.action_name,
            ComposioActionCache.display_name,
            ComposioActionCache.parameters,
        )
        # Both sync writers store app_name upper case; the exact match keeps its index.
        .filter(ComposioActionCache.app_name.in_(apps))
        .filter(func.upper(ComposioActionCache.action_name).in_(slugs))
        .all()
    )
    return {(row.app_name.lower(), row.action_name.upper()): row for row in rows}


def media_capabilities(db: Session, workspace_id: Any) -> MediaCapabilities:
    """What the workspace's connected Composio tools may do for Socials (D16)."""
    allowlist, problem = read_media_actions()
    connected = frozenset(
        _toolkit(app) for app in EntityManager(db).get_connected_apps(_workspace(workspace_id))
    )
    wanted = {toolkit: actions for toolkit, actions in allowlist.items() if toolkit in connected}
    cached = _cached_actions(db, wanted)

    offered: Dict[str, Dict[str, OfferedAction]] = {}
    withheld: Dict[str, str] = {}
    for toolkit in sorted(wanted):
        for slug, capabilities in sorted(wanted[toolkit].items()):
            row = cached.get((toolkit, slug))
            if row is None:
                continue  # not in the toolkit's cached schemas: unavailable
            denial = composio_action_denial(slug)
            if denial is not None:
                withheld[slug] = denial  # the deny list always wins
                continue
            offered.setdefault(toolkit, {})[slug] = OfferedAction(
                toolkit=toolkit,
                slug=slug,
                capabilities=capabilities,
                display_name=row.display_name or slug,
                parameters=MappingProxyType(dict(row.parameters or {})),
            )

    return MediaCapabilities(
        offered=MappingProxyType({toolkit: MappingProxyType(actions) for toolkit, actions in offered.items()}),
        allowlisted=MappingProxyType(
            {
                toolkit: frozenset(capability for capabilities in actions.values() for capability in capabilities)
                for toolkit, actions in sorted(allowlist.items())
            }
        ),
        connected=connected,
        withheld=MappingProxyType(withheld),
        problem=problem,
    )
