"""PRD-251 D16: the capability registry — what a workspace's connected Composio
tools may do for Socials: its media tools, and (D8, US-203) its social channels.

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

Channels (D8, S3.2): the seeded adapters are data (``channel_adapters.py``: LinkedIn,
X, Instagram, TikTok, YouTube), checked when this module loads, so a malformed entry
fails the import and names itself. A post kind is an ordered action sequence whose
steps are classed ``read``, ``upload``, ``status`` or ``publish``. ``social_channels``
lists each connected channel, and each of its kinds is unavailable when the deny list
refuses an action it must run (in the deny list's words), when ``composio_actions_cache``
lacks one (named: the cache is filled at start-up and by ``POST /api/tools/sync``, not
on connect; the slug's presence is enough, since the bulk sync often leaves
``parameters`` empty), or when a step that must run takes only a link and
``media_urls.media_public_url_available()`` is false (D9). A link step marks its kind
``needs_public_storage``; an optional step that cannot run is skipped at publish
instead, and never makes its kind unavailable. A connected
toolkit outside the data is offered by the generic adapter (``GENERIC_ADAPTER``): its
first cached action, by slug, whose name creates a post and whose non-empty schema has
a text field and a media field, never a denied one; it is an "unverified channel" until
one of its targets has published in the workspace (``social_post_targets``).

One way out (D14): the post gate (``core/composio/post_gate.py``) refuses an agent's
call to any action this registry classes as ``publish`` for a channel connected in the
workspace: ``publish_candidate`` answers from the data, ``channel_publish_action`` reads.

``media_capabilities``, ``social_channels`` and ``channel_publish_action`` read the
database synchronously, so code on the event loop runs them in a worker thread.
``get_connected_apps`` may commit the session (it upgrades a pending connection
whose OAuth has completed): call the registry before staging writes of your own.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from core.composio.deny_list import composio_action_denial
from core.composio.entity_manager import EntityManager
from core.llm.manager import read_system_setting
from core.models.composio_cache import ComposioActionCache
from core.models.socials import SOCIAL_TARGET_POST_KINDS, SocialPost, SocialPostTarget
from modules.socials.channel_adapters import CHANNEL_ADAPTERS, GENERIC_ADAPTER
from modules.socials.settings import SOCIALS_SETTINGS_CATEGORY
from modules.socials.step_results import ID, parse_permalink, parse_returns, parse_until

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


def _connected_toolkits(db: Session, workspace_id: Any) -> FrozenSet[str]:
    """Every toolkit connected in the workspace, lower case (may commit: see above)."""
    return frozenset(_toolkit(app) for app in EntityManager(db).get_connected_apps(_workspace(workspace_id)))


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
    connected = _connected_toolkits(db, workspace_id)
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


# ---------------------------------------------------------------------------
# D8 (US-203): the channels — what a connected social toolkit can post, and how
# ---------------------------------------------------------------------------

READ = "read"  # a channel step's class; UPLOAD and STATUS are the media vocabulary's own
PUBLISH = "publish"
STEP_CLASSES = (READ, UPLOAD, STATUS, PUBLISH)
SEEDED, GENERIC = "seeded", "generic"  # what publish_candidate answers
TEXT_KIND, IMAGE_KIND, VIDEO_KIND = "text", "image", "video"  # the generic adapter's post kinds
PUBLISHED_TARGET = "published"  # social_post_targets.status once a target has published
UNVERIFIED_CHANNEL = "unverified channel"
COPY_SOURCE, MEDIA_SOURCE, MEDIA_LIST_SOURCE = "$copy", "$media", "$media[]"
GENERIC_STEP_ID = "post"

MISSING_ACTION = (
    "Missing action {slug}: Automatos's list of Composio actions does not hold it yet. "
    "The list is filled when Automatos starts and by a tool sync (Settings → Tools → Sync, "
    "or POST /api/tools/sync), not when an app is connected."
)
NEEDS_PUBLIC_LINK = (
    "{needs}: {slug} takes only a link the platform fetches, and this instance's object "
    "storage is private. Set SOCIALS_PUBLIC_MEDIA_BUCKET to a bucket the platform can reach."
)

_STEP_KEYS = frozenset(
    {"id", "action", "class", "params", "files", "urls", "optional", "returns", "until", "permalink"}
)
_ADAPTER_KEYS = frozenset({"label", "setup_note", "kinds", "never_offered"})
_STEP_REF = "$steps."
_SOURCE = re.compile(
    r"\$(?:copy|title|thumbnail|idempotency_key|media(?:\[\]|\.content_type|\.bytes)?"
    r"|option\.[a-z][a-z0-9_]*|steps\.[a-z][a-z0-9_]*)"
)
_NOT_A_WORD = re.compile(r"[^A-Z0-9]+")
# The sources that name a media FILE: a param reading one takes a file or a link.
_FILE_SOURCES = ("$media", "$media[]", "$thumbnail")


@dataclass(frozen=True)
class ChannelStep:
    """One action of a post kind's sequence (see ``channel_adapters.py``)."""

    id: str
    action: str  # the Composio action slug, upper case
    step_class: str  # read, upload, status or publish
    params: Mapping[str, Any]  # the documented parameters → their sources
    files: Tuple[str, ...] = ()  # the params that take a file: the adapter's upload spec
    urls: Tuple[str, ...] = ()  # the params that take nothing but a link (D9)
    optional: bool = False  # skipped at publish when it cannot run
    returns: Mapping[str, str] = dataclass_field(default_factory=lambda: MappingProxyType({}))  # name → where the output holds it (step_results)
    until: Optional[Mapping[str, Any]] = None  # a status step's end condition
    permalink: Optional[str] = None  # a link template built from the returned id


@dataclass(frozen=True)
class ChannelAdapter:
    """A seeded channel (``channel_adapters.py``): its post kinds and their sequences."""

    toolkit: str  # the Composio toolkit, lower case
    label: str
    setup_note: Optional[str]
    kinds: Mapping[str, Tuple[ChannelStep, ...]]
    never_offered: FrozenSet[str]

    @property
    def publish_actions(self) -> FrozenSet[str]:
        """What the post gate refuses an agent (D14): every ``publish`` step's
        action, and the publish actions the channel never offers."""
        steps = (step for sequence in self.kinds.values() for step in sequence)
        return self.never_offered | {step.action for step in steps if step.step_class == PUBLISH}


@dataclass(frozen=True)
class GenericRules:
    """The generic adapter's rules (``GENERIC_ADAPTER``): what makes a cached action
    a post action, and its fields the copy and the media."""

    post_words: FrozenSet[str]
    skip_words: FrozenSet[str]
    text_fields: Tuple[str, ...]
    media_fields: Mapping[str, Tuple[str, ...]]  # field name → the post kinds it carries
    url_suffixes: Tuple[str, ...]
    file_marker: str
    returns: Mapping[str, str]  # what the post action returns (step_results)


@dataclass(frozen=True)
class ChannelKind:
    """One post kind of a channel, resolved for a workspace."""

    kind: str
    available: bool
    reason: Optional[str]
    needs_public_storage: bool
    steps: Tuple[ChannelStep, ...]  # the sequence that publishes it

    def to_dict(self) -> Dict[str, Any]:
        """As the channel list answers it: the sequence stays server-side."""
        return {
            "kind": self.kind,
            "available": self.available,
            "reason": self.reason,
            "needs_public_storage": self.needs_public_storage,
        }


@dataclass(frozen=True)
class SocialChannel:
    """A channel connected in the workspace, and what it can post."""

    toolkit: str
    label: str
    post_kinds: Tuple[ChannelKind, ...]
    verified: bool  # seeded, or generic and a target has published to it
    setup_note: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """As ``GET /api/socials/channels`` answers it."""
        return {
            "toolkit": self.toolkit,
            "label": self.label,
            "post_kinds": [kind.to_dict() for kind in self.post_kinds],
            "verified": self.verified,
            "setup_note": self.setup_note,
        }


# ---- the data, checked when the module loads ------------------------------


def _names(value: Any, where: str, prefix: str = "") -> Tuple[str, ...]:
    """An optional list of non-blank names; with ``prefix``, upper-case slugs bearing it."""
    names = [] if value is None else value
    if not isinstance(names, list) or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError(f"{where} is not a list of names")
    if prefix and any(name != name.strip().upper() or not name.startswith(prefix) for name in names):
        raise ValueError(f"{where}: {names} are not upper-case {prefix}* action slugs")
    return tuple(names)


def _check_source(value: Any, earlier: FrozenSet[str], where: str) -> None:
    """A list of sources, a literal, or ``$`` sources joined by ``|``: each one the
    grammar knows, and a step source naming an EARLIER step."""
    if isinstance(value, list):
        for item in value:
            _check_source(item, earlier, where)
        return
    for ref in value.split("|") if isinstance(value, str) and value.startswith("$") else ():
        if not _SOURCE.fullmatch(ref) or (ref.startswith(_STEP_REF) and ref[len(_STEP_REF):] not in earlier):
            raise ValueError(f"{where}: {ref!r} is not a source (a step source names an earlier step that returns an id)")


def _file_params(params: Mapping[str, Any]) -> FrozenSet[str]:
    """The params whose source reads a media file (``$media``, ``$media[]``, ``$thumbnail``)."""
    def reads_file(value: Any) -> bool:
        if isinstance(value, list):
            return any(reads_file(item) for item in value)
        return isinstance(value, str) and any(ref in _FILE_SOURCES for ref in value.split("|"))

    return frozenset(name for name, value in params.items() if reads_file(value))


def _parse_step(toolkit: str, where: str, raw: Any, seen: FrozenSet[str], referable: FrozenSet[str]) -> ChannelStep:
    """One step, checked. ``seen``: the earlier steps' ids; ``referable``: those of
    them that return an id, the only ones a ``$steps.<id>`` source may name."""
    if not isinstance(raw, Mapping) or set(raw) - _STEP_KEYS or not isinstance(raw.get("action"), str):
        raise ValueError(f"{where}: a step is an object of {sorted(_STEP_KEYS)} naming its action")
    (action,) = _names([raw["action"]], where, f"{toolkit.upper()}_")
    step_id, step_class, params = raw.get("id"), raw.get("class"), raw.get("params") or {}
    if not isinstance(step_id, str) or not step_id or step_id in seen or step_class not in STEP_CLASSES:
        raise ValueError(f"{where}.{action}: a step needs an id of its own and a class in {STEP_CLASSES}")
    if not isinstance(params, Mapping):
        raise ValueError(f"{where}.{action}: its params are not an object")
    for name, value in params.items():
        _check_source(value, referable, f"{where}.{action}.{name}")
    files = _names(raw.get("files"), f"{where}.{action} files")
    urls = _names(raw.get("urls"), f"{where}.{action} urls")
    if not set(files) | set(urls) <= set(params):
        raise ValueError(f"{where}.{action}: a file or link param is not one of its params")
    if not _file_params(params) <= set(files) | set(urls):
        raise ValueError(f"{where}.{action}: a param reading a media file must be one of its files or urls")
    returns = parse_returns(raw.get("returns"), f"{where}.{action}")
    return ChannelStep(
        step_id, action, step_class, MappingProxyType(dict(params)), files, urls, raw.get("optional") is True,
        returns=returns,
        until=parse_until(raw.get("until"), f"{where}.{action}", step_class, STATUS),
        permalink=parse_permalink(raw.get("permalink"), f"{where}.{action}", returns),
    )


def _parse_kind(toolkit: str, kind: Any, raw: Any) -> Tuple[ChannelStep, ...]:
    where = f"{toolkit}.{kind}"
    if kind not in SOCIAL_TARGET_POST_KINDS or not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: a post kind is one of {SOCIAL_TARGET_POST_KINDS}, with a list of steps")
    steps: Tuple[ChannelStep, ...] = ()
    for item in raw:
        seen = frozenset(step.id for step in steps)
        referable = frozenset(step.id for step in steps if ID in step.returns)
        steps += (_parse_step(toolkit, where, item, seen, referable),)
    if not any(step.step_class == PUBLISH for step in steps):
        raise ValueError(f"{where}: no step publishes the post")
    return steps


def _parse_adapter(toolkit: Any, raw: Any) -> ChannelAdapter:
    if not isinstance(toolkit, str) or not toolkit or toolkit != toolkit.strip().lower():
        raise ValueError(f"{toolkit!r} is not a lower-case toolkit name")
    if not isinstance(raw, Mapping) or set(raw) - _ADAPTER_KEYS:
        raise ValueError(f"{toolkit}: an adapter is an object of {sorted(_ADAPTER_KEYS)}")
    label, note, kinds = raw.get("label"), raw.get("setup_note"), raw.get("kinds")
    blank_note = note is not None and not (isinstance(note, str) and note.strip())
    if not isinstance(label, str) or not label.strip() or blank_note or not isinstance(kinds, Mapping) or not kinds:
        raise ValueError(f"{toolkit}: an adapter needs a label, post kinds, and a setup note that is None or text")
    return ChannelAdapter(
        toolkit=toolkit,
        label=label,
        setup_note=note,
        kinds=MappingProxyType({kind: _parse_kind(toolkit, kind, steps) for kind, steps in kinds.items()}),
        never_offered=frozenset(_names(raw.get("never_offered"), f"{toolkit}.never_offered", f"{toolkit.upper()}_")),
    )


def parse_channel_adapters(data: Any) -> Mapping[str, ChannelAdapter]:
    """The seeded adapters, checked: raises ``ValueError`` naming the first entry
    that is not the shape ``channel_adapters.py`` describes."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("the channel adapters are not an object of toolkits")
    return MappingProxyType({toolkit: _parse_adapter(toolkit, raw) for toolkit, raw in data.items()})


def parse_generic_adapter(raw: Any) -> GenericRules:
    """The generic adapter's rules, checked like the seeded data."""
    media = raw.get("media_fields") if isinstance(raw, Mapping) else None
    marker = raw.get("file_marker") if isinstance(raw, Mapping) else None
    if not isinstance(media, Mapping) or not media or not isinstance(marker, str) or not marker.strip():
        raise ValueError("the generic adapter needs media fields and a file marker")
    kinds = {str(name).lower(): _names(value, f"generic media field {name}") for name, value in media.items()}
    unknown = sorted({kind for carried in kinds.values() for kind in carried} - set(SOCIAL_TARGET_POST_KINDS))
    if unknown:
        raise ValueError(f"the generic adapter's media fields carry unknown post kinds {unknown}")
    return GenericRules(
        post_words=frozenset(word.upper() for word in _names(raw.get("post_words"), "generic post_words")),
        skip_words=frozenset(word.upper() for word in _names(raw.get("skip_words"), "generic skip_words")),
        text_fields=_names(raw.get("text_fields"), "generic text_fields"),
        media_fields=MappingProxyType(kinds),
        url_suffixes=_names(raw.get("url_suffixes"), "generic url_suffixes"),
        file_marker=marker,
        returns=parse_returns(raw.get("returns"), "generic"),
    )


SEEDED_CHANNELS: Mapping[str, ChannelAdapter] = parse_channel_adapters(CHANNEL_ADAPTERS)
GENERIC_RULES = parse_generic_adapter(GENERIC_ADAPTER)
_SEEDED_PUBLISH: Mapping[str, str] = MappingProxyType(  # a seeded publish action → its toolkit
    {slug: toolkit for toolkit, adapter in SEEDED_CHANNELS.items() for slug in adapter.publish_actions}
)


# ---- one way out (D14): what the post gate asks ----------------------------


def _creates_a_post(slug: str) -> bool:
    """The generic adapter's name rule: a post word and no skip word among the slug's words."""
    words = frozenset(word for word in _NOT_A_WORD.split(slug.upper()) if word)
    return bool(words & GENERIC_RULES.post_words) and not words & GENERIC_RULES.skip_words


def publish_candidate(slug: Any) -> Optional[str]:
    """Whether ``slug`` may be a channel's publish action, from the data alone (no
    read): ``SEEDED`` for a seeded channel's (those it never offers included),
    ``GENERIC`` for a name the generic adapter could offer, else ``None``."""
    name = str(slug or "").strip().upper()
    if name in _SEEDED_PUBLISH:
        return SEEDED
    seeded_prefix = any(name.startswith(f"{toolkit.upper()}_") for toolkit in SEEDED_CHANNELS)
    return GENERIC if name and not seeded_prefix and _creates_a_post(name) else None


def channel_publish_action(db: Session, workspace_id: Any, slug: Any) -> bool:
    """Whether this registry classes ``slug`` as the publish action of a channel
    connected in the workspace: a seeded channel's, or the post action of a
    connected toolkit the generic adapter offers (its cached schema qualifies)."""
    name = str(slug or "").strip().upper()
    candidate = publish_candidate(name)
    if candidate == SEEDED:
        return _SEEDED_PUBLISH[name] in _connected_toolkits(db, workspace_id)
    if candidate != GENERIC:
        return False
    rows = (
        db.query(ComposioActionCache.app_name, ComposioActionCache.parameters)
        .filter(func.upper(ComposioActionCache.action_name) == name)
        .all()
    )
    offering = {_toolkit(row.app_name) for row in rows if _generic_fields(row.parameters)} - set(SEEDED_CHANNELS)
    return bool(offering) and not offering.isdisjoint(_connected_toolkits(db, workspace_id))


# ---- the channels a workspace has ------------------------------------------


@dataclass(frozen=True)
class _Storage:
    public: bool  # media_urls.media_public_url_available() (D9)
    needs: str  # the flag's words, "Needs public storage"


def social_channels(db: Session, workspace_id: Any) -> Tuple[SocialChannel, ...]:
    """The workspace's connected social channels and what each can post (D8, S3.2):
    the seeded ones in the data's order, then the generic ones by toolkit."""
    from modules.socials import media_urls  # object storage stays off the post gate's import path

    connected = _connected_toolkits(db, workspace_id)
    storage = _Storage(media_urls.media_public_url_available(), media_urls.NEEDS_PUBLIC_STORAGE)
    seeded = [adapter for toolkit, adapter in SEEDED_CHANNELS.items() if toolkit in connected]
    wanted: Allowlist = {
        adapter.toolkit: {step.action: frozenset({step.step_class}) for seq in adapter.kinds.values() for step in seq}
        for adapter in seeded
    }
    cached = _cached_actions(db, wanted)
    channels = tuple(
        SocialChannel(
            toolkit=adapter.toolkit,
            label=adapter.label,
            post_kinds=tuple(
                _seeded_kind(adapter.toolkit, kind, seq, cached, storage) for kind, seq in adapter.kinds.items()
            ),
            verified=True,
            setup_note=adapter.setup_note,
        )
        for adapter in seeded
    )
    return channels + _generic_channels(db, workspace_id, connected - set(SEEDED_CHANNELS), storage)


def _seeded_kind(
    toolkit: str, kind: str, steps: Tuple[ChannelStep, ...], cached: Mapping[Tuple[str, str], Any], storage: _Storage,
) -> ChannelKind:
    """Unavailable when the deny list refuses an action it must run (the deny list
    always wins) or the cache lacks one, the first such action named, or when a step
    it must run takes only a link and there is no public storage. An optional step
    that cannot run (refused, missing, or a link with no public storage) is skipped
    at publish instead, and the kind stays available."""
    required = tuple(step for step in steps if not step.optional)
    denial = next((denied for denied in (composio_action_denial(step.action) for step in required) if denied), None)
    missing = next((step.action for step in required if (toolkit, step.action) not in cached), None)
    needs_public = not storage.public and any(step.urls for step in steps)
    blocking = next((step.action for step in required if step.urls), None) if needs_public else None
    if denial or missing:
        reason = denial or MISSING_ACTION.format(slug=missing)
    else:
        reason = NEEDS_PUBLIC_LINK.format(needs=storage.needs, slug=blocking) if blocking else None
    return ChannelKind(kind, reason is None, reason, needs_public, steps)


# ---- the generic adapter (D8 "Generic (text + media)") ---------------------


@dataclass(frozen=True)
class _MediaField:
    name: str
    kinds: Tuple[str, ...]  # the post kinds it carries
    link: bool  # takes a link (D9), else a file
    many: bool  # a list: all the post's media


@dataclass(frozen=True)
class _GenericFields:
    text: str
    media: Tuple[_MediaField, ...]
    media_required: bool


def _media_field(name: str, schema: Any) -> Optional[_MediaField]:
    """A schema property as a media field: a name the rules list, or a property
    flagged with the file marker (a file of any kind)."""
    flagged = isinstance(schema, Mapping) and schema.get(GENERIC_RULES.file_marker) is True
    kinds = GENERIC_RULES.media_fields.get(name.lower()) or ((IMAGE_KIND, VIDEO_KIND) if flagged else ())
    if not kinds:
        return None
    link = not flagged and name.lower().endswith(GENERIC_RULES.url_suffixes)
    return _MediaField(name, tuple(kinds), link, isinstance(schema, Mapping) and schema.get("type") == "array")


def _generic_fields(parameters: Any) -> Optional[_GenericFields]:
    """A cached schema's text field and media fields, when it has both (an empty
    schema, as the bulk sync leaves it, has neither)."""
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if not isinstance(properties, Mapping):
        return None
    text = next((name for name in GENERIC_RULES.text_fields if name in properties), None)
    media = tuple(field for field in (_media_field(str(n), s) for n, s in properties.items()) if field)
    if text is None or not media:
        return None
    required = parameters.get("required")
    names = {name for name in required if isinstance(name, str)} if isinstance(required, list) else set()
    return _GenericFields(text, media, any(field.name in names for field in media))


def _generic_kind(
    kind: str, slug: str, fields: _GenericFields, media: Optional[_MediaField], storage: _Storage,
) -> ChannelKind:
    params: Dict[str, Any] = {fields.text: COPY_SOURCE}
    links: Tuple[str, ...] = ()
    files: Tuple[str, ...] = ()
    if media is not None:
        params[media.name] = MEDIA_LIST_SOURCE if media.many else MEDIA_SOURCE
        links, files = ((media.name,), ()) if media.link else ((), (media.name,))
    step = ChannelStep(GENERIC_STEP_ID, slug, PUBLISH, MappingProxyType(params), files, links, returns=GENERIC_RULES.returns)
    needs_public = bool(links) and not storage.public
    reason = NEEDS_PUBLIC_LINK.format(needs=storage.needs, slug=slug) if needs_public else None
    return ChannelKind(kind, not needs_public, reason, needs_public, (step,))


def _generic_kinds(slug: str, fields: _GenericFields, storage: _Storage) -> Tuple[ChannelKind, ...]:
    """Text alone unless the action requires media, then each kind its media fields
    carry, through a file field before a link (file-first)."""
    kinds = [] if fields.media_required else [_generic_kind(TEXT_KIND, slug, fields, None, storage)]
    for kind in SOCIAL_TARGET_POST_KINDS:
        carriers = [field for field in fields.media if kind in field.kinds]
        if carriers:
            kinds.append(_generic_kind(kind, slug, fields, min(carriers, key=lambda field: field.link), storage))
    return tuple(kinds)


def _generic_offers(db: Session, toolkits: FrozenSet[str], storage: _Storage) -> Dict[str, Tuple[ChannelKind, ...]]:
    """Each toolkit's first cached action, by slug, that creates a post and has a
    text field and a media field. A denied action is never offered."""
    if not toolkits or not GENERIC_RULES.post_words:
        return {}
    named = func.upper(ComposioActionCache.action_name)
    rows = (
        db.query(ComposioActionCache.app_name, ComposioActionCache.action_name, ComposioActionCache.parameters)
        .filter(ComposioActionCache.app_name.in_(sorted(toolkit.upper() for toolkit in toolkits)))
        .filter(or_(*(named.like(f"%{word}%") for word in sorted(GENERIC_RULES.post_words))))
        .order_by(ComposioActionCache.app_name, ComposioActionCache.action_name)
        .all()
    )
    offered: Dict[str, Tuple[ChannelKind, ...]] = {}
    for row in rows:
        toolkit, slug = _toolkit(row.app_name), str(row.action_name).strip().upper()
        fields = _generic_fields(row.parameters) if _creates_a_post(slug) else None
        if toolkit not in offered and fields is not None and not composio_action_denial(slug):
            offered[toolkit] = _generic_kinds(slug, fields, storage)
    return offered


def _published_toolkits(db: Session, workspace_id: Any) -> FrozenSet[str]:
    """The toolkits one of the workspace's targets has published to, lower case."""
    rows = (
        db.query(SocialPostTarget.toolkit)
        .join(SocialPost, SocialPost.id == SocialPostTarget.post_id)
        .filter(SocialPost.workspace_id == _workspace(workspace_id), SocialPostTarget.status == PUBLISHED_TARGET)
        .distinct()
        .all()
    )
    return frozenset(_toolkit(row.toolkit) for row in rows)


def _generic_channels(
    db: Session, workspace_id: Any, toolkits: FrozenSet[str], storage: _Storage,
) -> Tuple[SocialChannel, ...]:
    """The generic channels by toolkit, each an "unverified channel" until one of the
    workspace's targets has published to it."""
    offered = _generic_offers(db, toolkits, storage)
    published = _published_toolkits(db, workspace_id) if offered else frozenset()
    channels = []
    for toolkit, kinds in sorted(offered.items()):
        name = toolkit.replace("_", " ").title()
        verified = toolkit in published
        label = name if verified else f"{name} ({UNVERIFIED_CHANNEL})"
        channels.append(SocialChannel(toolkit, label, kinds, verified, None))
    return tuple(channels)
