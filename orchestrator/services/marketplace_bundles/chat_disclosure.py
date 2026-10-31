"""PRD-270 W2 US-270-14g (§5.6, §5.8, K17, K28-1) — the Cloud item Auto's chat install names, and its card's text.

``cloud_target(db, action, params)`` reads which Marketplace row one of the three install actions names, the way
its handler resolves it:

* platform_install_marketplace_agent: ``agent_id``, else ``agent_name``, as ``install_marketplace_agent_tool``
  hands either to ``package_installer._find_marketplace_agent`` (an id, a public id, a slug or a name);
* platform_install_package: ``slug``, as ``marketplace_packages.get_by_slug`` reads it;
* platform_install_plugin: ``plugin_id`` (a uuid) when it is given, else ``plugin_slug``, as
  ``handlers_marketplace.install_plugin`` reads them.

It answers a ``CloudTarget`` only when a ``marketplace_cloud_items`` mapping points at that row (a Sync with
Automatos Cloud wrote it); a seeded or an operator-made row, or a call that names no row, gives None. The
target's ``bound`` is the call as its card binds it: the row's own key and the server's plumbing, so the click
installs the row the card showed, never a name read again.

``disclosure_markdown(db, kind, key)`` renders US-270-14a's disclosure as the card's text: the name, 'Reviewed on
Automatos Cloud' (never Official), the publisher and version, the prompt or 'No custom prompt: it uses the default
for its type' (cut at ``PROMPT_SHOWN``), the model and whether this install has it, the tools and Composio apps
with 'needs a connection', the skills and plugins, a package's members and its agents' prompts, a plugin's two
reports, and a 'Needs …' line for each thing the Sync could not find here. Every text that came from a bundle is
inert as markdown: a prompt or a report's summary sits in a fenced block whose backtick runs are broken, and
anything else is escaped (each ``\\``, ``[``, ``]``, ``<``, ``>``, ``:`` and ``@``, and a ``www.``), so it renders
as text: no link, image or HTML. The chat's card shows it as plain text; a markdown reader shows the same words.
Nothing in it is hidden from the person approving: a control or format character (a bidi override, a zero-width
or a tag character) is named, as ``(U+202E)``, and a prompt cut at ``PROMPT_SHOWN`` says how much is left out.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.marketplace_cloud import CLOUD_ITEM_KEYS, MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin
from modules.tools.discovery.cloud_install_asks import AGENT_INSTALL, PACKAGE_INSTALL, PLUGIN_INSTALL
from services.marketplace_bundles.disclosure import (
    PACKAGE_ITEM, PERSONA_PROMPT, PLUGIN_ITEM, NotACloudItem, dicts, disclosure, strings,
)
from services.marketplace_bundles.resolve import AGENT_ITEM

AGENT_ID, AGENT_NAME, SLUG, PLUGIN_ID, PLUGIN_SLUG = "agent_id", "agent_name", "slug", "plugin_id", "plugin_slug"
PLUMBING = "_"  # a server-set key's prefix (``_agent_id``): never the model's, kept on the bound call

PROMPT_SHOWN = 4000  # characters of a prompt, or of a report's summary, the card shows
THE_REST = "(the rest is in the Marketplace)"
NOT_SHOWN = "{count} more characters are not shown here"
NO_PROMPT = "No custom prompt: it uses the default for its type"
REVIEWED = "Reviewed on Automatos Cloud"
ON_THIS_INSTALL, NOT_HERE = "on this install", "not on this install"
NEEDS_A_CONNECTION = "needs a connection"
NEEDS = "Needs the {kind} {name}: not on this install"
NO_MODEL = "Model: none named"
NO_CLOUD_REPORT = "Automatos Cloud's review: none kept on this install"
NO_LOCAL_CHECK = "This install's static check: none"
UPDATES_IN_PLACE = "Updates apply to every workspace that enabled it."
UNKNOWN = "unknown"
NONE_LISTED = "none"
CONTENT_PARTS = ("skills", "commands", "agents", "hooks")
KIND_WORDS: Mapping[str, str] = {AGENT_ITEM: "agent", PACKAGE_ITEM: "package", PLUGIN_ITEM: "plugin"}
BLOCK_BREAK = "\n\n"

FENCE = "```"
FENCE_OPEN = FENCE + "text"
ZERO_WIDTH_SPACE = chr(0x200B)  # set between the backticks of a run, so nothing inside a fence closes it
_BACKTICK_RUN = re.compile(r"`{2,}")
_MARKUP = re.compile(r"([\\\[\]<>:@])")  # what opens a link, an image, HTML or an autolink (with ':' and '@')
_WWW = re.compile(r"(www)\.", re.IGNORECASE)  # a www. autolink
# CR, CRLF, NEL and the line and paragraph separators, each read as one newline.
_LINE_ENDINGS = re.compile("\r\n|[\r" + "".join(chr(code) for code in (0x85, 0x2028, 0x2029)) + "]")
HIDDEN_CATEGORIES = frozenset({"Cc", "Cf"})  # control and format characters: shown by name, never as themselves
KEPT = frozenset("\n\t")
NAMED = "(U+{code:04X})"


class CloudTarget(NamedTuple):
    """The Cloud item an install call names: its kind, its disclosure key (the agent's id, the package's slug,
    the plugin's uuid), how the card names it, and the call bound to that row."""

    kind: str
    key: str
    label: str
    bound: Mapping[str, Any]


# --------------------------------------------------------------------------- the item a call names


def cloud_target(db: Session, action: str, params: Any) -> Optional[CloudTarget]:
    """The Cloud item ``action`` names, or None: another action, a call that names no row, or a row no mapping
    points at (a seeded or an operator item)."""
    find = _TARGETS.get(action)
    if find is None or not isinstance(params, Mapping):
        return None
    return find(db, params)


def _agent_target(db: Session, params: Mapping[str, Any]) -> Optional[CloudTarget]:
    """The marketplace agent install_marketplace_agent_tool would install, when it is a Cloud agent."""
    from services.package_installer import _find_marketplace_agent

    agent_id, agent_name = params.get(AGENT_ID), params.get(AGENT_NAME)
    if agent_id is None and not agent_name:
        return None
    agent = _find_marketplace_agent(db, str(agent_id) if agent_id is not None else str(agent_name))
    if agent is None or not _mapped(db, AGENT_ITEM, agent.id):
        return None
    return CloudTarget(AGENT_ITEM, str(agent.id), f"agent #{agent.id}", _bound(params, AGENT_ID, agent.id))


def _package_target(db: Session, params: Mapping[str, Any]) -> Optional[CloudTarget]:
    """The package install_package_tool would install, when it is a Cloud package."""
    from services.marketplace_packages import get_by_slug

    slug = params.get(SLUG)
    if not isinstance(slug, str) or not slug:
        return None  # the handler refuses it, or its lookup fails: nothing installs
    package = get_by_slug(db, slug)
    if package is None or not _mapped(db, PACKAGE_ITEM, package.id):
        return None
    return CloudTarget(PACKAGE_ITEM, package.slug, f"package {inline(package.slug)}", _bound(params, SLUG, package.slug))


def _plugin_target(db: Session, params: Mapping[str, Any]) -> Optional[CloudTarget]:
    """The plugin install_plugin would enable, when it is a Cloud plugin."""
    plugin = _plugin_named(db, params.get(PLUGIN_ID), params.get(PLUGIN_SLUG))
    if plugin is None or not _mapped(db, PLUGIN_ITEM, plugin.id):
        return None
    label = f"plugin {inline(plugin.slug)}"
    return CloudTarget(PLUGIN_ITEM, str(plugin.id), label, _bound(params, PLUGIN_ID, str(plugin.id)))


def _plugin_named(db: Session, plugin_id: Any, plugin_slug: Any) -> Optional[MarketplacePlugin]:
    """The plugin install_plugin resolves: by ``plugin_id`` when it is given, else by ``plugin_slug``."""
    query = db.query(MarketplacePlugin)
    if plugin_id:
        try:
            wanted = UUID(str(plugin_id))
        except ValueError:
            return None  # install_plugin cannot read it either: nothing installs
        return query.filter(MarketplacePlugin.id == wanted).first()
    if not isinstance(plugin_slug, str) or not plugin_slug:
        return None
    return query.filter(MarketplacePlugin.slug == plugin_slug).first()


def _bound(params: Mapping[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """The call as its card binds it: the row's own key and the server's plumbing; any other key is left out."""
    return {**{name: v for name, v in params.items() if str(name).startswith(PLUMBING)}, key: value}


def _mapped(db: Session, kind: str, local_id: Any) -> bool:
    """Whether a ``marketplace_cloud_items`` mapping points at the ``kind`` row ``local_id``."""
    column = getattr(MarketplaceCloudItem, CLOUD_ITEM_KEYS[kind])
    found = db.query(MarketplaceCloudItem.id).filter(MarketplaceCloudItem.item_type == kind, column == local_id)
    return found.first() is not None


_TARGETS: Mapping[str, Callable[[Session, Mapping[str, Any]], Optional[CloudTarget]]] = {
    AGENT_INSTALL: _agent_target, PACKAGE_INSTALL: _package_target, PLUGIN_INSTALL: _plugin_target,
}


# --------------------------------------------------------------------------- the card's text


def disclosure_markdown(db: Session, kind: str, key: str) -> str:
    """The card's text for the Cloud item of ``kind`` with the local ``key``: its disclosure, every bundle text
    inert. Raises NotACloudItem as ``disclosure`` does, and for a kind no chat install takes (a Playbook)."""
    render = _RENDERERS.get(kind)
    if render is None:
        raise NotACloudItem("no chat install takes that kind")
    shown = disclosure(db, kind, key)
    blocks = [_heading(shown), *render(shown), _needs(shown.get("unresolved"))]
    return BLOCK_BREAK.join(block for block in blocks if block)


def shown_as_text(text: str) -> str:
    """``text`` with every line ending a newline and every other control or format character (a bidi override, a
    zero-width or a tag character) named, as ``(U+202E)``: the card hides nothing the text holds."""
    return "".join(
        char if char in KEPT or unicodedata.category(char) not in HIDDEN_CATEGORIES else NAMED.format(code=ord(char))
        for char in _LINE_ENDINGS.sub("\n", text)
    )


def inline(value: Any) -> str:
    """A value from a bundle on one line, escaped so that markdown renders it as text: nothing in it opens a link,
    an image, HTML or an autolink."""
    one_line = " ".join(shown_as_text(str(value)).split())
    return _WWW.sub(r"\1\\.", _MARKUP.sub(r"\\\1", one_line))


def fenced(text: str, shown: int = PROMPT_SHOWN) -> str:
    """``text`` in a fenced block, cut at ``shown`` characters (saying how many are left out), each backtick run
    broken so nothing in it closes the fence: inside one, markdown renders nothing (no link, image or HTML)."""
    visible = shown_as_text(text)
    body = _BACKTICK_RUN.sub(lambda run: ZERO_WIDTH_SPACE.join(run.group()), visible[:shown])
    left_out = len(visible) - shown
    rest = f"\n{NOT_SHOWN.format(count=left_out)} {THE_REST}" if left_out > 0 else ""
    return f"{FENCE_OPEN}\n{body}\n{FENCE}{rest}"


def _heading(shown: Mapping[str, Any]) -> str:
    kind = KIND_WORDS.get(shown.get("kind"), "item")
    return "\n".join([
        f"Add the {kind} {inline(shown.get('name'))} from Automatos Cloud to this workspace",
        "",
        f"- {REVIEWED}",
        f"- Publisher: {inline(shown.get('publisher') or UNKNOWN)}",
        f"- Version: {inline(shown.get('version'))}",
    ])


def _needs(unresolved: Any) -> str:
    """A 'Needs …' line for each thing the Sync recorded as not found here."""
    return "\n".join(
        f"- {NEEDS.format(kind=inline(entry.get('kind')), name=inline(entry.get('name')))}"
        for entry in dicts(unresolved)
    )


def _listed(title: str, lines: Sequence[str]) -> str:
    if not lines:
        return f"{title}: {NONE_LISTED}"
    return "\n".join([f"{title}:", "", *(f"- {line}" for line in lines)])


def _here_line(name: Any, here: Any) -> str:
    return inline(name) if here else f"{inline(name)}: {NOT_HERE}"


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _prompt_block(title: str, prompt: Any) -> str:
    """A prompt as the card shows it: whose it is, then its text in a fence; or the no-prompt line."""
    prompt = prompt if isinstance(prompt, Mapping) else {}
    text = prompt.get("text")
    if not isinstance(text, str) or not text.strip():
        return f"{title}: {NO_PROMPT}."
    persona = prompt.get("personaName")
    whose = f" (from its persona {inline(persona)})" if prompt.get("source") == PERSONA_PROMPT and persona else ""
    return f"{title}{whose}:\n\n{fenced(text)}"


# --------------------------------------------------------------------------- an agent


def _agent_blocks(shown: Mapping[str, Any]) -> List[str]:
    return [
        _prompt_block("Its prompt", shown.get("prompt")),
        _model_line(shown.get("model")),
        _listed("Tools and apps", [_app_line(app) for app in dicts(shown.get("apps"))]),
        _listed("Skills", [_here_line(skill.get("name"), skill.get("here")) for skill in dicts(shown.get("skills"))]),
        _listed("Plugins", [_here_line(plugin.get("slug"), plugin.get("here")) for plugin in dicts(shown.get("plugins"))]),
    ]


def _model_line(model: Any) -> str:
    if not isinstance(model, Mapping):
        return NO_MODEL
    provider = f" ({inline(model.get('provider'))})" if model.get("provider") else ""
    return f"Model: {inline(model.get('modelId'))}{provider}: {ON_THIS_INSTALL if model.get('here') else NOT_HERE}"


def _app_line(app: Mapping[str, Any]) -> str:
    return f"{inline(app.get('name'))}: {NEEDS_A_CONNECTION}" if app.get("needsConnection") else inline(app.get("name"))


# --------------------------------------------------------------------------- a package


def _package_blocks(shown: Mapping[str, Any]) -> List[str]:
    members = [_here_line(f"{member.get('type')} {member.get('name')}", member.get("here"))
               for member in dicts(shown.get("members"))]
    playbooks = [f"{inline(playbook.get('name'))}: {_count(playbook.get('steps'))} steps"
                 for playbook in dicts(shown.get("playbooks"))]
    return [
        _listed("Members", members),
        _listed("Apps it connects", [_connect_line(connect) for connect in dicts(shown.get("connects"))]),
        *(_prompt_block(f"The agent {inline(agent.get('name'))}'s prompt", agent.get("prompt"))
          for agent in dicts(shown.get("agents"))),
        _listed("Playbooks", playbooks),
    ]


def _connect_line(connect: Mapping[str, Any]) -> str:
    optional = " (optional)" if connect.get("optional") else ""
    note = f": {inline(connect.get('note'))}" if connect.get("note") else ""
    return f"{inline(connect.get('app'))}{optional}{note}"


# --------------------------------------------------------------------------- a plugin


def _plugin_blocks(shown: Mapping[str, Any]) -> List[str]:
    contents = shown.get("contents") if isinstance(shown.get("contents"), Mapping) else {}
    return [
        _cloud_report(shown.get("scan")),
        _local_check(shown.get("localCheck")),
        "Contents: " + ", ".join(f"{_count(contents.get(part))} {part}" for part in CONTENT_PARTS),
        _listed("Skills it adds", [inline(name) for name in strings(shown.get("skills"))]),
        UPDATES_IN_PLACE if shown.get("updatesInPlace") else "",
    ]


def _cloud_report(scan: Any) -> str:
    """Automatos Cloud's review as the Sync kept it: its verdict and risk, its summary in a fence, its findings."""
    if not isinstance(scan, Mapping):
        return NO_CLOUD_REPORT
    head = [f"Automatos Cloud's review: {inline(scan.get('verdict'))}, risk score {inline(scan.get('riskScore'))}"]
    summary = scan.get("summary")
    told = ["", fenced(summary)] if isinstance(summary, str) and summary.strip() else []
    findings = [f"- {inline(finding.get('severity'))}: {inline(finding.get('description'))}"
                for finding in dicts(scan.get("findings"))]
    return "\n".join([*head, *told, *([""] + findings if findings else [])])


def _local_check(check: Any) -> str:
    """This install's static re-run: its status and each finding's severity, kind and place."""
    if not isinstance(check, Mapping):
        return NO_LOCAL_CHECK
    findings = [
        f"- {inline(finding.get('severity'))} {inline(finding.get('type'))} in {inline(finding.get('file'))} "
        f"(line {inline(finding.get('line'))}): {inline(finding.get('description'))}"
        for finding in dicts(check.get("findings"))
    ]
    head = f"This install's static check: {inline(check.get('status'))}"
    return "\n".join([head, *([""] + findings if findings else [])])


_RENDERERS: Mapping[str, Callable[[Mapping[str, Any]], List[str]]] = {
    AGENT_ITEM: _agent_blocks, PACKAGE_ITEM: _package_blocks, PLUGIN_ITEM: _plugin_blocks,
}


__all__ = [
    "NEEDS", "NOT_SHOWN", "NO_PROMPT", "PROMPT_SHOWN", "REVIEWED", "THE_REST", "CloudTarget", "cloud_target",
    "disclosure_markdown", "fenced", "inline", "shown_as_text",
]
