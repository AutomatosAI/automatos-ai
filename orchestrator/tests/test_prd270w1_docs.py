"""PRD-270 US-270-06: the Sync with Automatos Cloud docs exist and say what the code does.

Pure file reads, no services. They lock what the story asks for:

* ``docs/community-marketplace/sync-with-automatos-cloud.md`` names every local setting
  (each ``AUTOMATOS_CLOUD_*`` attribute of ``orchestrator/config.py``), both Sync routes (the
  ``/api/marketplace/cloud-sync`` entries of ``reports/route-manifest.json``) and every header a
  read from Automatos Cloud sends (``services/marketplace_cloud/fetch.py`` ``request_headers``),
  so a setting, route or header added later fails here until the page says it;
* the page is linked from the Marketplace index and the Themes tab page, and its relative links
  resolve;
* ``docs/getting-started/self-hosting.md``'s community hub row no longer calls themes "Planned,
  not shipped": it names Sync with Automatos Cloud and links the page;
* neither file holds a local path, a partner's name or a secret (a public repository).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import List, Set

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MARKETPLACE_DOCS = _REPO_ROOT / "docs" / "community-marketplace"
_PAGE = _MARKETPLACE_DOCS / "sync-with-automatos-cloud.md"
_INDEX = _MARKETPLACE_DOCS / "_index.md"
_THEMES_TAB = _MARKETPLACE_DOCS / "themes-tab.md"
_SELF_HOSTING = _REPO_ROOT / "docs" / "getting-started" / "self-hosting.md"
_CONFIG = _REPO_ROOT / "orchestrator" / "config.py"
_ROUTE_MANIFEST = _REPO_ROOT / "orchestrator" / "reports" / "route-manifest.json"
_FETCH = _REPO_ROOT / "orchestrator" / "services" / "marketplace_cloud" / "fetch.py"
_PUBLIC_DOCS = (_PAGE, _SELF_HOSTING)

_PAGE_NAME = "sync-with-automatos-cloud.md"
_SYNC_ROUTE = "/api/marketplace/cloud-sync"
_KNOWN_SETTINGS = {"AUTOMATOS_CLOUD_URL", "AUTOMATOS_CLOUD_FEED_MAX_KB", "AUTOMATOS_CLOUD_TIMEOUT_SECONDS"}
_LOCAL_SETTING = re.compile(r"^\s+(AUTOMATOS_CLOUD_[A-Z0-9_]+)\s*:", re.MULTILINE)
_REQUEST_HEADERS = re.compile(r"def request_headers\(.*?return \{(.*?)\}", re.DOTALL)
_HEADER_NAME = re.compile(r'"([A-Za-z-]+)":')
_WHAT_LEAVES = "## What leaves your install"

_HUB_ROW = "| The community hub |"
_OLD_HUB_STATUS = "Planned, not shipped"
_SYNC_NAME = "Sync with Automatos Cloud"
_RELATIVE_LINK = re.compile(r"\]\(((?:\.\./|\./)?[^)#\s:]+\.(?:md|py|ts|tsx|js|json|defaults))(?:#[^)]*)?\)")

# A user's home folder (the AC's /Users/ and C:\, plus /home/); the docs' own placeholder user is `you`.
_HOME_FOLDER = re.compile(r"(?:/Users/|/home/|\b[A-Za-z]:\\(?:Users\\)?)([^/\\\s`'\")]*)")
_PLACEHOLDER_USER = "you"
# The partners' names, kept as the sha256 of a lower-case word so this public test names none of them.
_PARTNER_NAME_SHA256 = frozenset({"02ea5273aa03c5598156e043ebb77d01c36d9a50dde47269c8feb332e0656b1e"})
_WORD = re.compile(r"[a-z0-9]+")
# The shapes of the secrets this platform handles: provider, Stripe, GitHub, Slack, AWS and Google keys, private keys.
_SECRET_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{10,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _text(doc: pathlib.Path) -> str:
    return doc.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` up to the next heading of the same or a higher level."""
    start = text.index(heading) + len(heading)
    level = heading.split(" ", 1)[0]
    following = re.search(rf"^#{{1,{len(level)}}} ", text[start:], re.MULTILINE)
    return text[start:start + following.start()] if following else text[start:]


def _local_settings() -> Set[str]:
    return set(_LOCAL_SETTING.findall(_text(_CONFIG)))


def _sync_routes() -> List[str]:
    routes = json.loads(_text(_ROUTE_MANIFEST))["routes"]
    return sorted(
        f"{route['method']} {route['path']}" for route in routes
        if route["path"] == _SYNC_ROUTE or route["path"].startswith(f"{_SYNC_ROUTE}/")
    )


def _sent_headers() -> List[str]:
    body = _REQUEST_HEADERS.search(_text(_FETCH))
    assert body, "fetch.py has no request_headers to read"
    return _HEADER_NAME.findall(body.group(1))


def _hub_row() -> str:
    rows = [line for line in _text(_SELF_HOSTING).splitlines() if line.startswith(_HUB_ROW)]
    assert len(rows) == 1, f"self-hosting.md has {len(rows)} community hub rows"
    return rows[0]


def test_the_page_names_every_local_setting() -> None:
    settings = _local_settings()
    assert _KNOWN_SETTINGS <= settings, f"config.py lost a setting: {_KNOWN_SETTINGS - settings}"
    page = _text(_PAGE)
    missing = sorted(name for name in settings if f"`{name}`" not in page)
    assert not missing, f"the page doesn't name {missing}"


def test_the_page_names_both_sync_routes() -> None:
    routes = _sync_routes()
    assert {f"GET {_SYNC_ROUTE}/status", f"POST {_SYNC_ROUTE}"} <= set(routes), f"the manifest lists {routes}"
    page = _text(_PAGE)
    missing = [route for route in routes if f"`{route}`" not in page]
    assert not missing, f"the page doesn't name {missing}"


def test_the_page_names_every_header_a_read_sends() -> None:
    headers = _sent_headers()
    assert headers, "request_headers sends no header"
    leaves = _section(_text(_PAGE), _WHAT_LEAVES)
    missing = [name for name in headers if f"`{name}" not in leaves]
    assert not missing, f"'What leaves your install' doesn't name {missing}"
    for withheld in ("cookie", "Authorization", "install id"):
        assert withheld in leaves, f"'What leaves your install' doesn't say no {withheld} is sent"


def test_the_page_is_linked_from_the_index_and_the_themes_tab() -> None:
    for doc in (_INDEX, _THEMES_TAB):
        assert f"]({_PAGE_NAME}" in _text(doc), f"{doc.name} doesn't link {_PAGE_NAME}"


def test_the_pages_relative_links_resolve() -> None:
    links = _RELATIVE_LINK.findall(_text(_PAGE))
    assert links, "the page links nothing"
    broken = [link for link in links if not (_PAGE.parent / link).resolve().is_file()]
    assert not broken, f"broken links: {broken}"


def test_the_hub_row_ships_themes_and_links_the_page() -> None:
    row = _hub_row()
    assert _OLD_HUB_STATUS not in row
    assert _SYNC_NAME in row
    link = f"../community-marketplace/{_PAGE_NAME}"
    assert f"]({link})" in row
    assert (_SELF_HOSTING.parent / link).resolve() == _PAGE.resolve()


def test_no_local_path_in_either_file() -> None:
    for doc in _PUBLIC_DOCS:
        users = {match.group(1) for match in _HOME_FOLDER.finditer(_text(doc))}
        assert users <= {_PLACEHOLDER_USER}, f"{doc.name} holds a local path for {sorted(users)}"


def test_no_partner_name_in_either_file() -> None:
    for doc in _PUBLIC_DOCS:
        words = set(_WORD.findall(_text(doc).lower()))
        named = [word for word in words if hashlib.sha256(word.encode()).hexdigest() in _PARTNER_NAME_SHA256]
        assert not named, f"{doc.name} names a partner"


def test_no_secret_in_either_file() -> None:
    for doc in _PUBLIC_DOCS:
        text = _text(doc)
        shapes = [shape.pattern for shape in _SECRET_SHAPES if shape.search(text)]
        assert not shapes, f"{doc.name} holds something shaped like a secret: {shapes}"
