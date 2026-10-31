"""PRD-270 US-270-16 (§5.6, K20): the docs say what a Sync brings besides themes.

Pure file reads, no services. They lock what the story asks for:

* ``docs/community-marketplace/sync-with-automatos-cloud.md`` has the section 'Agents, Playbooks, packages and
  plugins' with its parts (Reviewed on Automatos Cloud vs Official, what a bundle holds, the plugins, the review
  before Add, the update notices, the two edges), names the wave's three settings, each a ``config.py`` attribute
  with the default the page gives, names every hosted catalogue route of ``reports/route-manifest.json``, and its
  in-page links land on a heading;
* ``docs/getting-started/self-hosting.md``'s community hub row names all five types as shipped, pull only and the
  button, and no longer calls any of them planned;
* neither file holds a local path, a partner's name or a secret (a public repository), with W1's checks.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List

from tests.test_prd270w1_docs import (
    _CONFIG, _HOME_FOLDER, _PAGE, _PARTNER_NAME_SHA256, _PLACEHOLDER_USER, _PUBLIC_DOCS, _ROUTE_MANIFEST,
    _SECRET_SHAPES, _WORD, _hub_row, _section, _text,
)

_ITEMS_SECTION = "## Agents, Playbooks, packages and plugins"
_SETTINGS_SECTION = "## Settings"
_REVIEWED = "Reviewed on Automatos Cloud"
_PARTS = (
    "### Reviewed on Automatos Cloud, never Official",
    "### What Automatos Cloud lists",
    "### What a bundle holds, and what it never holds",
    "### How a Sync brings them",
    "### Plugins",
    "### Before Add: what it will use",
    "### When Auto adds one from chat",
    "### Update notices",
    "### Two edges: a deleted source and a deleted row",
)
# Each setting the wave adds, and where the page gives its default: a Settings table row, or a sentence.
_LOCAL_SETTINGS = ("AUTOMATOS_CLOUD_BUNDLE_MAX_KB", "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC")
_HOSTED_SETTING = "MARKETPLACE_CATALOG_BUNDLE_MAX_KB"
_DEFAULT = r'os\.getenv\("{name}", "([^"]*)"\)'
_GIVEN_DEFAULT = r"`{name}`\s+\({default} by default\)"
_TABLE_ROW = r"^\|\s*`{name}`\s*\|\s*`{default}`\s*\|"
_CATALOG_ROUTE = "/api/marketplace/catalog"
_PATH_PARAM = re.compile(r"\{([a-z_]+)(?::path)?\}")

_FIVE_TYPES = ("themes", "agents", "playbooks", "packages", "plugins")
_PLANNED = re.compile(r"\bplanned\b|\bphase 2\b", re.IGNORECASE)
_W1_PLANNED_SENTENCE = "are still planned"

_HEADING = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)
_IN_PAGE_LINK = re.compile(r"\]\(#([^)]+)\)")


def _default(name: str) -> str:
    found = re.search(_DEFAULT.format(name=name), _text(_CONFIG))
    assert found, f"config.py doesn't read {name} with a default"
    return found.group(1)


def _slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _as_written(path: str) -> str:
    """A manifest path as the page writes it: ``{digest}`` and ``{path:path}`` read ``<digest>`` and ``<path>``."""
    return _PATH_PARAM.sub(r"<\1>", path)


def _hosted_routes() -> List[str]:
    """Each hosted catalogue GET of the manifest, as the page writes it."""
    routes = json.loads(_text(_ROUTE_MANIFEST))["routes"]
    return sorted(
        f"GET {_as_written(route['path'])}" for route in routes
        if route["method"] == "GET"
        and (route["path"] == _CATALOG_ROUTE or route["path"].startswith(f"{_CATALOG_ROUTE}/"))
    )


def test_the_page_has_the_cloud_items_section_and_its_parts() -> None:
    page = _text(_PAGE)
    assert _ITEMS_SECTION in page, "the page has no section on agents, Playbooks, packages and plugins"
    section = _section(page, _ITEMS_SECTION)
    missing = [part for part in _PARTS if part not in section]
    assert not missing, f"the section lacks {missing}"
    assert section.count(_REVIEWED) >= 2, f"the section doesn't say {_REVIEWED!r}"
    assert "never Official" in section and '"reviewed"' in section
    assert _W1_PLANNED_SENTENCE not in page, "the page still calls the other types planned"


def test_the_page_names_the_three_settings_with_their_defaults() -> None:
    page = _text(_PAGE)
    settings = _section(page, _SETTINGS_SECTION)
    for name in _LOCAL_SETTINGS:
        row = _TABLE_ROW.format(name=name, default=re.escape(_default(name)))
        assert re.search(row, settings, re.MULTILINE), f"the Settings table has no row for {name} with its default"
        assert f"`{name}`" in _section(page, _ITEMS_SECTION), f"the Cloud items section doesn't name {name}"
    given = _GIVEN_DEFAULT.format(name=_HOSTED_SETTING, default=re.escape(_default(_HOSTED_SETTING)))
    assert re.search(given, page), f"the page doesn't give {_HOSTED_SETTING}'s default"


def test_the_page_names_every_hosted_catalogue_route() -> None:
    routes = _hosted_routes()
    assert {f"GET {_CATALOG_ROUTE}", f"GET {_CATALOG_ROUTE}/bundles/<revision>",
            f"GET {_CATALOG_ROUTE}/plugins/<slug>/<version>/package.zip"} <= set(routes), routes
    page = _text(_PAGE)
    missing = [route for route in routes if f"`{route}`" not in page]
    assert not missing, f"the page doesn't name {missing}"


def test_the_pages_in_page_links_land_on_a_heading() -> None:
    page = _text(_PAGE)
    anchors: Dict[str, str] = {_slug(heading): heading for heading in _HEADING.findall(page)}
    links = _IN_PAGE_LINK.findall(page)
    assert links, "the page links none of its own sections"
    broken = sorted({link for link in links if link not in anchors})
    assert not broken, f"in-page links with no heading: {broken}"


def test_the_hub_row_ships_all_five_types_pull_only() -> None:
    row = _hub_row()
    lowered = row.lower()
    missing = [kind for kind in _FIVE_TYPES if not re.search(rf"\b{kind}\b", lowered)]
    assert not missing, f"the hub row doesn't name {missing}"
    assert "shipped" in lowered and "pull only" in lowered and "button" in lowered
    assert _REVIEWED in row
    assert not _PLANNED.search(row), "the hub row still calls a type planned"


def test_no_local_path_partner_name_or_secret_in_either_file() -> None:
    for doc in _PUBLIC_DOCS:
        text = _text(doc)
        users = {match.group(1) for match in _HOME_FOLDER.finditer(text)}
        assert users <= {_PLACEHOLDER_USER}, f"{doc.name} holds a local path for {sorted(users)}"
        words = set(_WORD.findall(text.lower()))
        assert not [word for word in words if hashlib.sha256(word.encode()).hexdigest() in _PARTNER_NAME_SHA256], (
            f"{doc.name} names a partner"
        )
        shapes = [shape.pattern for shape in _SECRET_SHAPES if shape.search(text)]
        assert not shapes, f"{doc.name} holds something shaped like a secret: {shapes}"
