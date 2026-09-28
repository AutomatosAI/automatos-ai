"""PRD-251 Wave 2, P251W1-RVW-7 — Wave 1 housekeeping: the config manifest names
every Socials setting (a), and a starter's seed file loses nothing (b).

Pins:

* **The manifest (a).** ``reports/config-surface.json`` names every environment
  variable ``config.py`` reads whose name starts with ``SOCIALS_``, and
  ``PLAYBOOK_PROGRESS_STAMP_SECONDS`` (US-120), in sorted position: names only.
  The deletion guard (tests/test_config_surface_guard.py) only warns on a name
  the manifest lacks; this fails. The names are read from config.py's source,
  so the next Socials setting is held to it too.
* **A starter's css (b).** The template contract takes a top-level ``css`` block,
  injected after the brand styles (``core/social_templates.py``). A seed file's
  css reaches the starter's blocks, the contract checks it (the brand rule, the
  placeholders, text), the seeder writes it to the row and the render bundle
  carries it. Every block the contract takes but the html (the seed's .html
  file) is a seed field, and a seed file field that is none of them is refused,
  named: nothing is dropped unread.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional, Set

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.models  # noqa: E402,F401  (registers every mapper)
import modules.documents.social_starters as starters_module  # noqa: E402
from core.media_render_bundle import build_bundle  # noqa: E402
from core.social_templates import (  # noqa: E402
    BLOCK_KEYS,
    SocialTemplateError,
    resolve_variables,
    validate_social_blocks,
)
from modules.documents.seed_templates import starter_columns  # noqa: E402

CONFIG_SOURCE = _ORCH / "config.py"
MANIFEST = _ORCH / "reports" / "config-surface.json"
SOCIALS_PREFIX = "SOCIALS_"
NAMED_SETTINGS = ("PLAYBOOK_PROGRESS_STAMP_SECONDS",)
WS = uuid.UUID("00000000-0000-0000-0000-0000000002b7")
CSS_STARTER = "title-card"
STARTER_CSS = "h1 { letter-spacing: -0.02em; color: var(--brand-primary); }"


# ---------------------------------------------------------------------------
# (a) The manifest names every Socials setting config.py reads
# ---------------------------------------------------------------------------


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _env_key(node: ast.AST) -> Optional[ast.AST]:
    """The key ``node`` reads the environment with: ``os.getenv(k)``, ``os.environ.get(k)``, ``os.environ[k]``."""
    if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
        return node.slice
    if not (isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute)):
        return None
    func = node.func
    getenv = func.attr == "getenv" and isinstance(func.value, ast.Name) and func.value.id == "os"
    environ_get = func.attr == "get" and _is_os_environ(func.value)
    return node.args[0] if getenv or environ_get else None


def env_names_read(source: str) -> Set[str]:
    """Every environment name ``source`` reads by a literal name."""
    keys = (_env_key(node) for node in ast.walk(ast.parse(source)))
    return {key.value for key in keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def _manifest_names():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["settings"]


def test_the_net_finds_every_way_config_reads_the_environment():
    source = "\n".join(
        [
            'A = os.getenv("SOCIALS_ONE", "1")',
            'B = os.environ.get("SOCIALS_TWO")',
            'C = os.environ["SOCIALS_THREE"]',
            'D = os.getenv(\n    "SOCIALS_FOUR", "x"\n).strip()',
            'E = other.getenv("SOCIALS_NOT_THE_ENVIRONMENT")',
            '# F = os.getenv("SOCIALS_IN_A_COMMENT")',
        ]
    )
    assert env_names_read(source) == {"SOCIALS_ONE", "SOCIALS_TWO", "SOCIALS_THREE", "SOCIALS_FOUR"}


def test_the_manifest_names_every_socials_setting_config_reads():
    reads = env_names_read(CONFIG_SOURCE.read_text(encoding="utf-8"))
    socials = sorted(name for name in reads if name.startswith(SOCIALS_PREFIX))
    # The net bites: Wave 0's switch, a Wave 1 cap and the post gate's cache are among them.
    assert {"SOCIALS_ENABLED_DEFAULT", "SOCIALS_MEDIA_MONTHLY_CAP_USD", "SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS"} <= set(socials)
    assert set(NAMED_SETTINGS) <= reads
    listed = set(_manifest_names())
    missing = [name for name in socials + list(NAMED_SETTINGS) if name not in listed]
    assert missing == [], (
        f"config.py reads {missing}, which reports/config-surface.json does not name. "
        "Add each name (never its value) in sorted position."
    )


def test_the_manifest_stays_sorted_with_each_name_once():
    names = _manifest_names()
    assert names == sorted(set(names))


# ---------------------------------------------------------------------------
# (b) A starter's seed file loses nothing: its css is carried and checked
# ---------------------------------------------------------------------------


@pytest.fixture
def seed_dir(monkeypatch, tmp_path):
    """A copy of every seed file, loaded in place of the real ones."""
    for slug in starters_module.SOCIAL_STARTER_SLUGS:
        for suffix in (".html", ".json"):
            shutil.copyfile(starters_module.STARTERS_DIR / f"{slug}{suffix}", tmp_path / f"{slug}{suffix}")
    monkeypatch.setattr(starters_module, "STARTERS_DIR", tmp_path)
    starters_module._starters.cache_clear()
    yield tmp_path
    starters_module._starters.cache_clear()


def _set_seed_field(seed_dir: Path, slug: str, key: str, value) -> None:
    path = seed_dir / f"{slug}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, key: value}), encoding="utf-8")


def _refusal() -> list:
    """The errors loading the starters is refused with."""
    with pytest.raises(SocialTemplateError) as refused:
        starters_module.social_starters()
    return refused.value.errors


def test_every_block_the_contract_takes_but_the_html_is_a_seed_field():
    assert set(starters_module.BLOCK_FIELDS) == set(BLOCK_KEYS) - {"html"}
    assert "css" in starters_module.BLOCK_FIELDS
    assert set(starters_module.SEED_FIELDS) == (
        set(starters_module.BLOCK_FIELDS) | set(starters_module.ROW_FIELDS) | {starters_module.PREVIEW_FIELD}
    )


def test_the_seeded_starters_carry_only_seed_fields_and_all_load():
    starters = starters_module.social_starters()
    assert [s["slug"] for s in starters] == list(starters_module.SOCIAL_STARTER_SLUGS)
    for slug in starters_module.SOCIAL_STARTER_SLUGS:
        meta = json.loads((starters_module.STARTERS_DIR / f"{slug}.json").read_text(encoding="utf-8"))
        assert set(meta) <= set(starters_module.SEED_FIELDS), slug


def test_a_starter_css_reaches_its_blocks_its_row_and_its_render(seed_dir):
    _set_seed_field(seed_dir, CSS_STARTER, "css", STARTER_CSS)
    starter = next(s for s in starters_module.social_starters() if s["slug"] == CSS_STARTER)
    blocks = starter["blocks"]
    assert blocks["css"] == STARTER_CSS
    assert validate_social_blocks(blocks, starter["format"]) == blocks
    assert starter_columns(starter)["blocks"]["css"] == STARTER_CSS

    resolved = resolve_variables(blocks["variables_schema"], starter["sample_data"])
    assert resolved.missing == [] and resolved.invalid == []
    bundle = build_bundle(
        workspace_id=WS,
        reference=f"seeded:{CSS_STARTER}",
        blocks=blocks,
        values=resolved.values,
        brand_kit={},
        fmt=starter["format"],
    )
    assert bundle["composition"]["css"] == STARTER_CSS


@pytest.mark.parametrize(
    ("css", "finding"),
    [
        pytest.param("h1 { color: #e96235; }", "#e96235", id="a-hardcoded-colour"),
        pytest.param('h1::after { content: "{{ nowhere }}"; }', "{{ nowhere }}", id="an-undeclared-variable"),
        pytest.param(42, "must be text", id="not-text"),
    ],
)
def test_a_starter_css_is_checked_by_the_contract(seed_dir, css, finding):
    _set_seed_field(seed_dir, CSS_STARTER, "css", css)
    errors = _refusal()
    assert any(e["field"] == f"{CSS_STARTER}.css" and finding in e["message"] for e in errors), errors


@pytest.mark.parametrize("field", ["stils", "html", "script"])
def test_a_seed_file_field_the_starter_does_not_take_is_refused_naming_it(seed_dir, field):
    _set_seed_field(seed_dir, CSS_STARTER, field, "…")
    errors = _refusal()
    assert [e["field"] for e in errors] == [f"{CSS_STARTER}.{field}"]
    assert "is not a field of a social starter" in errors[0]["message"]
