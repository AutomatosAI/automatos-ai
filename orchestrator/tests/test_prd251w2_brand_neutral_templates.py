"""PRD-251 Wave 2, P251W1-RVW-6 — brand-true templates: the seeded starters carry no
Automatos copy (a), and the brand rule passes only what no brand kit would set (b).

Turning Socials on seeds the twelve starters into every workspace, sample data
and all, and an agent reads a template's variables through
``platform_get_template_schema``, which returns that sample data. An agent short
of a value copies the example, so the examples are brand-neutral: ``@yourbrand``,
``yourbrand.com``, ``Your app``. Pins:

* **The seed files.** No string anywhere in a seed file under
  ``modules/documents/templates/social/`` (its name, description, sample data,
  and every label, default or example of its variables) carries Automatos's
  name, domains or handle, its assistant, founder or licence, its Academy or
  Markets product, or its tagline. The net bites on each of those.
* **What an agent reads.** ``platform_get_template_schema``, dispatched through
  ``PlatformActionExecutor.execute`` on a workspace ``seed_social_starters`` seeded,
  answers every seeded template with no Automatos copy in it.
* **The previews keep the references' copy (Goal 8).** The media-render CI job
  lays each reference video's own copy (``scripts/ci/social_reference_text.json``)
  over its starter's neutral sample, so the owner's side-by-side still reads as
  the reference did; the seeded row never carries it.
* **What the brand rule passes (b).** ``transparent``, ``currentColor``, black or
  white mixed in by ``color-mix()``, channels a script computes from a
  ``--brand-*`` variable, an alpha mask's black: none is a brand colour. What it
  finds is pinned in test_prd251w1_social_templates.py
  (``test_the_brand_rule_finds_what_a_template_hardcodes``).
"""
from __future__ import annotations

import asyncio
import html
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from sqlalchemy.sql.elements import BindParameter  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
from core.media_render_bundle import voice_script  # noqa: E402
from core.social_brand_rule import brand_literals  # noqa: E402
from modules.documents import seed_templates  # noqa: E402
from modules.documents.social_starters import (  # noqa: E402
    SOCIAL_STARTER_SLUGS,
    SOCIAL_VIDEO_STARTER_SLUGS,
    STARTERS_DIR,
    social_starters,
)
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000002b6")
SEED_FILES = sorted(STARTERS_DIR.glob("*.json"))
REFERENCES = _ROOT / "docs" / "PRDS" / "prd251-reference"
# F377 (night 11): the Data story is a business numbers story of its own, no longer the trading reference's port.
REFERENCE_OF = {
    "ui-story-promo": "v1-ui-story.html",
    "cinematic-product-promo": "v2-cinematic-product.html",
    "app-promo": "academy-app-promo.html",
}
# Automatos's own copy: its name, domains and handle, its assistant (Auto), founder and
# licence, its Academy and Markets products, and its tagline.
AUTOMATOS_COPY = re.compile(
    r"automatos|\bgerard\b|\bauto(?:'s| now)\b|\btell auto\b|apache 2\.0|academy coach|market intelligence|operating system for",
    re.IGNORECASE,
)


def _strings(node, where=""):
    """``(where, text)`` for every string in a parsed seed file, keys' values at any depth."""
    if isinstance(node, str):
        yield where, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{where}.{key}" if where else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _strings(value, f"{where}[{i}]")


# ---------------------------------------------------------------------------
# The seed files
# ---------------------------------------------------------------------------


def test_the_net_walks_every_seed_file_and_bites_on_automatos_copy():
    assert [path.stem for path in SEED_FILES] == sorted(SOCIAL_STARTER_SLUGS)
    # What the seeds carried before P251W1-RVW-6, each caught.
    for old in (
        "@automatos.app", "markets.automatos.app", "Ported from the automatos-social quote family.",
        "Automatos Academy", "Academy Coach", "Market Intelligence", "Hey Gerard, what can I do for you today?",
        "Tell Auto what you need.", "Chat: Auto's greeting", "AUTO NOW", "OPEN SOURCE · APACHE 2.0",
        "An operating system for",
    ):
        assert AUTOMATOS_COPY.search(old), old
    for neutral in ("@yourbrand", "yourbrand.com", "Your app", "Chat: the assistant's greeting", "An automated report"):
        assert not AUTOMATOS_COPY.search(neutral), neutral


@pytest.mark.parametrize("path", SEED_FILES, ids=lambda path: path.stem)
def test_no_seed_file_carries_automatos_copy(path):
    seed = json.loads(path.read_text(encoding="utf-8"))
    assert {"name", "description", "sample_data", "variables_schema"} <= set(seed)
    found = [(where, text) for where, text in _strings(seed) if AUTOMATOS_COPY.search(text)]
    assert found == [], found


def test_the_examples_are_brand_neutral_placeholders():
    """Where a starter shows the brand's own handle or address, the example says whose to put."""
    for starter in social_starters():
        sample = starter["sample_data"]
        if "handle" in sample and starter["slug"] != "infographic":
            assert sample["handle"] == "@yourbrand", starter["name"]
        if "end_url" in sample:
            assert sample["end_url"] == "yourbrand.com", starter["name"]
    assert next(s for s in social_starters() if s["slug"] == "app-promo")["sample_data"]["app_name"] == "Your app"


# ---------------------------------------------------------------------------
# What an agent reads
# ---------------------------------------------------------------------------


class _Templates:
    """One workspace's document_templates as ``seed_social_starters`` leaves them: each row
    gets the id a flush gives it, and a lookup answers by the columns it filters on."""

    def __init__(self):
        self.rows = []
        self._where = {}

    def query(self, _model):
        self._where = {}
        return self

    def filter(self, *criteria):
        for criterion in criteria:
            right = criterion.right
            # ``is_active == True`` compares with SQL true(), which binds no value.
            self._where[criterion.left.key] = right.value if isinstance(right, BindParameter) else True
        return self

    def first(self):
        return next((row for row in self.rows if all(getattr(row, k) == v for k, v in self._where.items())), None)

    def add(self, row):
        row.id = row.id or uuid.uuid4()
        self.rows.append(row)

    def commit(self):
        pass


def _dispatch(db, action, params):
    """Run a platform action the way an agent's call runs: through the executor's gates."""
    executor = PlatformActionExecutor(db, WS)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False), patch.object(
        PlatformActionExecutor, "_caller_is_admin", return_value=False
    ), patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params))


def test_an_agent_reading_a_seeded_templates_schema_sees_no_automatos_copy():
    db = _Templates()
    assert seed_templates.seed_social_starters(db, WS) == {"created": len(SOCIAL_STARTER_SLUGS), "refreshed": 0}
    assert len(db.rows) == len(SOCIAL_STARTER_SLUGS) == 19  # the six photo starters and the brand board too
    for row in db.rows:
        schema = _dispatch(db, "platform_get_template_schema", {"template_id": str(row.id)})
        assert schema["success"] is True, (row.name, schema)
        # The answer is the seeded row: its sample data and every variable's label and default.
        assert (schema["id"], schema["name"], schema["description"]) == (str(row.id), row.name, row.description)
        assert schema["sample_data"] == row.sample_data and schema["sample_data"]
        assert schema["variables_schema"] == row.blocks["variables_schema"]
        answer = json.dumps(schema, ensure_ascii=False)
        assert not AUTOMATOS_COPY.search(answer), (row.name, AUTOMATOS_COPY.findall(answer))


# ---------------------------------------------------------------------------
# The CI previews keep the references' own copy (Goal 8)
# ---------------------------------------------------------------------------


def _driver():
    path = _ROOT / "scripts" / "ci" / "social_template_previews.py"
    spec = importlib.util.spec_from_file_location("social_template_previews", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_videos_ci_preview_reads_as_its_reference_and_the_seeded_row_never_does():
    driver = _driver()
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    assert sorted(driver.REFERENCE_COPY) == sorted(REFERENCE_OF) and set(REFERENCE_OF) < set(SOCIAL_VIDEO_STARTER_SLUGS)
    for starter in (s for s in social_starters("social_video") if s["slug"] in REFERENCE_OF):
        overlay, schema = driver.REFERENCE_COPY[starter["slug"]], starter["blocks"]["variables_schema"]
        assert overlay and set(overlay) <= set(schema), starter["name"]
        bundle = driver.bundle_for(starter, kit, starter["preview"]["at"])
        assert {name: bundle["variables"][name] for name in overlay} == overlay, starter["name"]
        seeded = {name: starter["sample_data"].get(name, schema[name].get("default")) for name in overlay}
        assert all(seeded[name] != value for name, value in overlay.items()), seeded
        assert not AUTOMATOS_COPY.search(json.dumps(seeded, ensure_ascii=False)), seeded
        # It is the reference's own copy: its end card's address is in the reference composition.
        page = html.unescape((REFERENCES / REFERENCE_OF[starter["slug"]]).read_text(encoding="utf-8"))
        assert overlay["end_url"] in page, starter["name"]
    app = next(s for s in social_starters("social_video") if s["slug"] == "app-promo")
    spoken = [text for _id, text in voice_script(driver.bundle_for(app, kit, app["preview"]["at"]))]
    assert "Meet Automatos Academy. Train your AI mind." in spoken


def test_an_image_preview_renders_the_seeded_neutral_copy():
    driver = _driver()
    kit = {**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])}
    title = next(s for s in social_starters("social_image") if s["slug"] == "title-card")
    bundle = driver.image_bundle_for(title, kit, title["blocks"]["sizes"][0])
    assert bundle["variables"]["handle"] == "@yourbrand"


# ---------------------------------------------------------------------------
# What the brand rule passes (b)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "composition, css",
    [
        pytest.param("", ".s { box-shadow: 0 12px 40px color-mix(in srgb, black 45%, transparent); }", id="a-shadow-mixed-from-black"),
        pytest.param("", ".d { background: color-mix(in srgb, var(--brand-primary) 78%, black); }", id="a-brand-shade"),
        pytest.param(
            '<script>const ACCENT = getComputedStyle(document.documentElement).getPropertyValue("--brand-primary-rgb").trim();'
            ' tl.to("#glow", { backgroundColor: `rgba(${ACCENT}, 0.3)`, boxShadow: `0 0 40px rgba(${ACCENT}, 0.3)` }, 0);</script>',
            "", id="computed-from-a-brand-variable",
        ),
        pytest.param('<svg><path fill="currentColor" stroke="currentColor"/></svg>', ".i { fill: currentColor; }", id="currentColor"),
        pytest.param('<svg><rect fill="none"/></svg>', ".t { background: transparent; border-color: transparent; }", id="transparent"),
        pytest.param("", ".f { mask-image: linear-gradient(to bottom, transparent 0%, black 40%, black 100%); }", id="an-alpha-mask"),
        pytest.param("", ".c { color: rgb(var(--brand-rgb) / 0.5); background: hsl(var(--hue) 80% 50%); }", id="channels-from-a-variable"),
        pytest.param('<div data-composition-id="main" data-tone="red"></div>', ".a { animation: pulse 2s; }", id="a-word-that-paints-nothing"),
    ],
)
def test_the_brand_rule_passes_a_colour_no_brand_kit_would_set(composition, css):
    assert brand_literals(composition, css) == []
