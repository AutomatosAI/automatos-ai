"""Brand kit from your website: ``platform_web_fetch`` with ``brand: true`` reads what a site shows of its brand.

The text a fetch returns drops ``<style>``, ``<svg>``, the header and the footer, so the Brand designer could
never see a site's colours, fonts, logo or social links. Locked here:

* the evidence (``site_brand``): brand colours ahead of neutrals, each with where it was found and its
  contrast; ``var()`` resolved; transparent tints left out; fonts by role, ``@font-face`` and Google Fonts;
  the header's logo ahead of ``og:image``; marks largest first; names, lines and social profiles, share and
  intent links left out; notes for an inline SVG logo and a page drawn by JavaScript;
* the fetch: stylesheets go through the same outbound check and pinning as the page, one that redirects
  into a private range is never requested and is named as not read, and ``brand`` off changes nothing.

Pure: stubbed HTTP and DNS, no database, no network.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434): an earlier test can leave a stale,
# spec-less `modules`/`services` entry in sys.modules that breaks these imports.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from typing import List  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
from uuid import uuid4  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from config import config  # noqa: E402
import core.security.web_access as wa  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_web import register_web_actions  # noqa: E402
import modules.tools.discovery.handlers_web as hw  # noqa: E402
from modules.tools.discovery.site_brand import colours_in, site_brand  # noqa: E402

WS = uuid4()
SITE = "https://harbourline.example"
PUBLIC_IP = "93.184.216.34"
PRIVATE = {"internal.host": "10.0.0.9"}

PAGE = """<html><head><title>Harbourline Coffee | Roasters</title>
<meta property="og:site_name" content="Harbourline">
<meta name="description" content="Small-batch coffee from the harbour.">
<meta name="theme-color" content="#1E3A5F">
<meta property="og:image" content="/og-banner.jpg">
<link rel="stylesheet" href="/assets/site.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:wght@400;600&family=Inter:wght@400">
<link rel="icon" sizes="32x32" href="/favicon-32.png">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<style>:root{--brand-primary:#1E3A5F;--accent:#C26A2E;--grey:#f4f4f4;--font-size-base:16px}</style>
</head><body>
<header><a class="site-logo" href="/"><img src="/img/harbourline-logo.svg" alt="Harbourline"></a></header>
<h1>Coffee roasted by the sea</h1>
<p>We roast small batches every morning and ship them the same day, from the harbour to your door.</p>
<p>Single origins, honest blends and a subscription that fits how much you actually drink.</p>
<footer>
<a href="https://www.instagram.com/harbourline/">Instagram</a>
<a href="https://www.linkedin.com/company/harbourline-coffee">LinkedIn</a>
<a href="https://twitter.com/intent/tweet?text=hello">Share</a>
<a href="https://x.com/harbourline">X</a>
<a href="https://www.facebook.com/sharer/sharer.php?u=x">Share</a>
</footer></body></html>"""

CSS = """body{color:#1a1714;background:#fbf8f3;font-family:"Inter",system-ui,sans-serif}
h1,h2{font-family:'Newsreader',Georgia,serif;color:var(--brand-primary)}
.btn-primary{background-color:var(--accent);color:#fff}
a:hover{color:rgba(194,106,46,0.9)}
.shadow{box-shadow:0 0 0 rgba(0,0,0,.1);border-color:#00000010}
@media (min-width:600px){.card{background:hsl(210, 52%, 24%)}}
@font-face{font-family:"Harbour Display";src:url(/f.woff2)}"""


def _brand(html: str = PAGE, css: str = CSS):
    return site_brand(html, SITE + "/", {SITE + "/assets/site.css": css})


def _colour(brand, hex_value):
    return next(c for c in brand["colours"] if c["hex"] == hex_value)


# ── the evidence ───────────────────────────────────────────────────────────────────────────────

def test_brand_colours_come_first_each_naming_where_it_was_found():
    brand = _brand()
    first_two = [c["hex"] for c in brand["colours"][:2]]
    assert set(first_two) == {"#1e3a5f", "#c26a2e"}
    navy, copper = _colour(brand, "#1e3a5f"), _colour(brand, "#c26a2e")
    assert '<meta name="theme-color">' in navy["found_in"]
    assert "CSS variable --brand-primary" in navy["found_in"]
    assert "headings: color" in navy["found_in"]  # resolved through var(--brand-primary)
    assert "CSS variable --accent" in copper["found_in"] and "buttons: background-color" in copper["found_in"]
    assert "links: color" in copper["found_in"]  # rgba(194,106,46,0.9) is the same colour, solid enough
    assert navy["neutral"] is False and copper["neutral"] is False


def test_neutrals_rank_after_the_brand_and_tints_are_left_out():
    brand = _brand()
    hexes = [c["hex"] for c in brand["colours"]]
    for neutral in ("#1a1714", "#fbf8f3", "#ffffff", "#f4f4f4"):
        assert neutral in hexes and _colour(brand, neutral)["neutral"] is True
    last_brand = max(i for i, c in enumerate(brand["colours"]) if not c["neutral"])
    first_neutral = min(i for i, c in enumerate(brand["colours"]) if c["neutral"])
    assert last_brand < first_neutral
    assert "#000000" not in hexes  # rgba(0,0,0,.1) and #00000010 are shadows, not colours


def test_each_colour_carries_its_contrast_for_the_kit_check():
    navy = _colour(_brand(), "#1e3a5f")
    assert navy["contrast_on_white"] > 4.5  # usable as text on the page as it is
    assert round(navy["contrast_on_white"] * navy["contrast_on_black"], 0) == 21  # white:black is 21:1


def test_colour_syntaxes_are_read():
    assert colours_in("#abc") == [(170, 187, 204)]
    assert colours_in("rgb(255 0 0)") == [(255, 0, 0)]
    assert colours_in("rgba(0, 128, 0, 30%)") == []  # mostly transparent
    assert colours_in("hsl(0, 100%, 50%)") == [(255, 0, 0)]
    assert colours_in("linear-gradient(#000 0%, #fff 100%)") == [(0, 0, 0), (255, 255, 255)]


def test_fonts_by_role_with_the_ones_the_site_serves():
    fonts = {f["family"]: f["found_in"] for f in _brand()["fonts"]}
    assert "body: font-family" in fonts["Inter"] and "Google Fonts link" in fonts["Inter"]
    assert "headings: font-family" in fonts["Newsreader"] and "Google Fonts link" in fonts["Newsreader"]
    assert fonts["Harbour Display"] == ["@font-face (the site serves this font)"]
    assert not any(name.startswith("16") for name in fonts)  # --font-size-base is a size, not a family
    assert "Georgia" not in fonts and "system-ui" not in fonts  # fallbacks, not the brand's choice


def test_the_header_logo_comes_first_and_og_image_last():
    logos = _brand()["logos"]
    assert logos[0] == {"url": SITE + "/img/harbourline-logo.svg", "alt": "Harbourline",
                        "found_in": "header <img> marked logo", "format": "svg"}
    assert logos[-1]["url"] == SITE + "/og-banner.jpg" and logos[-1]["found_in"].startswith("og:image")


def test_marks_largest_first():
    marks = _brand()["marks"]
    assert marks[0]["url"] == SITE + "/apple-touch-icon.png" and marks[0]["size"] == 180
    assert marks[1]["url"] == SITE + "/favicon-32.png"


def test_names_lines_and_social_profiles():
    brand = _brand()
    assert brand["names"][0] == {"value": "Harbourline", "found_in": "og:site_name"}
    assert brand["taglines"][0] == {"value": "Small-batch coffee from the harbour.", "found_in": "meta description"}
    social = brand["social"]
    assert social["instagram"]["handle"] == "harbourline"
    assert social["linkedin"]["handle"] == "harbourline-coffee"
    assert social["twitter"]["handle"] == "harbourline"  # the intent/share link before it is not a profile
    assert "facebook" not in social  # only a sharer link
    assert brand["notes"] == []


def test_an_inline_svg_logo_says_a_person_uploads_it():
    html = PAGE.replace('<img src="/img/harbourline-logo.svg" alt="Harbourline">', "<svg><path d='M0 0'/></svg>")
    brand = _brand(html)
    assert any("inline as SVG" in note for note in brand["notes"])
    assert all("harbourline-logo.svg" not in logo["url"] for logo in brand["logos"])


def test_a_page_drawn_by_javascript_says_so():
    html = "<html><head><script src='/app.js'></script></head><body><div id='root'></div></body></html>"
    assert any("JavaScript" in note for note in site_brand(html, SITE, {})["notes"])


def _hexes_in(css: str):
    return {c["hex"]: c["found_in"] for c in site_brand("<html><body></body></html>", SITE, {SITE + "/a.css": css})["colours"]}


def test_a_dark_theme_never_replaces_the_brands_own_colour():
    """A site declares its default first and overrides it for dark mode later; the default is the brand."""
    css = (":root{--primary:#c26a2e}\n"
           "@media (prefers-color-scheme: dark){:root{--primary:#123456} .btn{background:#123456}}\n"
           ".dark{--primary:#654321} [data-theme=\"dark\"] .btn{background:#654321}\n"
           "@media (min-width: 900px){:root{--primary:#abcdef}}\n"
           ".btn{background:var(--primary)} .btn-dark{background:#0a7d5a}")
    colours = _hexes_in(css)
    assert "buttons: background" in colours["#c26a2e"] and "CSS variable --primary" in colours["#c26a2e"]
    assert "#123456" not in colours and "#654321" not in colours  # dark-theme overrides
    assert "#0a7d5a" in colours  # .btn-dark is a button style, not a dark theme


def test_a_commented_out_rule_is_not_read():
    colours = _hexes_in("/* .old { color: #ff0000; } */ .btn{background:#c26a2e}")
    assert "#ff0000" not in colours and "buttons: background" in colours["#c26a2e"]


# ── the fetch ──────────────────────────────────────────────────────────────────────────────────

def _fake_getaddrinfo(host, port, proto=None):
    return [(2, 1, 6, "", (PRIVATE.get(host, PUBLIC_IP), port))]


@pytest.fixture
def web_on(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", True)
    monkeypatch.setattr(config, "WEB_ACCESS_DENY", ())
    monkeypatch.setattr(wa, "_getaddrinfo", _fake_getaddrinfo)


def _stub_http(monkeypatch, handler) -> List[httpx.Request]:
    sent: List[httpx.Request] = []

    def recording(req: httpx.Request):
        sent.append(req)
        return handler(req)

    transport = httpx.MockTransport(recording)

    def factory(**kwargs):
        kwargs.pop("verify", None)
        return httpx.AsyncClient(transport=transport, **kwargs)

    monkeypatch.setattr(hw, "_async_client", factory)
    return sent


def _site(req: httpx.Request) -> httpx.Response:
    if req.url.path == "/assets/site.css":
        return httpx.Response(200, headers={"content-type": "text/css"}, text=CSS)
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE)


@pytest.mark.asyncio
async def test_brand_true_reads_the_stylesheets_through_the_same_checks(monkeypatch, web_on):
    sent = _stub_http(monkeypatch, _site)
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/", "brand": True})
    assert out["success"] is True
    brand = out["data"]["brand"]
    assert brand["stylesheets_read"] == [SITE + "/assets/site.css"]
    assert _colour(brand, "#c26a2e")["found_in"][:2] == ["CSS variable --accent", "buttons: background-color"]
    css_request = next(r for r in sent if r.url.path == "/assets/site.css")
    assert css_request.url.host == PUBLIC_IP and css_request.headers["host"] == "harbourline.example"
    assert all(r.url.host != "fonts.googleapis.com" for r in sent)  # a font link is read, never fetched
    assert "Coffee roasted by the sea" in out["data"]["content"]  # the text answer is unchanged


@pytest.mark.asyncio
async def test_a_stylesheet_that_redirects_inward_is_never_requested(monkeypatch, web_on):
    def handler(req: httpx.Request):
        if req.url.path == "/assets/site.css":
            return httpx.Response(302, headers={"location": "http://internal.host/theme.css"})
        return _site(req)

    sent = _stub_http(monkeypatch, handler)
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/", "brand": True})
    brand = out["data"]["brand"]
    assert brand["stylesheets_read"] == []
    assert any("not read" in note and "site.css" in note for note in brand["notes"])
    assert all(r.headers.get("host") != "internal.host" for r in sent)
    assert _colour(brand, "#1e3a5f")["found_in"][0] == '<meta name="theme-color">'  # the page itself still counts


@pytest.mark.asyncio
async def test_a_slow_stylesheet_is_cut_off_at_the_deadline(monkeypatch, web_on):
    """The page chooses its stylesheets' addresses, so the brand read is bounded in time as well as in count
    and size: a sheet still loading at the deadline is cancelled and named as not read."""
    import asyncio

    async def handler(req: httpx.Request):
        if req.url.path == "/assets/site.css":
            await asyncio.sleep(5)
        return _site(req)

    _stub_http(monkeypatch, handler)
    monkeypatch.setattr(hw, "BRAND_DEADLINE_SECONDS", 0.05)
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/", "brand": True})
    brand = out["data"]["brand"]
    assert brand["stylesheets_read"] == []
    assert any("not read" in note for note in brand["notes"])
    assert brand["colours"][0]["hex"] == "#1e3a5f"  # the page's own colours still count


@pytest.mark.asyncio
async def test_without_brand_the_answer_is_as_before(monkeypatch, web_on):
    sent = _stub_http(monkeypatch, _site)
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/"})
    assert "brand" not in out["data"]
    assert [r.url.path for r in sent] == ["/"]  # no stylesheet fetched


@pytest.mark.asyncio
async def test_brand_on_something_that_is_not_a_page(monkeypatch, web_on):
    _stub_http(monkeypatch, lambda req: httpx.Response(200, headers={"content-type": "application/json"}, text="{}"))
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/api", "brand": True})
    assert out["data"]["brand"] == {"notes": [hw.NOT_A_PAGE_NOTE]}


@pytest.mark.asyncio
async def test_brand_with_web_access_off_is_the_same_honest_answer(monkeypatch):
    monkeypatch.setattr(config, "WEB_ACCESS", False)
    out = await hw.web_fetch(MagicMock(), WS, {"url": SITE + "/", "brand": True})
    assert out["success"] is True and out["data"]["available"] is False


def test_the_brand_option_is_in_the_schema_and_never_required():
    reg = ActionRegistry()
    register_web_actions(reg)
    fetch = reg._actions["platform_web_fetch"]
    assert fetch.parameters["properties"]["brand"]["type"] == "boolean"
    assert fetch.parameters["required"] == ["url"]
    assert "brand: true" in fetch.description


# ── the Playbook files the designer's ticket ───────────────────────────────────────────────────

def _brand_kit_playbook():
    from core.seeds.seed_socials_package import BRAND_KIT_PLAYBOOK_TEMPLATE_ID, SOCIALS_PLAYBOOKS

    return next(spec for spec in SOCIALS_PLAYBOOKS if spec["template_id"] == BRAND_KIT_PLAYBOOK_TEMPLATE_ID)


def test_the_playbook_files_a_ticket_and_never_saves_the_kit_itself():
    """A step's call carries no owner and no card: the admin-only save is refused and a proposal has nowhere
    to go. FR-11: the owner approves every change, on the designer's own ticket."""
    from core.seeds.seed_socials_package import BRAND_DESIGNER_NAME

    (step,) = _brand_kit_playbook()["steps"]
    prompt = step["prompt_template"]
    assert "platform_update_brand_kit" not in prompt
    assert f'platform_create_task with assigned_agent_name "{BRAND_DESIGNER_NAME}"' in prompt
    assert "platform_update_task_status to 'in_progress'" in prompt
    assert "brand: true" in prompt and "only after the owner approves" in prompt
    assert "{input.website_url}" in prompt  # the f-string kept the Playbook's own placeholder


def test_an_unedited_copy_of_the_old_playbook_takes_the_new_prompt():
    from core.seeds.seed_socials_package import _BRAND_KIT_PROMPT_251, refreshed_steps

    spec = _brand_kit_playbook()
    old = [{**spec["steps"][0], "prompt_template": _BRAND_KIT_PROMPT_251}]
    (step,) = refreshed_steps(spec, old)
    assert step["prompt_template"] == spec["steps"][0]["prompt_template"]
    curated = [{**spec["steps"][0], "prompt_template": "The owner's own words."}]
    assert refreshed_steps(spec, curated) is None  # an owner's edit is never overwritten


def test_the_designer_reads_the_site_with_brand_true():
    from core.seeds.seed_socials_package import BRAND_DESIGNER, SOCIALS_AGENTS

    persona = next(spec for spec in SOCIALS_AGENTS if spec["slug"] == BRAND_DESIGNER)["custom_persona_prompt"]
    assert "platform_web_fetch with brand: true" in persona
    assert "Derive everything from the logo" in persona  # the logo still leads
