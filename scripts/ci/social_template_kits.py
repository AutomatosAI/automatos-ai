"""PRD-255 US-006: every seeded social template renders with the two night kits.

``social_template_previews.py`` (the driver) loads this module by path and hands
itself in, so this reuses its bundle builders, renderer and PNG reader; standard
library only, like the driver. For each kit in the driver's ``NIGHT_KITS``,
render-ready with a drawn logo:

* every video's preview (its key moments) and every image (Harbourline at every
  size it declares, Automatos at its first) pass ``hyperframes check`` with 0
  errors (contrast included) and return their outputs, kept under
  ``kit-<name>/`` in the job's artifact;
* each bundle is checked before it is sent. Automatos is a v1 kit with no logo for
  dark backgrounds: its dark stages set the logo on a light chip (FR-9).
  Harbourline is a v2 kit: its paper token is its stored paper, its type-scale
  tokens are its sizes as ratios (at the social bound: the worst case a kit can
  reach), and its dark stages show its uploaded dark logo
  (a drawn one here: a variant is never generated);
* on Harbourline's Title card the page, read back from the PNG, is its stored paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from core.brand_type import BODY_STEP, DEFAULT_TYPE_SCALE, DISPLAY_STEP
from core.media_render_bundle import (
    BODY_SCALE_TOKEN,
    DISPLAY_SCALE_TOKEN,
    LOGO_CHIP_CLEAR,
    LOGO_CHIP_TOKEN,
    MAX_SOCIAL_TYPE_SCALE,
    MIN_SOCIAL_TYPE_SCALE,
    VAR_BRAND_LOGO_ON_DARK,
)
from core.social_templates import SOCIAL_IMAGE, SOCIAL_VIDEO

HARBOURLINE = "harbourline"
# Harbourline carries the larger type scale, so its layouts are checked at every size.
EVERY_SIZE = frozenset({HARBOURLINE})
DARK_LOGO_PATH = "assets/brand/logo-dark.png"
DARK_LOGO_VARIABLE = "{{ brand.logo_on_dark }}"
TITLE_CARD = "title-card"
# The Title card's left margin, half way down: nothing there but the page.
PAGE_PROBE = (0.02, 0.5)
PAPER_TOKEN = "paper"


def night_kit(driver: Any, name: str) -> Dict[str, Any]:
    """The night kit ``name`` as a workspace has it render-ready: a drawn logo and, for a v2 kit, a drawn dark logo."""
    kit = dict(driver.NIGHT_KITS[name])
    kit["logo_url"] = driver.logo_png(kit["primary_color"])
    paper = (kit.get("palette") or {}).get(PAPER_TOKEN)
    if paper:
        kit["logo_dark_url"] = driver.logo_png(paper)
    return kit


def night_bundles(driver: Any, name: str, kit: Mapping[str, Any]) -> Iterator[Tuple[Mapping[str, Any], Optional[str], Dict[str, Any]]]:
    """``(starter, size, bundle)`` for every seeded template: a video's preview, an image at its sizes."""
    for starter in driver.social_starters(SOCIAL_VIDEO):
        yield starter, None, driver.bundle_for(starter, kit, starter["preview"]["at"])
    for starter in driver.social_starters(SOCIAL_IMAGE):
        sizes = starter["blocks"]["sizes"]
        for size in sizes if name in EVERY_SIZE else sizes[:1]:
            yield starter, size, driver.image_bundle_for(starter, kit, size)


def _ratio(kit: Mapping[str, Any], step: str) -> str:
    """The kit's ``step`` size over the default, within the social bounds, as the token carries it."""
    ratio = kit["type_scale"][step]["size_pt"] / DEFAULT_TYPE_SCALE[step][0]
    return f"{min(max(ratio, MIN_SOCIAL_TYPE_SCALE), MAX_SOCIAL_TYPE_SCALE):g}"


def bundle_findings(kit: Mapping[str, Any], starter: Mapping[str, Any], bundle: Mapping[str, Any]) -> List[str]:
    """What the bundle gets wrong about the kit's v2 rules and its logo on a dark stage; ``[]`` when nothing."""
    tokens, variables = bundle["brand"]["tokens"], bundle["variables"]
    dark_stage = DARK_LOGO_VARIABLE in starter["blocks"]["html"]
    stored = kit.get("palette") or {}
    if not kit.get("logo_dark_url"):
        light = tokens.get(LOGO_CHIP_TOKEN) == tokens.get(PAPER_TOKEN)
        return [] if light else [f"no dark logo, yet the logo chip is {tokens.get(LOGO_CHIP_TOKEN)}, not the paper"]
    wanted = {
        PAPER_TOKEN: stored[PAPER_TOKEN].lower(),
        DISPLAY_SCALE_TOKEN: _ratio(kit, DISPLAY_STEP),
        BODY_SCALE_TOKEN: _ratio(kit, BODY_STEP),
        LOGO_CHIP_TOKEN: LOGO_CHIP_CLEAR,
    }
    findings = [f"token {name} is {tokens.get(name)}, not {value}" for name, value in wanted.items() if tokens.get(name) != value]
    if dark_stage and variables.get(VAR_BRAND_LOGO_ON_DARK) != DARK_LOGO_PATH:
        findings.append(f"the dark stage shows {variables.get(VAR_BRAND_LOGO_ON_DARK)!r}, not the kit's dark logo")
    return findings


def check_page(driver: Any, kit: Mapping[str, Any], data: bytes) -> None:
    """The Title card's page is the kit's stored paper, read back from the PNG."""
    got = driver.sample(data, *PAGE_PROBE)
    want = driver.hex_rgb(kit["palette"][PAPER_TOKEN])
    if any(abs(a - b) > driver.PROBE_TOKEN_TOLERANCE for a, b in zip(got, want)):
        raise driver.PreviewFailure(f"the page is {got}, not the kit's stored paper {want}")
    print(f"      PASS: the page is the kit's stored paper {kit['palette'][PAPER_TOKEN]}")


def render_one(driver: Any, renderer: Any, kit: Mapping[str, Any], starter: Mapping[str, Any],
               bundle: Mapping[str, Any], folder: Path) -> Dict[str, Any]:
    """Check the bundle, render it (0 check errors), keep its outputs; the job's check."""
    findings = bundle_findings(kit, starter, bundle)
    if findings:
        raise driver.PreviewFailure("; ".join(findings))
    job = renderer.render(bundle)
    print(f"    {driver._check_summary(job['report'])}")
    if (job["report"].get("check") or {}).get("errors"):
        raise driver.PreviewFailure("the check passed the job with errors")
    folder.mkdir(parents=True, exist_ok=True)
    for output in job["outputs"]:
        (folder / output["name"]).write_bytes(output.pop("data"))
    if kit.get("palette") and starter["slug"] == TITLE_CARD:
        check_page(driver, kit, (folder / job["outputs"][0]["name"]).read_bytes())
    return {"check": job["report"].get("check"), "outputs": job["outputs"]}


def run_night_kits(driver: Any, renderer: Any, out: Path, report: Dict[str, Any]) -> List[str]:
    """Every seeded template with each night kit: the failures, one line each."""
    failures: List[str] = []
    for name in driver.NIGHT_KITS:
        kit = night_kit(driver, name)
        entry: Dict[str, Any] = report.setdefault("night_kits", {}).setdefault(name, {})
        print(f"\n{name} kit (PRD-255 US-006): every seeded template again")
        for starter, size, bundle in night_bundles(driver, name, kit):
            label = starter["name"] + (f" at {size}" if size else " (preview)")
            print(f"\n== {name} kit: {label}")
            try:
                folder = out / f"kit-{name}" / starter["slug"] / (size or "preview")
                entry[f"{starter['slug']} {size or 'preview'}"] = render_one(driver, renderer, kit, starter, bundle, folder)
            except driver.PreviewFailure as exc:
                print(f"    FAIL: {exc}")
                failures.append(f"{name} kit, {label}: {exc}")
    return failures


__all__ = ["bundle_findings", "night_bundles", "night_kit", "run_night_kits"]
