"""PRD-251D US-D301: every seeded social template renders in every look it is offered in.

``social_template_previews.py`` (the driver) loads this module by path and hands itself in, so
this reuses its bundle builders and its renderer; standard library only, like the driver. The
look is the orchestrator's own (``modules/socials/post_look.py``), applied as a post's render
applies it (``modules/socials/render.bundle_for``): the look's accent on a copy of the kit before
the bundle is built, then its light or dark and its logo on the bundle's tokens and variables.

For each starter, every look its gallery entry offers (``post_look.looks_of``): light and dark
for a template built on paper, the accent strong and sparing, the logo on and off. That is seven
looks for each of the nine paper cards and three for each of the other ten, beside the look the
driver already renders (the template as authored with the driver's kit: light, its accent, its
logo), which is not rendered twice. An image renders at its first size and a video its preview,
with the driver's kit. Each bundle must carry its look (:func:`look_findings`) and each render
must pass ``hyperframes check`` with 0 errors, its contrast pass included. The outputs are kept
under ``look-<slug>/<look>/`` in the job's artifact.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

from core.brand_palette import INK, PAPER
from core.media_render_bundle import NO_LOGO, VAR_BRAND_NAME
from core.social_kit_tokens import ACCENT_BAR_SPAN_TOKEN, ACCENT_BOLD, ACCENT_USE_FIELD, BAR_FULL, BAR_RULE
from core.social_templates import SOCIAL_VIDEO
from modules.socials import post_look

ACCENTS = post_look.ACCENTS
LOGOS = (True, False)
NO_LOGO_LABEL = "no-logo"
LOGO_LABEL = "logo"


def looks_offered(starter: Mapping[str, Any], kit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every look the starter's gallery entry offers, but the one the driver renders already."""
    offered = post_look.looks_of(starter["blocks"])
    modes = post_look.MODES if post_look.LOOK_MODE in offered else (None,)
    own_accent = ACCENT_BOLD if kit.get(ACCENT_USE_FIELD) == ACCENT_BOLD else post_look.ACCENT_SPARING
    looks = []
    for mode, accent, logo in product(modes, ACCENTS, LOGOS):
        if mode in (None, post_look.LIGHT) and accent == own_accent and logo:
            continue
        mode_choice = {post_look.LOOK_MODE: mode} if mode else {}
        looks.append({**mode_choice, post_look.LOOK_ACCENT: accent, post_look.LOOK_LOGO: logo})
    return looks


def look_label(look: Mapping[str, Any]) -> str:
    """``dark-bold-no-logo``: the look's choices, as a folder and a report key."""
    mode = [look[post_look.LOOK_MODE]] if post_look.LOOK_MODE in look else []
    return "-".join([*mode, look[post_look.LOOK_ACCENT], LOGO_LABEL if look[post_look.LOOK_LOGO] else NO_LOGO_LABEL])


def look_units(driver: Any, kit: Mapping[str, Any]) -> Iterator[Tuple[Mapping[str, Any], Dict[str, Any]]]:
    """``(starter, look)`` for every seeded template and every look it is offered in, in one fixed order."""
    for starter in driver.social_starters():
        for look in looks_offered(starter, kit):
            yield starter, look


def look_bundle(driver: Any, starter: Mapping[str, Any], kit: Mapping[str, Any], look: Mapping[str, Any]) -> Dict[str, Any]:
    """The starter's bundle in ``look``, as a post's render builds it: a video's preview, an image at its first size."""
    looked_kit = post_look.kit_with_look(kit, look)
    if starter["format"] == SOCIAL_VIDEO:
        bundle = driver.bundle_for(starter, looked_kit, starter["preview"]["at"])
    else:
        bundle = driver.image_bundle_for(starter, looked_kit, starter["blocks"]["sizes"][0])
    return post_look.with_look(bundle, starter["blocks"], look, kit)


def look_findings(plain: Mapping[str, Any], looked: Mapping[str, Any], look: Mapping[str, Any]) -> List[str]:
    """What the looked bundle gets wrong about its look, against the plain one; ``[]`` when nothing."""
    before, after = plain["brand"]["tokens"], looked["brand"]["tokens"]
    findings = []
    if look.get(post_look.LOOK_MODE) == post_look.DARK and (after.get(PAPER), after.get(INK)) != (before.get(INK), before.get(PAPER)):
        findings.append(f"dark, yet the page is {after.get(PAPER)}, not the stage's ink {before.get(INK)}")
    span = BAR_FULL if look[post_look.LOOK_ACCENT] == ACCENT_BOLD else BAR_RULE
    if after.get(ACCENT_BAR_SPAN_TOKEN) != span:
        findings.append(f"{look[post_look.LOOK_ACCENT]}, yet the brand bar's span is {after.get(ACCENT_BAR_SPAN_TOKEN)}")
    logos = [looked["variables"].get(name) for name in post_look.LOGO_VARIABLES]
    if not look[post_look.LOOK_LOGO] and any(value != NO_LOGO for value in logos):
        findings.append("the logo is off, yet a logo variable still names a logo")
    if looked["variables"].get(VAR_BRAND_NAME) != plain["variables"].get(VAR_BRAND_NAME):
        findings.append("the brand name changed with the look")
    return findings


def render_look(driver: Any, renderer: Any, kit: Mapping[str, Any], starter: Mapping[str, Any],
                look: Mapping[str, Any], folder: Path) -> Dict[str, Any]:
    """Check the looked bundle, render it (0 check errors, its contrast pass included), keep its outputs."""
    looked = look_bundle(driver, starter, kit, look)
    findings = look_findings(look_bundle(driver, starter, kit, {}), looked, look)
    if findings:
        raise driver.PreviewFailure("; ".join(findings))
    job = renderer.render(looked)
    print(f"    {driver._check_summary(job['report'])}")
    if (job["report"].get("check") or {}).get("errors"):
        raise driver.PreviewFailure("the check passed the job with errors")
    if not job["outputs"]:
        raise driver.PreviewFailure("the render returned no file")
    folder.mkdir(parents=True, exist_ok=True)
    for output in job["outputs"]:
        (folder / output["name"]).write_bytes(output.pop("data"))
    return {"check": job["report"].get("check"), "outputs": job["outputs"]}


def run_looks(driver: Any, renderer: Any, out: Path, kit: Mapping[str, Any], report: Dict[str, Any]) -> List[str]:
    """Every seeded template in every look it is offered in: the failures, one line each."""
    failures: List[str] = []
    entry: Dict[str, Any] = report.setdefault("looks", {})
    print("\nPRD-251D US-D301: every seeded template in every look it is offered in")
    for starter, look in driver.SHARD.mine(look_units(driver, kit)):
        label = look_label(look)
        print(f"\n== {starter['name']} in the look {label}")
        try:
            folder = out / f"look-{starter['slug']}" / label
            entry.setdefault(starter["slug"], {})[label] = render_look(driver, renderer, kit, starter, look, folder)
        except driver.PreviewFailure as exc:
            print(f"    FAIL: {exc}")
            failures.append(f"{starter['name']} in the look {label}: {exc}")
    return failures


__all__ = ["look_bundle", "look_findings", "look_label", "look_units", "looks_offered", "run_looks"]
