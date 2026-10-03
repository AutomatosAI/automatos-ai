"""PRD-251B US-B104: a video template's shorter cuts.

A video template offers the lengths in ``blocks.durations``. Its own length plays the
timeline as authored. A shorter length is a cut: ``blocks.cuts`` maps the length (whole
seconds, as a string) to the stretches of the authored timeline it keeps, in order, e.g.
``{"15": [[0, 4.0], [12.59, 18.0], [31.85, 37.44]]}``. The kept stretches play back to
back, so they add up to the length (``core.social_templates`` checks the declaration).

:func:`cut_to_length` is the one place a chosen length reaches a composition: the post's
render and preview (``modules/socials/render.py``) and the media-render CI job.

- The root's ``data-duration`` is the length. With a cut, the root also carries the kept
  stretches as ``data-cut``, and :data:`CUT_RUNTIME`, placed before the composition's
  first inline script, moves every tween of its GSAP timelines with them: a tween inside
  a kept stretch moves with it, a tween in a dropped stretch lands as its end state where
  the next kept stretch begins, and nothing runs past the cut's end.
- Each clip's ``data-start``/``data-duration`` follows the stretches it overlaps. A clip
  outside them all stops being a clip and is hidden; its elements stay, because the
  composition's own script builds and reads them. Footage in a clip the cut enters
  part-way plays from its own start.
- Voice lines and SFX keep their place inside a kept stretch and are dropped outside one
  (:func:`cut_audio_plan`); so are snapshot moments (:func:`cut_moments`).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from core.social_templates import with_root_attribute, with_root_duration

Stretch = Tuple[float, float]

CUT_ATTRIBUTE = "data-cut"
HIDDEN_ATTRIBUTE = "data-cut-hidden"
HIDDEN_CSS = f"\n[{HIDDEN_ATTRIBUTE}]{{display:none !important}}\n"
TIMING_ATTRIBUTES = ("data-start", "data-duration", "data-track-index")
TIME_DECIMALS = 3
# A clip must overlap a kept stretch by more than this to stay a clip.
MIN_OVERLAP_SECONDS = 0.001

_START_TAG = re.compile(r"<[a-zA-Z][^<>]*>")
_RAW_TEXT = re.compile(r"<(script|style)\b[^<>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_INLINE_SCRIPT = re.compile(r"<script\b[^<>]*>", re.IGNORECASE)
_CLASS = re.compile(r"""(?<![\w-])class\s*=\s*["']([^"']*)["']""", re.IGNORECASE)

# Loaded before the composition's own script. It wraps gsap.timeline(): each timeline the
# composition makes moves its tweens through the root's data-cut. ``{{`` never appears,
# so media-render's placeholder filling leaves it as it is.
CUT_RUNTIME = """
(function () {
  var g = window.gsap;
  if (!g || g.__cutTimeline) return;
  g.__cutTimeline = true;
  var TIMING = ["duration", "ease", "stagger", "yoyo", "repeat", "repeatDelay", "delay", "immediateRender",
    "runBackwards", "keyframes", "parent", "onStart", "onUpdate", "onComplete", "onRepeat"];
  function stretches() {
    var root = document.querySelector("[data-composition-id]");
    try {
      var cut = JSON.parse((root && root.getAttribute("data-cut")) || "null");
      return Array.isArray(cut) && cut.length ? cut : null;
    } catch (e) {
      return null;
    }
  }
  function place(cut, t) {
    var out = 0;
    for (var i = 0; i < cut.length; i++) {
      if (t < cut[i][0]) return { at: out, kept: false };
      if (t < cut[i][1]) return { at: out + t - cut[i][0], kept: true };
      out += cut[i][1] - cut[i][0];
    }
    return { at: out, kept: false, after: true };
  }
  function endState(vars, fromVars) {
    var source = vars.yoyo && (vars.repeat || 0) % 2 === 1 ? fromVars : vars;
    if (!source) return null;
    var state = {};
    for (var key in source) if (TIMING.indexOf(key) < 0) state[key] = source[key];
    return state;
  }
  function fit(vars, at, total) {
    var runs = 1 + (vars.repeat > 0 ? vars.repeat : 0);
    var duration = typeof vars.duration === "number" ? vars.duration : 0.5;
    if (at + duration * runs <= total) return vars;
    var fitted = {};
    for (var key in vars) fitted[key] = vars[key];
    fitted.duration = Math.max(0.001, (total - at) / runs);
    return fitted;
  }
  var make = g.timeline.bind(g);
  g.timeline = function (vars) {
    var tl = make(vars);
    var cut = stretches();
    if (!cut) return tl;
    var total = cut.reduce(function (sum, s) { return sum + s[1] - s[0]; }, 0);
    var to = tl.to.bind(tl), fromTo = tl.fromTo.bind(tl), from = tl.from.bind(tl);
    var set = tl.set.bind(tl), call = tl.call.bind(tl), add = tl.add.bind(tl);
    function moved(position, kept, dropped) {
      if (typeof position !== "number") return kept(position, false);
      var p = place(cut, position);
      if (p.kept) return kept(p.at, true);
      if (!p.after && dropped) dropped(p.at);
      return tl;
    }
    function settle(target, state, at) {
      if (state) set(target, state, at);
    }
    tl.to = function (target, v, position) {
      if (typeof v !== "object" || v === null) return to.apply(tl, arguments);
      return moved(position, function (at, cutAt) { return to(target, cutAt ? fit(v, at, total) : v, at); },
        function (at) { settle(target, endState(v, null), at); });
    };
    tl.fromTo = function (target, f, v, position) {
      if (typeof f !== "object" || f === null) return fromTo.apply(tl, arguments);
      return moved(position, function (at, cutAt) { return fromTo(target, f, cutAt ? fit(v, at, total) : v, at); },
        function (at) { settle(target, endState(v, f), at); });
    };
    tl.from = function (target, v, position) {
      if (typeof v !== "object" || v === null) return from.apply(tl, arguments);
      return moved(position, function (at, cutAt) { return from(target, cutAt ? fit(v, at, total) : v, at); }, null);
    };
    tl.set = function (target, v, position) {
      return moved(position, function (at) { return set(target, v, at); },
        function (at) { settle(target, endState(v, null), at); });
    };
    tl.call = function (fn, params, position) {
      return moved(position, function (at) { return call(fn, params, at); }, function (at) { call(fn, params, at); });
    };
    tl.add = function (child, position) {
      return moved(position, function (at) { return add(child, at); }, null);
    };
    return tl;
  };
})();
"""


def stretches_for(blocks: Mapping[str, Any], seconds: int) -> Optional[List[Stretch]]:
    """The stretches the template keeps at ``seconds``, or ``None`` when that length plays the timeline as authored."""
    cuts = blocks.get("cuts") if isinstance(blocks, Mapping) else None
    declared = cuts.get(str(int(seconds))) if isinstance(cuts, Mapping) else None
    if not declared:
        return None
    return [(float(start), float(end)) for start, end in declared]


def place(second: float, stretches: Sequence[Stretch]) -> Tuple[float, bool]:
    """Where source ``second`` lands in the cut, and whether a kept stretch holds it.

    A second the cut drops lands where the next kept stretch begins, or at the cut's end.
    """
    out = 0.0
    for start, end in stretches:
        if second < start:
            return out, False
        if second < end:
            return out + second - start, True
        out += end - start
    return out, False


def _overlaps(start: float, end: float, stretches: Sequence[Stretch]) -> List[Stretch]:
    """The parts of the source span [start, end) the cut keeps, in output seconds."""
    spans: List[Stretch] = []
    out = 0.0
    for kept_start, kept_end in stretches:
        low, high = max(start, kept_start), min(end, kept_end)
        if high - low > MIN_OVERLAP_SECONDS:
            spans.append((out + low - kept_start, out + high - kept_start))
        out += kept_end - kept_start
    return spans


def _number(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _attribute(tag: str, name: str) -> Optional[str]:
    found = re.search(rf"""(?<![\w-]){re.escape(name)}\s*=\s*["']([^"']*)["']""", tag, re.IGNORECASE)
    return found.group(1) if found else None


def _appended(tag: str, attribute: str) -> str:
    """``tag`` with ``attribute`` (``name`` or ``name="value"``) added at its end, a self-closing tag kept so."""
    body = tag[:-1].rstrip()
    closing = ">"
    if body.endswith("/"):
        body, closing = body[:-1].rstrip(), "/>"
    return f"{body} {attribute}{closing}"


def _with_attribute(tag: str, name: str, value: str) -> str:
    pattern = re.compile(rf"""(?<![\w-]){re.escape(name)}\s*=\s*["'][^"']*["']""", re.IGNORECASE)
    if pattern.search(tag):
        return pattern.sub(lambda _: f'{name}="{value}"', tag, count=1)
    return _appended(tag, f'{name}="{value}"')


def _without_attribute(tag: str, name: str) -> str:
    return re.sub(rf"""\s{re.escape(name)}\s*=\s*["'][^"']*["']""", "", tag, flags=re.IGNORECASE)


def _hidden(tag: str) -> str:
    """A dropped clip: no longer a clip, and hidden; its element stays for the composition's script."""
    for name in TIMING_ATTRIBUTES:
        tag = _without_attribute(tag, name)
    classes = _attribute(tag, "class")
    if classes is not None:
        kept = " ".join(c for c in classes.split() if c != "clip")
        tag = _CLASS.sub(lambda _: f'class="{kept}"', tag, count=1)
    return _appended(tag, HIDDEN_ATTRIBUTE)


def _cut_tag(tag: str, stretches: Sequence[Stretch]) -> str:
    """A clip's start tag with its span moved through the cut; other tags as they are."""
    if _attribute(tag, "data-composition-id") is not None:
        return tag
    start, duration = _number(_attribute(tag, "data-start")), _number(_attribute(tag, "data-duration"))
    if start is None or duration is None:
        return tag
    spans = _overlaps(start, start + duration, stretches)
    if not spans:
        return _hidden(tag)
    first, last = spans[0][0], spans[-1][1]
    tag = _with_attribute(tag, "data-start", f"{round(first, TIME_DECIMALS):g}")
    return _with_attribute(tag, "data-duration", f"{round(last - first, TIME_DECIMALS):g}")


def cut_html(html: str, stretches: Sequence[Stretch]) -> str:
    """``html`` with every clip moved through the cut; scripts and styles are left as they are."""
    raw = [match.span() for match in _RAW_TEXT.finditer(html)]

    def inside_raw_text(position: int) -> bool:
        return any(start < position < end for start, end in raw)

    def cut(match: "re.Match[str]") -> str:
        return match.group(0) if inside_raw_text(match.start()) else _cut_tag(match.group(0), stretches)

    return _START_TAG.sub(cut, html)


def _with_runtime(html: str) -> str:
    """``html`` with :data:`CUT_RUNTIME` before its first inline script (after any GSAP it loads)."""
    for match in _INLINE_SCRIPT.finditer(html):
        if _attribute(match.group(0), "src") is None:
            return html[: match.start()] + f"<script>{CUT_RUNTIME}</script>\n" + html[match.start():]
    return html


def _cut_entries(entries: Any, stretches: Sequence[Stretch]) -> Any:
    if not isinstance(entries, list):
        return entries
    kept = []
    for entry in entries:
        at = entry.get("at") if isinstance(entry, dict) else None
        if not isinstance(at, (int, float)) or isinstance(at, bool):
            kept.append(entry)
            continue
        moved, inside = place(float(at), stretches)
        if inside:
            kept.append({**entry, "at": round(moved, TIME_DECIMALS)})
    return kept


def cut_audio_plan(plan: Mapping[str, Any], stretches: Sequence[Stretch]) -> Dict[str, Any]:
    """The audio plan through the cut: voice lines and SFX inside a kept stretch move with it, the rest drop.

    The music cue stays as it is: it plays from its start for the cut's length.
    """
    cut = dict(plan)
    voice = plan.get("voice")
    if isinstance(voice, dict) and isinstance(voice.get("lines"), list):
        lines = _cut_entries(voice["lines"], stretches)
        if lines:
            cut["voice"] = {**voice, "lines": lines}
        else:
            cut.pop("voice")
    if isinstance(plan.get("sfx"), list):
        cut["sfx"] = _cut_entries(plan["sfx"], stretches)
    return cut


def cut_moments(blocks: Mapping[str, Any], seconds: int, moments: Sequence[float]) -> List[float]:
    """Snapshot ``moments`` (source seconds) of a render at ``seconds``: moved through its cut, those outside it dropped.

    A length without a cut keeps the moments that fall within it.
    """
    stretches = stretches_for(blocks, seconds)
    if not stretches:
        return [float(m) for m in moments if float(m) <= seconds]
    placed = (place(float(m), stretches) for m in moments)
    return [round(at, TIME_DECIMALS) for at, inside in placed if inside]


def slots_cut_out(blocks: Mapping[str, Any], seconds: int) -> Set[str]:
    """The footage slots a render at ``seconds`` never shows: their clips lie outside its cut,
    so no footage is made for them."""
    stretches = stretches_for(blocks, seconds)
    if not stretches:
        return set()
    hidden: Set[str] = set()
    for match in _START_TAG.finditer(blocks.get("html") or ""):
        tag = match.group(0)
        name = _attribute(tag, "data-slot")
        start, duration = _number(_attribute(tag, "data-start")), _number(_attribute(tag, "data-duration"))
        if name and start is not None and duration is not None and not _overlaps(start, start + duration, stretches):
            hidden.add(name)
    return hidden


def cut_to_length(blocks: Mapping[str, Any], seconds: int) -> Dict[str, Any]:
    """The blocks a render at ``seconds`` takes: the root's duration set, and the template's cut applied when it declares one."""
    html = with_root_duration(blocks.get("html") or "", seconds)
    stretches = stretches_for(blocks, seconds)
    if not stretches:
        return {**blocks, "html": html}
    encoded = json.dumps([[round(s, TIME_DECIMALS), round(e, TIME_DECIMALS)] for s, e in stretches], separators=(",", ":"))
    cut = {
        **blocks,
        "html": _with_runtime(cut_html(with_root_attribute(html, CUT_ATTRIBUTE, encoded), stretches)),
        "css": (blocks.get("css") or "") + HIDDEN_CSS,
    }
    if isinstance(blocks.get("audio_plan"), Mapping):
        cut["audio_plan"] = cut_audio_plan(blocks["audio_plan"], stretches)
    return cut


__all__ = [
    "CUT_ATTRIBUTE",
    "CUT_RUNTIME",
    "HIDDEN_ATTRIBUTE",
    "cut_audio_plan",
    "cut_html",
    "cut_moments",
    "cut_to_length",
    "place",
    "slots_cut_out",
    "stretches_for",
]
