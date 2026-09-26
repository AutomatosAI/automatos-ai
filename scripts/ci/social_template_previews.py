"""PRD-251 US-106/US-107 (S1.2): check every seeded social template through a running media-render.

The media-render CI job starts the image and runs this with the runner's own
python3 (standard library only) and ``PYTHONPATH=orchestrator``. It uses the
orchestrator's very code, so what it renders is what a workspace gets:

* the starters come from ``modules.documents.social_starters`` (the seed files,
  checked against the template contract, the rows ``seed_social_starters`` writes);
* each bundle is built by ``core.media_render_bundle.build_bundle`` from the
  starter's sample data (the reference video's own copy) and a brand kit, with
  every footage slot empty, so the templates' own motion graphics play.

For each template it posts the bundle with a ``preview`` (the template's key
moments) to ``POST /render``. media-render stages it, speaks it with Kokoro,
mixes it and runs ``hyperframes check`` on the FULL composition: a 202 means the
check found no errors (a 422 carries the findings and fails this script). The
job then snapshots the composition at those moments: small PNG frames and a
short reel, kept as the job's artifact.

Every seeded social IMAGE template (US-107) is rendered for real, as the
workspace gets it, at every size it declares: a still bundle, the full check
(202 = no errors), and the PNGs it returns, one per still (a carousel's
slides), each read back and measured against the size.

The music (US-112, S1.6): a video whose audio plan names a library track is
mixed with it for the preview. The job's report must name that track, with its
licence and the credit line a post carries, and the mix must measure -14 +/- 1
LUFS (loudnorm's own measurement of the mix the video plays).

The pixel probe (S1.2: changing the brand kit's primary colour changes the
render): a template whose seed names a probe (a spot the brand colour fills)
is rendered again with the primary swapped. The pixel there must match the
bundle's own token (``primary-on-ink`` on a video's stage, ``primary`` on an
image's brand stripe) in both renders, and the two must differ.

The infographic (US-113, S1.7: a chart bound to a report). The Infographic's
sample data is the binding of a fixture report's table (its top five rows, the
chip naming the report), so its renders above ARE a report's table rendered;
the driver checks that binding against the seed. Then it binds more fixture
reports with the orchestrator's own parser and binder (``core.report_tables``,
``core.chart_binding``): a line of a monthly table, a number grid of a
percentage column, a stress table whose labels are longer than the template
holds, as a bar, a line and a grid, and a table of 20-character figures (a
currency code and cents) as a grid and a bar.
Each renders at every size with 0 check errors (the axis labels fit), and
every figure the bundle puts on the chart is the report's own cell, the chip
naming the report.

The heading font (US-108, S1.3: a template renders with the brand kit's heading
font, an uploaded woff2): the Title card is rendered with a brand kit whose
``font_files`` carry "CI Block" (``ci_block_font.py``, a woff2 whose every
character is a solid block), named in ``heading_font``, and with an uploaded
logo mark in place of the logo. The headline's accent words, the only text in
``primary-on-paper-large``, must fill their bounding box as solid blocks do,
where the same words in the kit's own heading font (a real typeface) do not.

    python3 scripts/ci/social_template_previews.py --url http://127.0.0.1:8090 --token "$TOKEN" --out "$RUNNER_TEMP/templates"
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.chart_binding import chart_values, chip_text, max_chars_of, shown_rows, spec_of
from core.media_render_bundle import build_bundle
from core.report_tables import Series, first_table, parse_figure, table_series
from core.social_templates import SOCIAL_IMAGE, SOCIAL_VIDEO, parse_size, resolve_variables
from modules.documents.social_starters import social_starters

# The brand kit the previews render with: the reference videos' own Studio Dark,
# in fonts the image has installed (fonts-liberation), and a drawn logo.
KIT: Dict[str, Any] = {
    "name": "Automatos",
    "tagline": "An operating system for autonomous agent teams",
    "primary_color": "#e96235",
    "secondary_color": "#1a1714",
    "accent_color": "#90af5a",
    "text_color": "#f0e8db",
    "font_family": '"Liberation Sans", sans-serif',
    "heading_font": '"Liberation Serif", serif',
}
# The primary the probe swaps in: a blue, as far from the kit's orange as it gets.
PROBE_PRIMARY = "#2f7bf6"
# US-108: the heading-font render. The Title card's headline is weight 900 and its
# accent words are the only text in this token; set in solid blocks, they fill at
# least this much of their bounding box (anti-aliased edges and the gaps between
# blocks are the rest), and no real typeface comes close.
HEADING_FONT_TEMPLATE = "title-card"
HEADING_FONT_TOKEN = "primary-on-paper-large"
HEADING_FONT_WEIGHT = 900
BLOCK_FILL = 0.8
# A row of the accent words holds at least this many of their pixels; a stray
# anti-aliased edge elsewhere (the eyebrow's orange triangle on its dark pill
# blends through the accent colour) never does.
MIN_ROW_INK = 10
PROBE_TOKEN_TOLERANCE = 8
POLL_SECONDS = 2.0
JOB_WAIT_SECONDS = 900
TOKEN_HEADER = "X-Internal-Token"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PreviewFailure(RuntimeError):
    """A template that did not check, preview or render cleanly."""


# ── PNG, both ways (standard library only) ──────────────────────────────────
def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(width: int, height: int, pixel) -> bytes:
    """An 8-bit RGBA PNG; ``pixel(x, y)`` gives each ``(r, g, b, a)``."""
    rows = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows, 9)) + _chunk(b"IEND", b"")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def decode_png(data: bytes) -> Tuple[int, int, int, List[bytearray]]:
    """``(width, height, channels, rows)`` of an 8-bit, non-interlaced RGB or RGBA PNG."""
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    pos, idat, header = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    width, height, depth, colour, _, _, interlace = header
    if depth != 8 or colour not in (2, 6) or interlace:
        raise ValueError(f"unsupported PNG: depth {depth}, colour type {colour}, interlace {interlace}")
    channels = 3 if colour == 2 else 4
    stride = width * channels
    raw = zlib.decompress(idat)
    rows: List[bytearray] = []
    previous = bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = previous[i]
            corner = previous[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + left) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + up) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + ((left + up) >> 1)) & 0xFF
            elif kind == 4:
                line[i] = (line[i] + _paeth(left, up, corner)) & 0xFF
        rows.append(line)
        previous = line
    return width, height, channels, rows


def sample(data: bytes, fx: float, fy: float, radius: int = 1) -> Tuple[int, int, int]:
    """The average colour around ``(fx, fy)``, given as fractions of the frame."""
    width, height, channels, rows = decode_png(data)
    cx, cy = round(fx * (width - 1)), round(fy * (height - 1))
    picked = [
        rows[y][x * channels : x * channels + 3]
        for y in range(max(0, cy - radius), min(height, cy + radius + 1))
        for x in range(max(0, cx - radius), min(width, cx + radius + 1))
    ]
    return tuple(round(sum(p[i] for p in picked) / len(picked)) for i in range(3))  # type: ignore[return-value]


def hex_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _block_font():
    """``scripts/ci/ci_block_font.py``, loaded by path: the tests load this driver by path too."""
    spec = importlib.util.spec_from_file_location("ci_block_font", Path(__file__).resolve().with_name("ci_block_font.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def heading_font_kit(kit: Mapping[str, Any]) -> Dict[str, Any]:
    """``kit`` as a workspace with an uploaded heading font and logo mark has it, render-ready.

    The shape ``brand_kit_for_media_render`` hands the bundle: the woff2 and the
    mark inlined as data: URIs, and no wordmark logo.
    """
    block = _block_font()
    woff2 = base64.b64encode(block.block_font_woff2()).decode("ascii")
    face = {"family": block.FAMILY, "weight": HEADING_FONT_WEIGHT, "style": "normal", "data_uri": f"data:font/woff2;base64,{woff2}"}
    unbranded = {key: value for key, value in kit.items() if key != "logo_url"}
    return {
        **unbranded,
        "heading_font": f'"{block.FAMILY}", {kit["heading_font"]}',
        "font_files": [face],
        "logo_mark_url": logo_png(kit["primary_color"]),
    }


def ink_fill(data: bytes, colour: Tuple[int, int, int]) -> Tuple[int, float]:
    """``(pixels, fill)`` of the text set in ``colour``: its pixels, and how much of their bounding box they fill.

    Only rows holding at least :data:`MIN_ROW_INK` pixels of the colour count as the text's.
    """
    width, _, channels, rows = decode_png(data)
    lines = []
    for y, row in enumerate(rows):
        hits = [
            x for x in range(width)
            if all(abs(row[x * channels + i] - colour[i]) <= PROBE_TOKEN_TOLERANCE for i in range(3))
        ]
        if len(hits) >= MIN_ROW_INK:
            lines.append((y, hits))
    if not lines:
        return 0, 0.0
    count = sum(len(hits) for _, hits in lines)
    left, right = min(hits[0] for _, hits in lines), max(hits[-1] for _, hits in lines)
    top, bottom = lines[0][0], lines[-1][0]
    return count, count / ((right - left + 1) * (bottom - top + 1))


def logo_png(colour: str) -> str:
    """A drawn 128 px mark (a ring and a dot in ``colour``), as the data: URI a brand kit's uploaded logo becomes."""
    r, g, b = hex_rgb(colour)

    def pixel(x: int, y: int):
        d = ((x - 63.5) ** 2 + (y - 63.5) ** 2) ** 0.5
        if d <= 20 or 44 <= d <= 60:
            return r, g, b, 255
        return 0, 0, 0, 0

    return "data:image/png;base64," + base64.b64encode(encode_png(128, 128, pixel)).decode("ascii")


# ── media-render ────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, url: str, token: str) -> None:
        self.url, self.token = url.rstrip("/"), token

    def _request(self, method: str, path: str, body: Optional[bytes] = None) -> Tuple[int, bytes]:
        request = urllib.request.Request(self.url + path, data=body, method=method)
        request.add_header(TOKEN_HEADER, self.token)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=JOB_WAIT_SECONDS) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def render(self, bundle: Mapping[str, Any]) -> Dict[str, Any]:
        """Post the bundle; the finished job (outputs fetched), or PreviewFailure with the service's answer."""
        started = time.monotonic()
        status, body = self._request("POST", "/render", json.dumps(bundle).encode("utf-8"))
        answer = json.loads(body or b"{}")
        if status != 202:
            findings = answer.get("findings") or []
            for finding in findings[:40]:
                partner = f" with {finding['containerSelector']}" if finding.get("containerSelector") else ""
                print(f"    {finding.get('severity')}: {finding.get('section')}/{finding.get('code')}: {finding.get('message')} "
                      f"{finding.get('selector') or ''}{partner} t={finding.get('time')}")
            raise PreviewFailure(f"POST /render answered {status}: {answer.get('message') or answer}")
        print(f"    checked in {time.monotonic() - started:.1f} s (staged, spoken, mixed, checked); job {answer['id']}")
        job = answer
        while job.get("status") not in ("done", "failed", "rejected"):
            if time.monotonic() - started > JOB_WAIT_SECONDS:
                raise PreviewFailure(f"job {answer['id']} did not finish in {JOB_WAIT_SECONDS} s")
            time.sleep(POLL_SECONDS)
            status, body = self._request("GET", f"/render/{answer['id']}")
            job = json.loads(body)
        if job["status"] != "done":
            raise PreviewFailure(f"job {job['id']} {job['status']}: {json.dumps(job.get('error'))}")
        for output in job["outputs"]:
            status, data = self._request("GET", output["path"])
            if status != 200:
                raise PreviewFailure(f"GET {output['path']} answered {status}")
            output["data"] = data
        job["seconds"] = round(time.monotonic() - started, 1)
        return job


def _sample_values(starter: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"])
    if resolved.missing or resolved.invalid:
        raise PreviewFailure(f"{starter['name']}: its sample data leaves {resolved.missing} missing, {resolved.invalid} invalid")
    return resolved.values


def bundle_for(starter: Mapping[str, Any], kit: Mapping[str, Any], at: List[float]) -> Dict[str, Any]:
    """A video's preview bundle: its sample data, snapshotted at ``at``."""
    bundle = build_bundle(
        workspace_id="ci-social-templates",
        reference=f"seeded template: {starter['name']}",
        blocks=starter["blocks"],
        values=_sample_values(starter),
        brand_kit=kit,
    )
    bundle["preview"] = {"at": at}
    return bundle


def image_bundle_for(starter: Mapping[str, Any], kit: Mapping[str, Any], size: str) -> Dict[str, Any]:
    """An image's bundle at ``size``, exactly as a workspace renders it: a still per moment."""
    return build_bundle(
        workspace_id="ci-social-templates",
        reference=f"seeded template: {starter['name']} {size}",
        blocks=starter["blocks"],
        values=_sample_values(starter),
        brand_kit=kit,
        size=size,
        fmt=SOCIAL_IMAGE,
    )


LUFS_TARGET = -14.0
LUFS_TOLERANCE = 1.0


def check_music(starter: Mapping[str, Any], report: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The template's library track in the job's report, with its credit, and the mix at -14 +/- 1 LUFS."""
    wanted = ((starter["blocks"].get("audio_plan") or {}).get("music") or {}).get("track")
    if not wanted:
        return None
    music = report.get("music") or {}
    if music.get("track") != wanted:
        raise PreviewFailure(f"the audio plan names {wanted}, the report's music is {json.dumps(music)}")
    if not music.get("attribution") or not music.get("licence"):
        raise PreviewFailure(f"the report gives {wanted} no licence or attribution: {json.dumps(music)}")
    lufs = (report.get("audio") or {}).get("integrated_lufs")
    if not isinstance(lufs, (int, float)) or abs(lufs - LUFS_TARGET) > LUFS_TOLERANCE:
        raise PreviewFailure(f"the mix measured {lufs} LUFS, not {LUFS_TARGET:g} +/- {LUFS_TOLERANCE:g}")
    print(f"    music: {wanted} from {music.get('start')} s to {music.get('end')} s, the mix at {lufs} LUFS; "
          f"credit ({music.get('licence')}): {music.get('attribution')}")
    return {"track": wanted, "window": [music.get("start"), music.get("end")], "lufs": lufs, "credit": music.get("attribution")}


def _check_summary(report: Mapping[str, Any]) -> str:
    check = report.get("check") or {}
    sections = ", ".join(
        f"{name} {section.get('errors', 0)}E/{section.get('warnings', 0)}W" for name, section in (check.get("sections") or {}).items()
    )
    return f"hyperframes check: {check.get('errors')} error(s), {check.get('warnings')} warning(s) [{sections}]"


# ── the infographic (US-113, S1.7) ─────────────────────────────────────────
INFOGRAPHIC = "infographic"
# The reports the infographic is bound to: a title, when it was made, and its file (markdown).
CHANNEL_REPORT: Dict[str, str] = {
    "title": "Harbourline Coffee — September social report",
    "as_of": "2026-09-21T09:30:00+00:00",
    "markdown": (
        "# Harbourline Coffee — September social report\n\n"
        "Every channel we posted on in September, busiest first. Reach counts the unique accounts each channel reached.\n\n"
        "| Channel | Posts | Reach | Engagement rate |\n"
        "| --- | ---: | ---: | ---: |\n"
        "| Instagram | 14 | 48,210 | 5.2% |\n"
        "| LinkedIn | 9 | 21,480 | 3.9% |\n"
        "| TikTok | 6 | 19,305 | 6.8% |\n"
        "| Threads | 8 | 7,940 | 2.4% |\n"
        "| X | 11 | 6,115 | 1.7% |\n"
        "| YouTube Shorts | 2 | 3,020 | 4.4% |\n\n"
        "Next month: two more Shorts a week.\n"
    ),
}
TREND_REPORT: Dict[str, str] = {
    "title": "Harbourline Coffee — followers, May to September",
    "as_of": "2026-09-30T17:05:00+00:00",
    "markdown": (
        "## Followers at each month's end\n\n"
        "| Month | Followers |\n|:--|--:|\n"
        "| May | 8,120 |\n| June | 9,480 |\n| July | 11,905 |\n| August | 13,260 |\n| September | 16,340 |\n"
    ),
}
# The stress: labels longer than the template holds (cut, with an ellipsis), the widest figures, a long title.
PROGRAMME_REPORT: Dict[str, str] = {
    "title": "Northwind Studio — marketing programmes, spend to date, all regions, fiscal year 2026",
    "as_of": "2026-09-24T08:00:00+00:00",
    "markdown": (
        "| Programme | Spend to date |\n| --- | ---: |\n"
        "| Customer success stories from the Lisbon launch week | $1,204,560.75 |\n"
        "| Partner webinars with the regional distributors in Iberia | $986,410.20 |\n"
        "| Always-on brand campaign across every paid social channel | $874,002.00 |\n"
        "| Founder-led posts and replies on the two biggest networks | $512,950.55 |\n"
        "| Retargeting for visitors who read the pricing page twice | $98,745.10 |\n"
        "| Print | $1,200.00 |\n"
    ),
}
# The widest figures a row holds: 20 characters of a 24-character limit, a currency code and cents.
REVENUE_REPORT: Dict[str, str] = {
    "title": "Northwind Studio \u2014 revenue by region, fiscal year 2026",
    "as_of": "2026-09-25T12:00:00+00:00",
    "markdown": (
        "| Region | Revenue |\n| --- | ---: |\n"
        "| North America | USD 1,234,567,890.12 |\n"
        "| Europe, Middle East and Africa | USD 987,654,321.09 |\n"
        "| Asia Pacific | USD 876,543,210.98 |\n"
        "| Latin America | USD 123,456,789.01 |\n"
        "| Rest of world | USD 12,345,678.90 |\n"
    ),
}
# The starter's sample data is this binding (its renders at every size, above, are a report's table).
SAMPLE_BINDING = (CHANNEL_REPORT, "Reach", "bar")
# (name, report, the figures' column, the kind): the other bindings, each rendered at every size.
INFOGRAPHIC_RENDERS = (
    ("followers-line", TREND_REPORT, None, "line"),
    ("engagement-grid", CHANNEL_REPORT, "Engagement rate", "grid"),
    ("programmes-bar", PROGRAMME_REPORT, None, "bar"),
    ("programmes-line", PROGRAMME_REPORT, None, "line"),
    ("programmes-grid", PROGRAMME_REPORT, None, "grid"),
    ("revenue-grid", REVENUE_REPORT, None, "grid"),
    ("revenue-bar", REVENUE_REPORT, None, "bar"),
)
# Each bound render's own copy (the rest is the starter's sample copy), so a preview reads as a post would.
RENDER_COPY: Dict[str, Dict[str, str]] = {
    "followers-line": {"eyebrow": "FIVE MONTHS", "headline": "FOLLOWERS,|MONTH BY|MONTH", "headline_accent": "MONTH BY",
                       "note": "Followers at each month's end."},
    "engagement-grid": {"eyebrow": "SEPTEMBER IN NUMBERS", "headline": "WHERE PEOPLE|TALKED BACK", "headline_accent": "TALKED BACK",
                        "note": "Engagement rate: interactions per account reached."},
    "programmes-bar": {"eyebrow": "SPEND TO DATE", "headline": "WHERE THE|BUDGET WENT", "headline_accent": "BUDGET",
                       "note": "The top 5 of 6 programmes, in the report's order."},
}
RENDER_COPY["programmes-line"] = RENDER_COPY["programmes-grid"] = RENDER_COPY["programmes-bar"]
RENDER_COPY["revenue-grid"] = RENDER_COPY["revenue-bar"] = {
    "eyebrow": "REVENUE BY REGION", "headline": "WHERE THE|REVENUE CAME|FROM", "headline_accent": "REVENUE",
    "note": "Each figure as the finance report wrote it.",
}


def bind_fixture(starter: Mapping[str, Any], report: Mapping[str, str], column: Optional[str], kind: str) -> Tuple[Dict[str, str], Series]:
    """The chart variables a workspace's binding gives (``modules/socials/report_charts.bind_report``), from the fixture."""
    blocks = starter["blocks"]
    spec = spec_of(blocks)
    series = table_series(first_table(report["markdown"]), column=column, limit=spec.rows)
    values = chart_values(spec, blocks["variables_schema"], series, kind=kind, title=report["title"], as_of=report["as_of"])
    return values, series


def check_sample_binding(starter: Mapping[str, Any]) -> None:
    """The seed's sample chart is the channel report's binding, row for row, chip and kind."""
    report, column, kind = SAMPLE_BINDING
    values, _ = bind_fixture(starter, report, column, kind)
    spec = spec_of(starter["blocks"])
    sample = starter["sample_data"]
    wanted = {name: value for name, value in values.items() if name != spec.header}
    got = {name: sample.get(name, "") for name in wanted}
    if got != wanted:
        raise PreviewFailure(f"the seed's sample chart is not the channel report's binding: {got} != {wanted}")


def check_chart_from_report(bundle: Mapping[str, Any], starter: Mapping[str, Any], report: Mapping[str, str], series: Series) -> List[Tuple[str, str]]:
    """Every figure the chart shows is the report's own cell (and every label its row's), and the chip names the report."""
    blocks = starter["blocks"]
    spec = spec_of(blocks)
    shown = shown_rows(spec, bundle["variables"])
    table = first_table(report["markdown"])
    column = table.headers.index(series.value_header)
    for n, (label, value) in enumerate(shown, start=1):
        if n > len(series.rows):
            if (label, value) != ("", ""):
                raise PreviewFailure(f"row {n} shows {label!r} {value!r}; the report has no row {n} on the chart")
            continue
        cells = table.rows[n - 1]
        if value != cells[column] or parse_figure(value).value != parse_figure(cells[column]).value:
            raise PreviewFailure(f"row {n}'s figure {value!r} is not the report's {cells[column]!r}")
        if label != cells[0] and not (label.endswith("…") and cells[0].startswith(label[:-1].rstrip())):
            raise PreviewFailure(f"row {n}'s label {label!r} is not the report's {cells[0]!r}")
    chip = chip_text(report["title"], report["as_of"], max_chars_of(blocks["variables_schema"], spec.source))
    if bundle["variables"].get(spec.source) != chip:
        raise PreviewFailure(f"the chip reads {bundle['variables'].get(spec.source)!r}, not the report's {chip!r}")
    return shown


def chart_bundle_for(starter: Mapping[str, Any], kit: Mapping[str, Any], size: str, values: Mapping[str, str]) -> Dict[str, Any]:
    """The Infographic at ``size`` with its chart bound to a report: the sample copy, the report's rows."""
    resolved = resolve_variables(starter["blocks"]["variables_schema"], {**starter["sample_data"], **values})
    if resolved.missing or resolved.invalid:
        raise PreviewFailure(f"the binding leaves {resolved.missing} missing, {resolved.invalid} invalid")
    return build_bundle(
        workspace_id="ci-social-templates",
        reference=f"seeded template: {starter['name']} {size}, bound to a report",
        blocks=starter["blocks"],
        values=resolved.values,
        brand_kit=kit,
        size=size,
        fmt=SOCIAL_IMAGE,
    )


def run_infographic(renderer: Renderer, out: Path, kit: Mapping[str, Any], report: Dict[str, Any]) -> List[str]:
    """US-113 (S1.7): the Infographic bound to fixture reports, rendered as a line, a grid and a stress table, at every size."""
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == INFOGRAPHIC)
    entry: Dict[str, Any] = report.setdefault(INFOGRAPHIC, {}).setdefault("bound", {})
    failures: List[str] = []
    try:
        check_sample_binding(starter)
        print(f"\nThe {starter['name']}'s sample chart is the channel report's binding (column {SAMPLE_BINDING[1]!r}, {SAMPLE_BINDING[2]}): PASS")
    except PreviewFailure as exc:
        print(f"    FAIL: {exc}")
        failures.append(f"{starter['name']}: {exc}")
    for name, source, column, kind in INFOGRAPHIC_RENDERS:
        values, series = bind_fixture(starter, source, column, kind)
        for size in starter["blocks"]["sizes"]:
            folder = out / INFOGRAPHIC / name
            folder.mkdir(parents=True, exist_ok=True)
            print(f"\n== {starter['name']} bound to {source['title']!r} ({series.value_header}, {kind}) at {size}")
            try:
                bundle = chart_bundle_for(starter, kit, size, {**RENDER_COPY.get(name, {}), **values})
                shown = check_chart_from_report(bundle, starter, source, series)
                data = _one_png(_checked(renderer, bundle))
                if png_size(data) != parse_size(size):
                    raise PreviewFailure(f"the PNG is {png_size(data)}, not {size}")
                (folder / f"{size}.png").write_bytes(data)
                print(f"    rows: {shown}")
                print(f"    chip: {bundle['variables'][spec_of(starter['blocks']).source]!r}; every figure is the report's own cell: PASS")
                entry.setdefault(name, {})[size] = {"rows": shown, "bytes": len(data)}
            except PreviewFailure as exc:
                print(f"    FAIL: {exc}")
                failures.append(f"{starter['name']} {name} at {size}: {exc}")
    return failures


def _checked(renderer: Renderer, bundle: Mapping[str, Any]) -> Dict[str, Any]:
    job = renderer.render(bundle)
    print(f"    {_check_summary(job['report'])}")
    return job


def run(renderer: Renderer, out: Path) -> List[str]:
    kit = {**KIT, "logo_url": logo_png(KIT["primary_color"])}
    report: Dict[str, Any] = {}
    failures = run_videos(renderer, out, kit, report) + run_images(renderer, out, kit, report)
    failures += run_infographic(renderer, out, kit, report)
    failures += run_heading_font(renderer, out, kit, report)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return failures


def run_videos(renderer: Renderer, out: Path, kit: Mapping[str, Any], report: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    videos = social_starters(SOCIAL_VIDEO)
    print(f"{len(videos)} seeded social video templates: {', '.join(s['name'] for s in videos)}")
    for starter in videos:
        name, preview = starter["name"], starter["preview"]
        folder = out / starter["slug"]
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n== {name} ({starter['slug']}): preview at {preview['at']}")
        try:
            bundle = bundle_for(starter, kit, preview["at"])
            job = renderer.render(bundle)
            print(f"    {_check_summary(job['report'])}")
            if (job["report"].get("check") or {}).get("errors"):
                raise PreviewFailure("the check passed the job with errors")
            for output in job["outputs"]:
                (folder / output["name"]).write_bytes(output.pop("data"))
                print(f"    {output['name']}: {output.get('width')}x{output.get('height')}, {output['bytes']} bytes"
                      + (f", at {output['at']} s" if "at" in output else f", {output.get('duration')} s reel"))
            print(f"    timings: {json.dumps(job['report'].get('timings'))}; {job['seconds']} s in all")
            report[starter["slug"]] = {"check": job["report"].get("check"), "outputs": job["outputs"], "timings": job["report"].get("timings")}
            music = check_music(starter, job["report"])
            if music:
                report[starter["slug"]]["music"] = music
            probe = preview.get("probe")
            if probe:
                report[starter["slug"]]["probe"] = probe_primary(renderer, starter, kit, bundle, job, probe, folder)
        except PreviewFailure as exc:
            print(f"    FAIL: {exc}")
            failures.append(f"{name}: {exc}")
    return failures


def png_size(data: bytes) -> Tuple[int, int]:
    if data[:8] != PNG_SIGNATURE:
        raise PreviewFailure("the output is not a PNG")
    return struct.unpack(">II", data[16:24])


def run_images(renderer: Renderer, out: Path, kit: Mapping[str, Any], report: Dict[str, Any]) -> List[str]:
    """Every seeded image template, rendered at every size it declares: checked, and one PNG per still."""
    failures: List[str] = []
    images = social_starters(SOCIAL_IMAGE)
    print(f"\n{len(images)} seeded social image templates: {', '.join(s['name'] for s in images)}")
    for starter in images:
        name, sizes = starter["name"], starter["blocks"]["sizes"]
        entry: Dict[str, Any] = report.setdefault(starter["slug"], {"sizes": {}})
        for size in sizes:
            folder = out / starter["slug"] / size
            folder.mkdir(parents=True, exist_ok=True)
            print(f"\n== {name} ({starter['slug']}) at {size}")
            try:
                bundle = image_bundle_for(starter, kit, size)
                job = renderer.render(bundle)
                print(f"    {_check_summary(job['report'])}")
                if (job["report"].get("check") or {}).get("errors"):
                    raise PreviewFailure("the check passed the job with errors")
                moments = bundle["still"]["at"]
                if len(job["outputs"]) != len(moments):
                    raise PreviewFailure(f"{len(job['outputs'])} PNGs for {len(moments)} stills")
                for output in job["outputs"]:
                    data = output.pop("data")
                    if png_size(data) != parse_size(size) or (output.get("width"), output.get("height")) != parse_size(size):
                        raise PreviewFailure(f"{output['name']} is {png_size(data)}, not {size}")
                    (folder / output["name"]).write_bytes(data)
                    print(f"    {output['name']}: {size} PNG at {output.get('at')} s, {output['bytes']} bytes")
                print(f"    timings: {json.dumps(job['report'].get('timings'))}; {job['seconds']} s in all")
                entry["sizes"][size] = {"check": job["report"].get("check"), "outputs": job["outputs"]}
            except PreviewFailure as exc:
                print(f"    FAIL: {exc}")
                failures.append(f"{name} at {size}: {exc}")
        probe = (starter.get("preview") or {}).get("probe")
        if probe and not any(f.startswith(f"{name} at ") for f in failures):
            try:
                entry["probe"] = probe_image(renderer, starter, kit, probe, out / starter["slug"])
            except PreviewFailure as exc:
                print(f"    FAIL: {exc}")
                failures.append(f"{name}: {exc}")
    return failures


def _one_png(job: Mapping[str, Any]) -> bytes:
    if (job["report"].get("check") or {}).get("errors"):
        raise PreviewFailure("the check passed the job with errors")
    if len(job["outputs"]) != 1:
        raise PreviewFailure(f"{len(job['outputs'])} PNGs for one still")
    return job["outputs"][0].pop("data")


def run_heading_font(renderer: Renderer, out: Path, kit: Mapping[str, Any], report: Dict[str, Any]) -> List[str]:
    """US-108 (S1.3): the Title card renders with the brand kit's heading font, an uploaded woff2."""
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == HEADING_FONT_TEMPLATE)
    size = starter["blocks"]["sizes"][0]
    folder = out / starter["slug"] / "heading-font"
    folder.mkdir(parents=True, exist_ok=True)
    font_kit = heading_font_kit(kit)
    family = font_kit["font_files"][0]["family"]
    print(f"\n== {starter['name']} at {size} with the heading font {family!r}, an uploaded woff2, and an uploaded logo mark")
    try:
        bundle = image_bundle_for(starter, font_kit, size)
        faces = bundle["brand"].get("fonts") or []
        if [face["family"] for face in faces] != [family] or bundle["variables"]["brand.logo_mark"] != "assets/brand/logo-mark.png":
            raise PreviewFailure(f"the bundle does not carry the uploaded font and mark: {faces}, {bundle['variables']['brand.logo_mark']}")
        print(f"    heading-font token: {bundle['brand']['tokens']['heading-font']}")
        colour = hex_rgb(bundle["brand"]["tokens"][HEADING_FONT_TOKEN])
        others = {name: value for name, value in bundle["brand"]["tokens"].items() if value.startswith("#") and name != HEADING_FONT_TOKEN}
        close = [name for name, value in others.items() if len(value) == 7 and all(abs(a - b) <= PROBE_TOKEN_TOLERANCE for a, b in zip(hex_rgb(value), colour))]
        if close:
            raise PreviewFailure(f"{HEADING_FONT_TOKEN} is too close to {close} to tell the accent words apart")
        base_job = renderer.render(image_bundle_for(starter, kit, size))
        base = _one_png(base_job)
        (folder / "kit-heading-font.png").write_bytes(base)
        job = renderer.render(bundle)
        print(f"    {_check_summary(job['report'])}")
        rendered = _one_png(job)
        (folder / "uploaded-heading-font.png").write_bytes(rendered)
        base_pixels, base_fill = ink_fill(base, colour)
        pixels, fill = ink_fill(rendered, colour)
        print(f"    the accent words ({HEADING_FONT_TOKEN} {bundle['brand']['tokens'][HEADING_FONT_TOKEN]}):")
        print(f"      in the kit's heading font ({kit['heading_font']}): {base_pixels} px, filling {base_fill:.2f} of their box")
        print(f"      in the uploaded {family!r}: {pixels} px, filling {fill:.2f} of their box")
        report["heading_font"] = {"template": starter["slug"], "size": size, "fill": [round(base_fill, 3), round(fill, 3)], "pixels": [base_pixels, pixels]}
        if not base_pixels or base_fill >= BLOCK_FILL:
            raise PreviewFailure(f"the kit's own heading font already fills {base_fill:.2f}: the measure cannot tell the fonts apart")
        if fill < BLOCK_FILL:
            raise PreviewFailure(f"the headline did not render in the uploaded {family!r} (fill {fill:.2f} < {BLOCK_FILL})")
        print("      PASS: the headline renders in the brand kit's heading font, the uploaded woff2")
    except PreviewFailure as exc:
        print(f"    FAIL: {exc}")
        return [f"{starter['name']} with the uploaded heading font: {exc}"]
    return []


def probe_image(renderer, starter, kit, probe, folder: Path) -> Dict[str, Any]:
    """The image at its first size, rendered again with the primary swapped: the pixel follows the brand kit."""
    size = starter["blocks"]["sizes"][0]
    token = probe["token"]
    first_bundle = image_bundle_for(starter, kit, size)
    first = sample((folder / size / _first_still(first_bundle)).read_bytes(), probe["x"], probe["y"])
    swapped_kit = {**kit, "primary_color": PROBE_PRIMARY}
    swapped_bundle = image_bundle_for(starter, swapped_kit, size)
    swapped_job = renderer.render(swapped_bundle)
    data = swapped_job["outputs"][0].pop("data")
    (folder / f"probe-swapped-{size}.png").write_bytes(data)
    second = sample(data, probe["x"], probe["y"])
    expected = (hex_rgb(first_bundle["brand"]["tokens"][token]), hex_rgb(swapped_bundle["brand"]["tokens"][token]))
    return _judge_probe(probe, f"the first still at {size}", kit, first, second, expected, token)


def _first_still(bundle: Mapping[str, Any]) -> str:
    return "render.png" if len(bundle["still"]["at"]) == 1 else "render-01.png"


def _judge_probe(probe, where, kit, first, second, expected, token) -> Dict[str, Any]:
    close = lambda got, want: all(abs(a - b) <= PROBE_TOKEN_TOLERANCE for a, b in zip(got, want))  # noqa: E731
    print(f"    pixel probe on {where} ({probe['what']}, {probe['x']},{probe['y']} of the frame):")
    print(f"      primary {kit['primary_color']}: pixel {first}, {token} {expected[0]}")
    print(f"      primary {PROBE_PRIMARY}: pixel {second}, {token} {expected[1]}")
    if not (close(first, expected[0]) and close(second, expected[1])):
        raise PreviewFailure("the probed pixel does not show the brand kit's primary colour")
    if first == second:
        raise PreviewFailure("swapping the primary colour did not change the probed pixel")
    print("      PASS: the pixel is the brand's primary in both renders, and it changed with the brand kit")
    return {"x": probe["x"], "y": probe["y"], "pixels": [first, second], "tokens": expected}


def _frame_at(job: Mapping[str, Any], folder: Path, at: float) -> bytes:
    for output in job["outputs"]:
        if output.get("kind") == "frame" and abs(output.get("at", -1) - at) < 1e-6:
            return (folder / output["name"]).read_bytes()
    raise PreviewFailure(f"the preview has no frame at {at} s")


def probe_primary(renderer, starter, kit, bundle, job, probe, folder: Path) -> Dict[str, Any]:
    """The same moment rendered with the primary swapped: the pixel follows the brand kit."""
    fx, fy = probe["x"] / bundle_width(bundle), probe["y"] / bundle_height(bundle)
    token = probe["token"]
    first = sample(_frame_at(job, folder, probe["at"]), fx, fy)
    swapped_kit = {**kit, "primary_color": PROBE_PRIMARY}
    swapped_bundle = bundle_for(starter, swapped_kit, [probe["at"]])
    swapped_job = renderer.render(swapped_bundle)
    frame = next(o for o in swapped_job["outputs"] if o.get("kind") == "frame")
    data = frame.pop("data")
    (folder / f"probe-swapped-{frame['name']}").write_bytes(data)
    second = sample(data, fx, fy)
    expected = (hex_rgb(bundle["brand"]["tokens"][token]), hex_rgb(swapped_bundle["brand"]["tokens"][token]))
    judged = _judge_probe(probe, f"the preview at {probe['at']} s", kit, first, second, expected, token)
    return {"at": probe["at"], **judged}


def bundle_width(bundle: Mapping[str, Any]) -> int:
    return int(bundle["variables"]["size.width"])


def bundle_height(bundle: Mapping[str, Any]) -> int:
    return int(bundle["variables"]["size.height"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    failures = run(Renderer(args.url, args.token), args.out)
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nevery seeded social video template checked with 0 errors and rendered its preview,")
    print("every seeded social image template checked with 0 errors and rendered its PNGs at every size,")
    print("the Infographic, bound to a report, showed the report's own rows and chip as a bar, a line and a grid at every size,")
    print("and the Title card rendered its headline in the brand kit's uploaded heading font: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
