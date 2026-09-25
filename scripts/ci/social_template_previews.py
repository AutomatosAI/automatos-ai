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

The pixel probe (S1.2: changing the brand kit's primary colour changes the
render): a template whose seed names a probe (a spot the brand colour fills)
is rendered again with the primary swapped. The pixel there must match the
bundle's own token (``primary-on-ink`` on a video's stage, ``primary`` on an
image's brand stripe) in both renders, and the two must differ.

    python3 scripts/ci/social_template_previews.py --url http://127.0.0.1:8090 --token "$TOKEN" --out "$RUNNER_TEMP/templates"
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.media_render_bundle import build_bundle
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


def _check_summary(report: Mapping[str, Any]) -> str:
    check = report.get("check") or {}
    sections = ", ".join(
        f"{name} {section.get('errors', 0)}E/{section.get('warnings', 0)}W" for name, section in (check.get("sections") or {}).items()
    )
    return f"hyperframes check: {check.get('errors')} error(s), {check.get('warnings')} warning(s) [{sections}]"


def run(renderer: Renderer, out: Path) -> List[str]:
    kit = {**KIT, "logo_url": logo_png(KIT["primary_color"])}
    report: Dict[str, Any] = {}
    failures = run_videos(renderer, out, kit, report) + run_images(renderer, out, kit, report)
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
    print("and every seeded social image template checked with 0 errors and rendered its PNGs at every size: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
