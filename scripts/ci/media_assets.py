#!/usr/bin/env python3
"""The media assets image, ghcr.io/<owner>/automatos-media-assets (PRD-262 US-053).

media-render's build is to read the Kokoro voice model, its voices and the music
library from this image, pinned as its own stage (``FROM
ghcr.io/automatosai/automatos-media-assets:<tag>@sha256:… AS assets``, FR-9),
instead of fetching them from GitHub releases and ende.app on every build. PR
builds never push to GHCR, so images.yml's ``media-assets`` job publishes it, on
a manual run on main with ``media_assets``. The job runs this tool twice:

* ``stage --out DIR`` fetches every file ``services/media-render/kokoro/assets.json``
  and ``services/media-render/music/manifest.json`` name, and checks each one's
  sha256 (and a Kokoro file's size). A file whose bytes differ fails the run and
  is not kept. It then writes the notices and a ``FROM scratch`` Dockerfile with
  one layer per file, and the image's licence expression as a step output;
* ``verify --dir DIR``, after the push, reads the image back as published, one
  platform at a time, and checks that it holds exactly those files, with those
  sha256s, and the same NOTICE and licence text.

The image's layout, which media-render's ``kokoro`` and ``music`` stages copy from:

    /kokoro/<release>/<file>   e.g. /kokoro/model-files-v1.0/voices-v1.0.bin
    /music/<id>.<ext>          the library's own file name (music_build.track_file)
    /licenses/Apache-2.0.txt   the licence the Kokoro weights are under
    /NOTICE                    the Kokoro notice and every track's attribution line

Its ``org.opencontainers.image.licenses`` label joins every licence it carries
with AND, ``Apache-2.0 AND CC-BY-4.0`` today. It holds data only, no GPL code, so
publishing it is separate from PRD-262 §9-Q6.

The music manifest is read with media-render's own validator
(``media_render/music_build.py``, loaded by path; it imports only the standard
library until it analyses audio), so the image takes exactly the tracks the
library takes. Standard library only: the job runs it with the runner's python3.

    python3 scripts/ci/media_assets.py stage --out DIR [--outputs FILE] [--summary FILE]
    python3 scripts/ci/media_assets.py verify --dir DIR [--summary FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
MEDIA_RENDER = REPO / "services" / "media-render"
KOKORO_LIST = MEDIA_RENDER / "kokoro" / "assets.json"
MUSIC_MANIFEST = MEDIA_RENDER / "music" / "manifest.json"
MUSIC_BUILD = MEDIA_RENDER / "media_render" / "music_build.py"
LICENSE = REPO / "LICENSE"

KOKORO_DIR = "kokoro"
MUSIC_DIR = "music"
NOTICE_PATH = "NOTICE"
APACHE_PATH = "licenses/Apache-2.0.txt"
DOCKERFILE = "Dockerfile"

# The only licence whose text the image ships, so the only one the Kokoro files may name.
KOKORO_LICENCE = "Apache-2.0"
APACHE_HEAD = ("Apache License", "Version 2.0, January 2004")
APACHE_END = "END OF TERMS AND CONDITIONS"

FILE_FIELDS = ("path", "url", "bytes", "sha256", "use")
LIST_FIELDS = ("about", "licence", "notice", "files")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# A track has no size in its manifest; no track is near this.
TRACK_MAX_BYTES = 64 * 1024 * 1024
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SECONDS = 300
FETCH_BACKOFF_SECONDS = 5.0
CHUNK_BYTES = 1 << 20
FILE_MODE = 0o644
USER_AGENT = "automatos-media-assets/1 (+https://automatos.app)"

Opener = Callable[..., Any]


def _load_music_build():
    """media-render's music_build.py, by path: the manifest's one validator and file naming."""
    spec = importlib.util.spec_from_file_location("music_build", MUSIC_BUILD)
    module = importlib.util.module_from_spec(spec)
    sys.modules["music_build"] = module  # registered before exec, for its dataclasses
    spec.loader.exec_module(module)
    return module


music = _load_music_build()


class AssetsError(ValueError):
    """A list that cannot be used, or a file that is not the one it pins."""


@dataclass(frozen=True)
class Asset:
    """One file of the image: where it goes, where it comes from, and the bytes it must be."""

    path: str
    url: str
    sha256: str
    size: Optional[int]  # exact for a Kokoro file; a track's manifest gives none


# ── the lists ───────────────────────────────────────────────────────────────
def _value_problems(entry: Dict[str, Any], where: str) -> List[str]:
    problems = []
    path, url = entry["path"], entry["url"]
    parts = path.split("/") if isinstance(path, str) else []
    if len(parts) != 2 or not all(SEGMENT.match(part) for part in parts):
        problems.append(f"{where}: path must be <release>/<file>, letters, digits, '.', '_' and '-' only")
    if not isinstance(url, str) or not url.startswith("https://"):
        problems.append(f"{where}: url must be an https address")
    elif isinstance(path, str) and not url.endswith("/" + path):
        problems.append(f"{where}: url must end with the path, so the release in the path is the url's")
    size = entry["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        problems.append(f"{where}: bytes must be the file's size, a positive whole number")
    if not isinstance(entry["sha256"], str) or not SHA256.match(entry["sha256"]):
        problems.append(f"{where}: sha256 must be 64 lowercase hex characters")
    if not isinstance(entry["use"], str) or not entry["use"].strip():
        problems.append(f"{where}: use must say what the file is for")
    return problems


def _file_problems(entry: Any, index: int) -> List[str]:
    where = f"files[{index}]"
    if not isinstance(entry, dict):
        return [f"{where} must be an object"]
    problems = [f"{where}: {key} is required" for key in FILE_FIELDS if key not in entry]
    problems += [f"{where}: {key} is not a file field ({', '.join(FILE_FIELDS)})" for key in entry if key not in FILE_FIELDS]
    return problems or _value_problems(entry, where)


def _list_problems(data: Dict[str, Any]) -> List[str]:
    problems = [f"{key} is required" for key in LIST_FIELDS if key not in data]
    problems += [f"{key} is not a list field ({', '.join(LIST_FIELDS)})" for key in data if key not in LIST_FIELDS]
    if "licence" in data and data["licence"] != KOKORO_LICENCE:
        problems.append(f"licence must be {KOKORO_LICENCE}, the licence whose text the image ships")
    if "notice" in data and (not isinstance(data["notice"], str) or not data["notice"].strip()):
        problems.append("notice must be the Kokoro licence notice")
    if "files" in data and (not isinstance(data["files"], list) or not data["files"]):
        problems.append("files must be a non-empty list")
    return problems


def _files_problems(files: List[Any]) -> List[str]:
    problems = [problem for index, entry in enumerate(files) for problem in _file_problems(entry, index)]
    paths = [entry.get("path") for entry in files if isinstance(entry, dict)]
    repeated = sorted({path for path in paths if isinstance(path, str) and paths.count(path) > 1})
    return problems + [f"path {path} is listed more than once" for path in repeated]


def validate_kokoro(data: Any) -> Dict[str, Any]:
    """The Kokoro list, or :class:`AssetsError` naming every problem."""
    if not isinstance(data, dict):
        raise AssetsError("the Kokoro list must be a JSON object")
    problems = _list_problems(data)
    if not problems:
        problems = _files_problems(data["files"])
    if problems:
        raise AssetsError("the Kokoro list cannot be used:\n  " + "\n  ".join(problems))
    return data


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetsError(f"{path} cannot be read as JSON: {exc}") from None


def load(kokoro_path: Path, music_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The Kokoro list and the music manifest's tracks, both validated."""
    return validate_kokoro(_read_json(kokoro_path)), music.load_source(music_path)


def plan(kokoro: Dict[str, Any], tracks: Sequence[Dict[str, Any]]) -> List[Asset]:
    """Every fetched file of the image, in layer order: Kokoro as listed, then the tracks."""
    assets = [
        Asset(f"{KOKORO_DIR}/{entry['path']}", entry["url"], entry["sha256"], entry["bytes"])
        for entry in kokoro["files"]
    ]
    assets += [Asset(f"{MUSIC_DIR}/{music.track_file(track)}", track["url"], track["sha256"], None) for track in tracks]
    return assets


def licences(kokoro: Dict[str, Any], tracks: Sequence[Dict[str, Any]]) -> str:
    """The image's SPDX licence expression: every licence it carries, once each, joined with AND."""
    ordered = [kokoro["licence"]] + [track["licence"] for track in tracks]
    return " AND ".join(dict.fromkeys(ordered))


# ── what the image says about itself ────────────────────────────────────────
def notice(kokoro: Dict[str, Any], tracks: Sequence[Dict[str, Any]]) -> str:
    """/NOTICE: the Kokoro notice, each file with its sha256, and every track's attribution line."""
    lines = [
        "Automatos media assets (PRD-262 US-053)",
        "",
        "Data only: the Kokoro voice model files and the music library media-render builds from.",
        "",
        "Kokoro, in /kokoro",
        kokoro["notice"],
        f"Licence: Apache License 2.0, the text in /{APACHE_PATH}.",
    ]
    lines += [f"  /{KOKORO_DIR}/{entry['path']}  sha256 {entry['sha256']}" for entry in kokoro["files"]]
    lines += ["", "Music, in /music"]
    for track in tracks:
        lines += [
            "",
            f"/{MUSIC_DIR}/{music.track_file(track)}",
            track["attribution"],
            f"Licence: {music.LICENCES[track['licence']]['url']}",
            f"Source: {track['url']}  sha256 {track['sha256']}",
        ]
    return "\n".join(lines) + "\n"


def apache_text(license_path: Path) -> str:
    """The Apache License 2.0 terms from the repository's LICENSE, without its appendix and copyright line."""
    text = license_path.read_text(encoding="utf-8")
    end = text.find(APACHE_END)
    head = text[:end] if end >= 0 else ""
    if not all(marker in head for marker in APACHE_HEAD):
        raise AssetsError(f"{license_path} does not hold the Apache License 2.0 terms")
    return head.lstrip("\n") + APACHE_END + "\n"


def dockerfile(assets: Sequence[Asset]) -> str:
    """A data-only image: FROM scratch and one COPY, so one layer, per file."""
    paths = [asset.path for asset in assets] + [APACHE_PATH, NOTICE_PATH]
    lines = ["# Written by scripts/ci/media_assets.py (PRD-262 US-053). Data only: one layer per file.", "FROM scratch"]
    return "\n".join(lines + [f"COPY {path} /{path}" for path in paths]) + "\n"


def expected_files(kokoro: Dict[str, Any], tracks: Sequence[Dict[str, Any]], licence_text: str) -> Dict[str, str]:
    """Every file the image holds, by path, with its sha256."""
    files = {asset.path: asset.sha256 for asset in plan(kokoro, tracks)}
    files[APACHE_PATH] = hashlib.sha256(licence_text.encode("utf-8")).hexdigest()
    files[NOTICE_PATH] = hashlib.sha256(notice(kokoro, tracks).encode("utf-8")).hexdigest()
    return files


# ── fetching and checking ───────────────────────────────────────────────────
def _download(asset: Asset, target: Path, opener: Opener, timeout: float) -> Tuple[str, int]:
    request = urllib.request.Request(asset.url, headers={"User-Agent": USER_AGENT})
    limit = asset.size if asset.size is not None else TRACK_MAX_BYTES
    digest, size = hashlib.sha256(), 0
    with opener(request, timeout=timeout) as response, target.open("wb") as handle:
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            size += len(chunk)
            if size > limit:
                raise AssetsError(f"/{asset.path}: {asset.url} sent more than {limit} bytes")
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest(), size


def _download_with_retries(asset: Asset, target: Path, opener: Opener, attempts: int, backoff: float) -> Tuple[str, int]:
    last: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return _download(asset, target, opener, FETCH_TIMEOUT_SECONDS)
        except (OSError, http.client.HTTPException) as exc:  # URLError and HTTPError are OSErrors
            last = exc
            if attempt < attempts:
                time.sleep(backoff * attempt)
    raise AssetsError(f"/{asset.path}: {asset.url} could not be fetched ({attempts} attempts): {last}")


def check(asset: Asset, digest: str, size: int) -> None:
    """:class:`AssetsError` unless the bytes are the ones the list pins."""
    if asset.size is not None and size != asset.size:
        raise AssetsError(f"/{asset.path}: {size} bytes, where the list pins {asset.size}")
    if digest != asset.sha256:
        raise AssetsError(
            f"/{asset.path}: sha256 mismatch: the list pins {asset.sha256}, {asset.url} sent {digest}. "
            "Either the source changed the file or the list is wrong; nothing is published unchecked."
        )


def fetch(
    asset: Asset,
    target: Path,
    *,
    opener: Optional[Opener] = None,
    attempts: int = FETCH_ATTEMPTS,
    backoff: float = FETCH_BACKOFF_SECONDS,
) -> None:
    """Download ``asset`` to ``target`` and check it; nothing is left at ``target`` when it fails."""
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = False
    try:
        check(asset, *_download_with_retries(asset, target, opener or urllib.request.urlopen, attempts, backoff))
        kept = True
    finally:
        if not kept:
            target.unlink(missing_ok=True)
    os.chmod(target, FILE_MODE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def differences(root: Path, expected: Dict[str, str]) -> List[str]:
    """What keeps ``root`` from being exactly the ``expected`` files: missing, extra or other bytes."""
    found = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink() or path.is_file()}
    problems = [f"/{path}: missing" for path in sorted(set(expected) - found)]
    problems += [f"/{path}: is not one of the image's files" for path in sorted(found - set(expected))]
    for path in sorted(set(expected) & found):
        target = root / path
        actual = "a symlink" if target.is_symlink() else f"sha256 {file_sha256(target)}"
        if actual != f"sha256 {expected[path]}":
            problems.append(f"/{path}: {actual}, where the lists pin sha256 {expected[path]}")
    return problems


# ── the commands ────────────────────────────────────────────────────────────
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, FILE_MODE)


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def stage_tree(
    out: Path, kokoro: Dict[str, Any], tracks: Sequence[Dict[str, Any]], licence_text: str,
    *, opener: Optional[Opener] = None, backoff: float = FETCH_BACKOFF_SECONDS,
) -> List[Asset]:
    """Fetch and check every file into ``out`` and write the notices and the Dockerfile: the build context."""
    if out.exists() and any(out.iterdir()):
        raise AssetsError(f"{out} must be empty: the image is exactly what this run stages")
    assets = plan(kokoro, tracks)
    for asset in assets:
        fetch(asset, out / PurePosixPath(asset.path), opener=opener, backoff=backoff)
        print(f"/{asset.path}: sha256 {asset.sha256} checked", flush=True)
    _write(out / APACHE_PATH, licence_text)
    _write(out / NOTICE_PATH, notice(kokoro, tracks))
    _write(out / DOCKERFILE, dockerfile(assets))
    return assets


def _summary(assets: Sequence[Asset], out: Path, expression: str) -> str:
    rows = [f"| `/{asset.path}` | {(out / asset.path).stat().st_size:,} | `{asset.sha256}` |" for asset in assets]
    head = ["### Media assets (PRD-262 US-053)", "", f"Licences: `{expression}`", "",
            "| File | Bytes | sha256 (checked) |", "|---|---:|---|"]
    return "\n".join(head + rows) + "\n"


def stage(args: argparse.Namespace) -> int:
    kokoro, tracks = load(args.kokoro, args.music)
    assets = stage_tree(args.out, kokoro, tracks, apache_text(args.license))
    expression = licences(kokoro, tracks)
    _append(args.outputs, f"licences={expression}")
    _append(args.summary, _summary(assets, args.out, expression))
    print(f"media assets: {len(assets)} files checked into {args.out}; licences {expression}")
    return 0


def verify(args: argparse.Namespace) -> int:
    kokoro, tracks = load(args.kokoro, args.music)
    expected = expected_files(kokoro, tracks, apache_text(args.license))
    problems = differences(args.dir, expected)
    if problems:
        raise AssetsError(f"{args.dir} is not the image the lists describe:\n  " + "\n  ".join(problems))
    _append(args.summary, f"Read back from the registry: all {len(expected)} files match ({args.dir.name}).")
    print(f"media assets: all {len(expected)} files in {args.dir} match the lists")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Stage or verify the media assets image (PRD-262 US-053).")
    commands = root.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="fetch and check every file into the image's build context")
    staging.add_argument("--out", type=Path, required=True, help="an empty directory: the build context")
    staging.add_argument("--outputs", help="append licences=<expression> here ($GITHUB_OUTPUT)")
    checking = commands.add_parser("verify", help="check a published image's files, exported to a directory")
    checking.add_argument("--dir", type=Path, required=True, help="the image's files, as exported")
    for command in (staging, checking):
        command.add_argument("--kokoro", type=Path, default=KOKORO_LIST)
        command.add_argument("--music", type=Path, default=MUSIC_MANIFEST)
        command.add_argument("--license", type=Path, default=LICENSE)
        command.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    return root


def _annotation(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"stage": stage, "verify": verify}[args.command](args)
    except (AssetsError, music.MusicBuildError) as exc:
        print(f"::error title=media assets::{_annotation(str(exc))}")
        print(f"media assets: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
