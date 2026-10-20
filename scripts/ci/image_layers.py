"""What an image weighs and what an update costs, from its manifest (PRD-262 US-001).

Pure functions over registry JSON, so the tests feed them recorded manifests:

* ``layer_rows`` pairs each layer of an image manifest with the Dockerfile
  instruction that made it (the config's history entries that are not
  ``empty_layer``, in order) and a kind: ``system`` (OS and runtime packages),
  ``dependencies`` (language packages) or ``code`` (everything else).
* ``update_delta`` is what an existing user downloads to move from ``edge`` to this
  build: the layers ``edge`` does not have, split by kind, so a requirements
  change is reported apart from a code-only change.
* ``shared_layers`` and ``unique_bytes`` say what two or more images share, and
  what pulling them together costs.
* ``budget_verdict`` checks a compressed size against ``.github/image-budgets.json``,
  and ``on_disk_verdict`` an image's on-disk size against its ``on_disk_max_mb``.
* ``list_paths`` sums one layer's files by path prefix, from its tar stream.

Sizes are compressed bytes as the registry reports them; MB means 1,000,000 bytes,
the unit PRD-262 §8 uses.
"""
from __future__ import annotations

import re
import tarfile
from collections import Counter
from typing import IO, Dict, Iterable, List, NamedTuple, Optional

MB = 1_000_000
# A layer this small holds only directory or symlink entries (WORKDIR, ln -s):
# it costs nothing to pull, so the shared-layer count leaves it out.
TRIVIAL_LAYER_BYTES = 1024
INSTRUCTION_CHARS = 110
PATH_DEPTH = 5  # every prefix at this depth is kept: site-packages is depth 5
HEAVY_DEPTH = 6  # one level deeper, the heaviest only: one entry per Python package
HEAVY_COUNT = 25
UNKNOWN = "unknown"
KINDS = ("system", "dependencies", "code")
KIND_PATTERNS = (
    ("system", re.compile(
        r"^ADD \S*rootfs|^# debian\.sh|\bapt-get (?:-\S+ )*install\b|\bapk (?:--\S+ )*add\b")),
    ("dependencies", re.compile(
        r"\b(?:pip3?|uv pip|uv) (?:install|sync)\b|\b(?:npm|pnpm|yarn) (?:install|ci|add)\b"
        r"|\bplaywright install\b"
        r"|^COPY (?:--\S+ )*\S*(?:/install|requirements|package(?:-lock)?\.json|node_modules)\b")),
)
# ``RUN |<n> NAME=value ... /bin/sh -c ``: a build-arg name stops at its first
# ``=``, so each ``NAME=value `` token matches one way only (no backtracking blow-up).
RUN_PREFIX = re.compile(r"^RUN (?:\|\d+ (?:[^\s=]+=\S* )*)?/bin/(?:ba)?sh -c ")
TAR_MODES = {"gzip": "r|gz", "tar": "r|"}


class Verdict(NamedTuple):
    """``level`` is ``ok``, ``warning`` or ``error``; ``message`` says why."""

    level: str
    message: str


def mb(size: int) -> str:
    """``520480102`` -> ``"520.5 MB"``."""
    return f"{size / MB:,.1f} MB"


def instruction(entry: dict) -> str:
    """The history entry's command as the Dockerfile wrote it, on one line."""
    text = " ".join(entry.get("created_by", "").split())
    return RUN_PREFIX.sub("RUN ", text).removesuffix(" # buildkit")


def shorten(text: str) -> str:
    """``text`` cut to INSTRUCTION_CHARS for a report table."""
    return text if len(text) <= INSTRUCTION_CHARS else text[: INSTRUCTION_CHARS - 1] + "…"


def kind_of(text: str) -> str:
    """``system``, ``dependencies`` or ``code`` for one instruction (first match wins)."""
    for kind, pattern in KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return "code"


def layer_rows(manifest: dict, config: dict) -> List[dict]:
    """One row per layer: digest, size, media type, instruction and kind."""
    history = [entry for entry in config.get("history", []) if not entry.get("empty_layer")]
    layers = manifest["layers"]
    matched = len(history) == len(layers)
    rows = []
    for position, layer in enumerate(layers):
        text = instruction(history[position]) if matched else ""
        rows.append({
            "digest": layer["digest"],
            "size": layer["size"],
            "media_type": layer.get("mediaType", ""),
            "instruction": shorten(text) if matched else "(no matching history entry)",
            "kind": kind_of(text) if matched else UNKNOWN,
        })
    return rows


def change_kind(by_kind: Dict[str, int]) -> str:
    """How to file an update: dependencies beat system beat code-only."""
    for kind, label in (("dependencies", "dependencies"), ("system", "system"),
                        (UNKNOWN, "unclassified"), ("code", "code-only")):
        if by_kind.get(kind):
            return label
    return "unchanged"


def update_delta(rows: List[dict], edge_digests: Iterable[str]) -> dict:
    """The layers ``edge`` lacks: total bytes, count, bytes per kind, and the change kind."""
    have = set(edge_digests)
    fresh = {row["digest"]: row for row in rows if row["digest"] not in have}
    by_kind = Counter({kind: 0 for kind in KINDS})
    for row in fresh.values():
        by_kind[row["kind"]] += row["size"]
    return {
        "bytes": sum(row["size"] for row in fresh.values()),
        "layers": len(fresh),
        "by_kind": dict(by_kind),
        "change": change_kind(by_kind),
        "new_digests": sorted(fresh),
    }


def shared_layers(first: List[dict], second: List[dict]) -> dict:
    """Layers both images carry: the count and bytes of those over TRIVIAL_LAYER_BYTES."""
    second_digests = {layer["digest"] for layer in second}
    both = {layer["digest"]: layer["size"] for layer in first if layer["digest"] in second_digests}
    real = {digest: size for digest, size in both.items() if size >= TRIVIAL_LAYER_BYTES}
    return {
        "layers": len(real),
        "bytes": sum(real.values()),
        "digests": sorted(real),
        "trivial_layers": len(both) - len(real),
    }


def unique_bytes(layer_lists: Iterable[List[dict]]) -> int:
    """What pulling all the images costs: each distinct layer counted once."""
    sizes = {}
    for layers in layer_lists:
        for layer in layers:
            sizes[layer["digest"]] = layer["size"]
    return sum(sizes.values())


def _failing(budgets: dict) -> str:
    """The level a finding gets: ``error`` in mode ``fail``, ``warning`` otherwise."""
    return "error" if budgets.get("mode") == "fail" else "warning"


def _against(label: str, size: int, limit: int, failing: str) -> Verdict:
    if size <= limit:
        return Verdict("ok", f"{label} is {mb(size)}, within its {mb(limit)} budget")
    return Verdict(failing, f"{label} is {mb(size)}, {mb(size - limit)} over its {mb(limit)} budget")


def budget_verdict(name: str, arch: str, size: int, budgets: dict) -> Verdict:
    """``size`` against ``budgets["budgets"][name][arch]["max_mb"]``, in the file's mode.

    Mode ``warn`` (report only, until PRD-262 US-004) turns every finding into a
    warning; mode ``fail`` makes it an error.
    """
    failing = _failing(budgets)
    entry = budgets.get("budgets", {}).get(name, {}).get(arch)
    if not entry:
        return Verdict(failing, f"{name} ({arch}) has no budget in .github/image-budgets.json")
    return _against(f"{name} ({arch})", size, round(entry["max_mb"] * MB), failing)


def on_disk_verdict(name: str, arch: str, size: Optional[int], budgets: dict) -> Optional[Verdict]:
    """``size`` on disk against the entry's ``on_disk_max_mb``, in the file's mode (PRD-262 US-054).

    ``size`` is the Size ``docker image inspect`` reports, or None when the build did
    not measure it. §8 bars media-render on disk, because it is never published. An
    entry with an on-disk bar and no measured size is a finding, so a build that
    stopped measuring cannot pass unseen. None when there is neither a bar nor a size.
    """
    entry = budgets.get("budgets", {}).get(name, {}).get(arch) or {}
    label = f"{name} ({arch}) on disk"
    if "on_disk_max_mb" not in entry:
        return None if size is None else Verdict("ok", f"{label} is {mb(size)}, with no on-disk budget")
    if size is None:
        return Verdict(_failing(budgets), f"{label} was not measured, but it has an on-disk budget")
    return _against(label, size, round(entry["on_disk_max_mb"] * MB), _failing(budgets))


def compression(media_type: str) -> Optional[str]:
    """``gzip``, ``tar`` (uncompressed) or None (a compression this reader can't open)."""
    if media_type.endswith(("+gzip", ".gzip")):
        return "gzip"
    if media_type.endswith((".tar", "layer.v1.tar")):
        return "tar"
    return None


def _prefix(name: str, depth: int) -> str:
    return "/".join(name.split("/")[:depth])


def list_paths(stream: IO[bytes], media_type: str) -> dict:
    """Files, unpacked bytes, whiteouts and bytes per path prefix in one layer.

    ``paths`` holds every prefix of up to PATH_DEPTH components as ``[bytes,
    files]``; ``heaviest`` the HEAVY_COUNT largest prefixes at HEAVY_DEPTH.
    """
    mode = TAR_MODES.get(compression(media_type) or "")
    if mode is None:
        raise ValueError(f"cannot list a {media_type} layer")
    paths, heavy, counts = Counter(), Counter(), Counter()
    files = Counter()
    with tarfile.open(fileobj=stream, mode=mode) as tar:
        for member in tar:
            name = member.name.removeprefix("./").lstrip("/")
            if member.isdir() or not name:
                continue
            if name.rsplit("/", 1)[-1].startswith(".wh."):
                counts["whiteouts"] += 1
                continue
            size = member.size if member.isfile() else 0
            counts["files"] += 1
            counts["bytes"] += size
            paths[_prefix(name, PATH_DEPTH)] += size
            files[_prefix(name, PATH_DEPTH)] += 1
            heavy[_prefix(name, HEAVY_DEPTH)] += size
    return {
        "files": counts["files"],
        "bytes": counts["bytes"],
        "whiteouts": counts["whiteouts"],
        "paths": {prefix: [paths[prefix], files[prefix]] for prefix in sorted(paths)},
        "heaviest": [[prefix, size] for prefix, size in heavy.most_common(HEAVY_COUNT)],
    }
