"""PRD-262 US-022: the API's pip layers stay what edge has unless what made them moved.

orchestrator/Dockerfile's pybuild installs the API's lock in two halves, which the base
stage copies as two layers: ``COPY /install/stable /usr/local`` (the stable set
orchestrator/scripts/split_lock.py names) and ``COPY /install/rest /usr/local``.
images.yml builds the API alone reproducibly (a fixed SOURCE_DATE_EPOCH and
``rewrite-timestamp=true``), so a rebuild from the same inputs makes the same bytes. On
a push to main (never on the weekly schedule run), after ``image_report.py measure``,
this compares each layer with the published ``edge``'s:

- pybuild's inputs, at this commit and at edge's revision (its
  ``org.opencontainers.image.revision``): the uv and pybuild stages' instructions (the
  base and uv digests included, comments not) and split_lock.py. If either moved, no
  layer is expected to match, and the summary says so.
- each half of requirements.lock (the lock the published build installs), split with
  split_lock.py, as pins and hashes (comments aside). A half that did not move must
  give edge's layer digest. A half that moved gives a new layer: US-001's report counts
  what it costs an update, and the summary names it.
- a layer that should match and does not: pybuild's apt layer (binutils' strip, for
  one) or BuildKit's compression can move the bytes, and the image records neither. So
  the check then lists the dist-info directories (name==version) of both layers: the
  same set is a warning that the bytes moved, a different set fails.

The workflow step fetches edge's three files at its revision into ``--edge-dir``; one
that revision lacks (an edge built before the split) skips the check with a notice.

    image_pip_layers.py --report LAYERS.json --edge-dir DIR [--summary FILE]

Exit 0 when every layer is as expected, 1 otherwise. Standard library only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = "orchestrator/Dockerfile"
LOCK = "orchestrator/requirements.lock"
SPLIT = "orchestrator/scripts/split_lock.py"
EDGE_FILES = (DOCKERFILE, LOCK, SPLIT)
HALVES = {"stable": "COPY /install/stable /usr/local", "rest": "COPY /install/rest /usr/local"}
FIRST_RUNTIME_STAGE = "base"
LISTED_DIFFERENCES = 8
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_FROM_BASE = re.compile(rf"^FROM\s+\S+\s+AS\s+{FIRST_RUNTIME_STAGE}$", re.I)
_DIST_INFO = re.compile(r"(?:^|/)site-packages/([^/]+)\.dist-info/")


def _load(name: str, path: Path):
    """A module loaded by path and registered first, as the tests load these scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split_lock = _load("split_lock", ROOT / SPLIT)
layers = _load("image_layers", Path(__file__).with_name("image_layers.py"))
registry = _load("image_registry", Path(__file__).with_name("image_registry.py"))


class Inputs(NamedTuple):
    """What pybuild's output depends on, at one revision."""

    pybuild: Tuple[str, ...]
    split_script: str
    halves: Dict[str, Tuple[str, ...]]


class Verdict(NamedTuple):
    """``level`` is ``ok``, ``notice``, ``warning`` or ``error``; ``message`` says why."""

    level: str
    message: str


# --------------------------------------------------------------------------- inputs (pure)


def pybuild_instructions(dockerfile: str) -> Tuple[str, ...]:
    """Every instruction before the base stage's FROM, comments dropped and spaces collapsed."""
    code = "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#"))
    instructions = [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]
    end = next((i for i, line in enumerate(instructions) if _FROM_BASE.match(line)), None)
    if end is None:
        raise ValueError(f"no `FROM … AS {FIRST_RUNTIME_STAGE}` in the Dockerfile")
    return tuple(instructions[:end])


def _pins(text: str) -> Tuple[str, ...]:
    """Each block of a split half as its pin and hash lines, comments aside, sorted."""
    found = []
    for block in split_lock.blocks(text):
        lines = (line.strip().rstrip("\\").strip() for line in block.text.splitlines())
        found.append(" ".join(line for line in lines if line and not line.startswith("#")))
    return tuple(sorted(found))


def inputs(dockerfile: str, lock: str, split_script: str) -> Inputs:
    """pybuild's inputs from the three files' texts, the lock split with this split_lock.py."""
    stable, rest = split_lock.split(lock)
    return Inputs(pybuild_instructions(dockerfile), split_script, {"stable": _pins(stable), "rest": _pins(rest)})


def _layer(rows: List[dict], half: str) -> Optional[dict]:
    found = [row for row in rows if row.get("instruction") == HALVES[half]]
    if len(found) > 1:
        raise ValueError(f"{len(found)} layers were made by `{HALVES[half]}`")
    return found[0] if found else None


def expected(report: dict, now: Inputs, edge: Optional[Inputs]) -> Dict[str, Verdict]:
    """Per half: ``ok`` when its digest must and does equal edge's, ``notice`` when no match
    is expected (and why), ``error`` when it must match and does not. Pure; the dist-info
    comparison turns an ``error`` into a warning or keeps it."""
    if edge is None:
        return {half: Verdict("notice", "edge's revision has no split_lock.py or lock (it predates the "
                                        "split), or this split_lock.py cannot split its lock")
                for half in HALVES}
    if (now.pybuild, now.split_script) != (edge.pybuild, edge.split_script):
        return {half: Verdict("notice", "pybuild or split_lock.py changed since edge's revision, so the "
                                        "layer is new") for half in HALVES}
    verdicts = {}
    for half in HALVES:
        mine, theirs = _layer(report["layers"], half), _layer(report["edge"]["layers"], half)
        verdicts[half] = _verdict(half, mine, theirs, now.halves[half] == edge.halves[half])
    return verdicts


def _verdict(half: str, mine: Optional[dict], theirs: Optional[dict], same_pins: bool) -> Verdict:
    if mine is None:
        return Verdict("error", f"this build has no `{HALVES[half]}` layer")
    if theirs is None:
        return Verdict("notice", f"edge has no `{HALVES[half]}` layer: it predates the split")
    size = layers.mb(mine["size"])
    if not same_pins:
        return Verdict("notice", f"the lock's {half} half changed since edge, so the layer is new: "
                                 f"an update downloads its {size}")
    if mine["digest"] == theirs["digest"]:
        return Verdict("ok", f"the lock's {half} half and pybuild are edge's, and so is the layer ({size})")
    return Verdict("error", f"the lock's {half} half and pybuild are edge's, but the layer is not "
                            f"({mine['digest'][:19]}, edge {theirs['digest'][:19]})")


def distributions(names: List[str]) -> FrozenSet[str]:
    """``name==version`` for every dist-info directory among a layer's paths. Pure."""
    found = set()
    for name in names:
        match = _DIST_INFO.search(name)
        if match:
            distribution, _, version = match.group(1).rpartition("-")
            found.add(f"{re.sub(r'[-_.]+', '-', distribution).lower()}=={version}")
    return frozenset(found)


def same_distributions(mine: FrozenSet[str], theirs: FrozenSet[str]) -> Verdict:
    """A digest mismatch judged by what the two layers install. Pure."""
    if mine == theirs:
        return Verdict("warning", f"the same {len(mine)} distributions, so pybuild's tools (its apt layer, "
                                  "binutils' strip) or BuildKit's compression moved the bytes")
    added, gone = sorted(mine - theirs), sorted(theirs - mine)
    return Verdict("error", f"the distributions differ: new {', '.join(added[:LISTED_DIFFERENCES]) or 'none'}; "
                            f"gone {', '.join(gone[:LISTED_DIFFERENCES]) or 'none'}")


# --------------------------------------------------------------------------- registry and files


def layer_paths(ref: str, digest: str, media_type: str) -> List[str]:
    """Every path in one layer of ``ref``'s repository, read from the registry."""
    mode = layers.TAR_MODES.get(layers.compression(media_type) or "")
    if mode is None:
        raise ValueError(f"cannot list a {media_type} layer")
    session, _ = registry.connect(ref)
    with registry.blob(session, digest) as stream, tarfile.open(fileobj=stream, mode=mode) as tar:
        return [member.name for member in tar]


def _judge_bytes(report: dict, half: str) -> Verdict:
    """A layer that should be edge's and is not, judged by the distributions both install."""
    mine, theirs = _layer(report["layers"], half), _layer(report["edge"]["layers"], half)
    try:
        ours = distributions(layer_paths(f"{report['image']}@{report['digest']}", mine["digest"],
                                         mine.get("media_type", "")))
        edges = distributions(layer_paths(report["edge"]["ref"], theirs["digest"], theirs.get("media_type", "")))
    except (*registry.READ_ERRORS, tarfile.TarError, EOFError) as error:
        return Verdict("error", f"the layer is not edge's, and its distributions could not be read: {error}")
    return same_distributions(ours, edges)


def judge(report: dict, now: Inputs, edge: Optional[Inputs]) -> Dict[str, Verdict]:
    """Each half's verdict; a digest mismatch where a match was due is judged by its distributions."""
    if not report.get("edge"):
        return {half: Verdict("notice", f"no edge to compare with: {report.get('edge_note', '')}")
                for half in HALVES}
    verdicts = expected(report, now, edge)
    return {half: _judge_bytes(report, half) if verdict.level == "error" and _layer(report["layers"], half)
            else verdict for half, verdict in verdicts.items()}


def read_inputs(root: Path) -> Optional[Inputs]:
    """pybuild's inputs from the three files under ``root``; None when one is missing."""
    paths = [root / path for path in EDGE_FILES]
    if not all(path.is_file() for path in paths):
        return None
    dockerfile, lock, split_script = (path.read_text(encoding="utf-8") for path in paths)
    return inputs(dockerfile, lock, split_script)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, help="image_report.py measure's JSON")
    parser.add_argument("--edge-dir", required=True, help="edge's revision of the three files")
    parser.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    return finish(report, judge(report, read_inputs(ROOT), _edge_inputs(Path(args.edge_dir))), args.summary)


def _edge_inputs(edge_dir: Path) -> Optional[Inputs]:
    """edge's pybuild inputs, or None when its files are missing or its lock does not split
    (a stable package it never pinned): no layer is then expected to match."""
    try:
        return read_inputs(edge_dir)
    except ValueError as error:
        print(f"::notice title=Pip layers::edge's lock does not split with this split_lock.py: {error}")
        return None


def finish(report: dict, verdicts: Dict[str, Verdict], summary: Optional[str]) -> int:
    """Annotations, the summary section, and the exit code."""
    lines = [f"**Pip layers vs `edge`** ({report['name']}, {report['arch']}, PRD-262 US-022):", ""]
    for half, verdict in verdicts.items():
        if verdict.level in ("notice", "warning", "error"):
            print(f"::{verdict.level} title=Pip layer {half}::{verdict.message}")
        lines.append(f"- `{HALVES[half]}`: {verdict.level}: {verdict.message}.")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n\n")
    return 1 if any(verdict.level == "error" for verdict in verdicts.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
