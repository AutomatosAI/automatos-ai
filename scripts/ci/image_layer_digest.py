#!/usr/bin/env python3
"""Which layers two local images share, by the COPY that made them (PRD-262 US-053).

media-render's Dockerfile builds each part in its own stage and the runtime
``COPY --link``s it in, so a change to one part leaves the other parts' layers,
and their digests, as they were. media-render.yml proves it: it builds the image
a second time with only the Hyperframes pin changed, then runs

    python3 scripts/ci/image_layer_digest.py IMAGE_A IMAGE_B --same /opt/kokoro --differ /opt/media-render/js

``--same DEST`` requires the layer that ``COPY … DEST`` made to have the same
digest in both images; ``--differ DEST`` requires it to differ, which shows the
second build really changed what it was meant to. Each image is read from
``docker save``: its config's history entries that made a layer, in order,
paired with ``rootfs.diff_ids`` (the uncompressed digests a registry push keeps).
Exit 1 on any check that fails, or a COPY that is not found exactly once.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


layers = _sibling("image_layers")

# The manifest and the config are small; the layers are not, and are skipped.
SMALL_MEMBER_BYTES = 4 * 1024 * 1024
MANIFEST = "manifest.json"
SAVE_TIMEOUT_SECONDS = 600


class LayerError(ValueError):
    """An image this check cannot read, or a COPY it cannot find exactly once."""


def _entry(entries: List[dict], tag: Optional[str]) -> dict:
    """The manifest.json entry of the image saved: the only one, or the one tagged ``tag``."""
    if len(entries) > 1 and tag:
        # A registry host may be spelt out: docker.io/library/media-render:ci for media-render:ci.
        entries = [entry for entry in entries if any(t == tag or t.endswith("/" + tag) for t in entry.get("RepoTags") or [])]
    if len(entries) != 1:
        raise LayerError(f"the archive holds {len(entries)} matching images; save one at a time")
    return entries[0]


def read_config(stream: IO[bytes], tag: Optional[str] = None) -> dict:
    """The image config inside a ``docker save`` tar stream, read in one pass."""
    small: Dict[str, bytes] = {}
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            if member.isfile() and member.size <= SMALL_MEMBER_BYTES:
                small[member.name] = tar.extractfile(member).read()
    if MANIFEST not in small:
        raise LayerError("the archive has no manifest.json: not a docker save archive")
    entry = _entry(json.loads(small[MANIFEST]), tag)
    config = small.get(entry["Config"])
    if config is None:
        raise LayerError(f"the archive has no config at {entry['Config']}")
    return json.loads(config)


def copy_destination(instruction: str) -> Optional[str]:
    """Where a COPY instruction writes (its last word, without a trailing slash); None for anything else."""
    words = instruction.split()
    if len(words) < 3 or words[0] != "COPY":
        return None
    return words[-1].rstrip("/") or "/"


def layer_by_destination(config: dict, destination: str) -> str:
    """The diff id of the one layer whose COPY wrote ``destination``."""
    history = [entry for entry in config.get("history", []) if not entry.get("empty_layer")]
    diff_ids = config.get("rootfs", {}).get("diff_ids", [])
    if len(history) != len(diff_ids):
        raise LayerError(f"{len(history)} history entries made a layer, but the image has {len(diff_ids)} layers")
    wanted = destination.rstrip("/") or "/"
    found = [diff for entry, diff in zip(history, diff_ids) if copy_destination(layers.instruction(entry)) == wanted]
    if len(found) != 1:
        raise LayerError(f"{len(found)} layers were made by a COPY to {destination}; expected exactly one")
    return found[0]


def compare(first: dict, second: dict, same: Sequence[str], differ: Sequence[str]) -> List[str]:
    """One line per check, ``PASS`` or ``FAIL``, for the layers both configs name."""
    lines = []
    for destination, want_same in [(d, True) for d in same] + [(d, False) for d in differ]:
        a, b = layer_by_destination(first, destination), layer_by_destination(second, destination)
        verdict = "PASS" if (a == b) == want_same else "FAIL"
        wanted = "the same layer" if want_same else "different layers"
        lines.append(f"{verdict}  {destination}: {wanted} expected; {a} and {b}")
    return lines


def saved_config(image: str) -> dict:
    """``docker save IMAGE``, streamed: its config."""
    with subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE) as proc:
        config = read_config(proc.stdout, image)
        proc.stdout.read()  # drain what the tar reader left, so docker save can finish
        code = proc.wait(timeout=SAVE_TIMEOUT_SECONDS)
    if code != 0:
        raise LayerError(f"docker save {image} exited {code}")
    return config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--same", action="append", default=[], help="a COPY destination whose layer must match")
    parser.add_argument("--differ", action="append", default=[], help="a COPY destination whose layer must differ")
    args = parser.parse_args(argv)
    if not args.same and not args.differ:
        parser.error("name at least one --same or --differ destination")
    try:
        lines = compare(saved_config(args.first), saved_config(args.second), args.same, args.differ)
    except LayerError as exc:
        print(f"FAIL  {exc}")
        return 1
    print("\n".join(lines))
    return 1 if any(line.startswith("FAIL") for line in lines) else 0


if __name__ == "__main__":
    sys.exit(main())
