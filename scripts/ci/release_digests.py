#!/usr/bin/env python3
"""Each digest images.yml signs is the one its own push made (PRD-279 US-101, US-103, US-202).

The digest a job signs, checks, tags with the version and writes into the chart is read
from the output of the step that pushed it, never from a tag by name. Before it is used,
what it names is read back by that digest, its bytes must hash to it, and it must hold
exactly what this run built. Two commands, each writing ``digest=…`` to ``--outputs``:

* ``index`` (the merge job, every publishing run): the digest of the index that
  ``docker buildx imagetools create --metadata-file`` reports it pushed: the file's
  ``containerimage.descriptor``, the descriptor of the bytes it pushed (buildx v0.32.0
  and later; the merge job pins its buildx). The file's ``image.name`` is not relied on.
  This run's builds are the digests the build jobs pushed and scanned (``--built``: one
  empty file per digest, named by its hex, as they export them). BuildKit pushes each as
  an index of one linux image and its attestation, and ``imagetools create`` lists those
  entries in the new index. So, each read by digest the same way: there is one build per
  architecture the build matrix published for this image (``--matrix``, the build job's
  matrix), and the index's entries are exactly the builds' entries, none twice.
* ``chart`` (the release-chart job): the manifest digest ``helm push`` prints with the
  reference it pushed (``Pushed:`` and ``Digest:`` on its standard output, printed here
  again). The reference must be ``<chart>:<version>``, and the manifest's one layer the
  packaged ``.tgz``, byte for byte (its sha256 and size).

Standard library only, as the other release tools; the registry is read through
``scripts/ci/image_registry.py``. A problem prints ``::error`` and exits 1, so nothing
after the step is signed or published.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.client
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
CHART_LAYER = "application/vnd.cncf.helm.chart.content.v1.tar+gzip"
CHART_CONFIG = "application/vnd.cncf.helm.config.v1+json"
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json")
DESCRIPTOR_KEY = "containerimage.descriptor"
_PUSHED = re.compile(r"^Pushed: (\S+)\s*$", re.M)
_PUSH_DIGEST = re.compile(r"^Digest: (\S+)\s*$", re.M)

# (repository, sha256 digest) -> the bytes the registry serves for that digest.
Fetch = Callable[[str, str], bytes]


class DigestError(ValueError):
    """A pushed digest this run cannot take as its own."""


def _digest(value: object, what: str) -> str:
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise DigestError(f"{what} is not a sha256 digest: {str(value)[:80]!r}")
    return value


def read_by_digest(fetch: Fetch, repository: str, digest: str) -> dict:
    """The manifest stored under ``digest``, read by that digest; its bytes must hash to it."""
    body = fetch(repository, _digest(digest, "a digest to read"))
    served = "sha256:" + hashlib.sha256(body).hexdigest()
    if served != digest:
        raise DigestError(f"{repository}@{digest}: the registry served bytes that hash to {served}")
    manifest = json.loads(body)
    if not isinstance(manifest, dict):
        raise DigestError(f"{repository}@{digest} is not a JSON object")
    return manifest


def _objects(manifest: dict, digest: str) -> List[dict]:
    """An index's entries, each an object."""
    entries = manifest.get("manifests")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise DigestError(f"{digest} is not an index of descriptors")
    return entries


def _entries(manifest: dict, digest: str) -> List[str]:
    return [_digest(entry.get("digest"), f"an entry of {digest}") for entry in _objects(manifest, digest)]


# --------------------------------------------------------------------------- the images


def created_index(metadata: object) -> str:
    """The digest of the index ``imagetools create`` pushed, from its metadata file."""
    descriptor = metadata.get(DESCRIPTOR_KEY) if isinstance(metadata, dict) else None
    if not isinstance(descriptor, dict) or descriptor.get("mediaType") not in INDEX_TYPES:
        raise DigestError(f"the create step's metadata names no pushed index: {str(metadata)[:200]!r}")
    return _digest(descriptor.get("digest"), "the create step's digest")


def built_digests(directory: Path) -> List[str]:
    """This run's per-architecture builds, one file per digest named by its hex."""
    names = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    if not names:
        raise DigestError(f"no build digests in {directory}")
    return [_digest(f"sha256:{name}", f"the build digest file {name!r}") for name in names]


def expected_arches(matrix: object, image: str) -> List[str]:
    """The architectures the build matrix pushed ``image`` for, as published builds."""
    include = matrix.get("include") if isinstance(matrix, dict) else None
    if not isinstance(include, list) or not all(isinstance(entry, dict) for entry in include):
        raise DigestError("the build matrix is not {\"include\": [entries]}")
    arches = sorted(str(entry.get("arch")) for entry in include
                    if entry.get("image") == image and entry.get("publish") is True)
    if not arches:
        raise DigestError(f"the build matrix publishes no build of {image}")
    return arches


def build_arch(build: dict, digest: str) -> str:
    """The one linux architecture a per-arch build holds (its attestation is ``unknown``)."""
    platforms = [entry.get("platform") for entry in _objects(build, digest)]
    arches = [platform.get("architecture") for platform in platforms
              if isinstance(platform, dict) and platform.get("os") == "linux"]
    if len(arches) != 1:
        raise DigestError(f"build {digest} holds {len(arches)} linux images, not one")
    return str(arches[0])


def check_index(fetch: Fetch, image: str, index: str, built: Sequence[str], arches: Sequence[str]) -> List[str]:
    """Every way the index or the builds differ from what this run should publish; empty passes."""
    builds: Dict[str, dict] = {digest: read_by_digest(fetch, image, digest) for digest in built}
    found = sorted(build_arch(build, digest) for digest, build in builds.items())
    problems = [] if found == sorted(arches) else [f"{image}: builds for {found}, not one for each of {sorted(arches)}"]
    held = _entries(read_by_digest(fetch, image, index), index)
    expected = [entry for digest, build in builds.items() for entry in _entries(build, digest)]
    if not held or not expected:
        return problems + [f"{image}@{index}: {len(held)} entries, the builds {len(expected)}; neither may be empty"]
    if len(set(held)) != len(held) or len(set(expected)) != len(expected):
        problems.append(f"{image}@{index}: an entry is listed twice")
    problems += [f"{image}@{index} holds {entry}, which this run did not build"
                 for entry in sorted(set(held) - set(expected))]
    problems += [f"{image}@{index} lacks {entry}, which this run built and scanned"
                 for entry in sorted(set(expected) - set(held))]
    return problems


# --------------------------------------------------------------------------- the chart


def pushed_chart(log: str, ref: str) -> str:
    """The manifest digest ``helm push`` printed, once it says it pushed ``ref``."""
    pushed, digests = _PUSHED.findall(log), _PUSH_DIGEST.findall(log)
    if pushed != [ref]:
        raise DigestError(f"helm push reported pushing {pushed}, not [{ref!r}]")
    if len(digests) != 1:
        raise DigestError(f"helm push printed {len(digests)} digests, not one")
    return _digest(digests[0], "helm push's digest")


def check_chart(manifest: dict, package: bytes) -> List[str]:
    """Every way the pushed chart differs from the package; an empty list passes."""
    want = {"mediaType": CHART_LAYER, "digest": "sha256:" + hashlib.sha256(package).hexdigest(),
            "size": len(package)}
    layers, config = manifest.get("layers"), manifest.get("config")
    if not isinstance(layers, list) or not all(isinstance(layer, dict) for layer in layers):
        return ["the pushed chart's layers are not a list of descriptors"]
    got = [{key: layer.get(key) for key in want} for layer in layers]
    problems = [] if got == [want] else [f"the pushed chart's layers are {got}, not the package {want}"]
    config_type = config.get("mediaType") if isinstance(config, dict) else None
    if config_type != CHART_CONFIG:
        problems.append(f"the pushed chart's config is a {config_type!r}, not {CHART_CONFIG}")
    return problems


# --------------------------------------------------------------------------- commands


@functools.lru_cache(maxsize=1)
def _registry():
    """``scripts/ci/image_registry.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location("image_registry", Path(__file__).resolve().with_name("image_registry.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["image_registry"] = module
    spec.loader.exec_module(module)
    return module


def _registry_fetch(repository: str, digest: str) -> bytes:
    """The bytes the registry serves for ``repository@digest``."""
    registry = _registry()
    session, reference = registry.connect(f"{repository}@{digest}")
    return registry.manifest_bytes(session, reference)


def _finish(problems: List[str], digest: str, args: argparse.Namespace, line: str) -> int:
    for problem in problems:
        print(f"::error title=Release digest::{problem}")
    if problems:
        return 1
    for path, text in ((args.outputs, f"digest={digest}\n"), (args.summary, line + "\n")):
        if path:
            with open(path, "a", encoding="utf-8") as out:
                out.write(text)
    print(line)
    return 0


def cmd_index(args: argparse.Namespace, fetch: Fetch) -> int:
    index = created_index(json.loads(Path(args.metadata).read_text(encoding="utf-8")))
    arches = expected_arches(json.loads(args.matrix), args.image)
    built = built_digests(Path(args.built))
    problems = check_index(fetch, args.image, index, built, arches)
    return _finish(problems, index, args, f"Published {args.image}@{index}, holding this run's scanned "
                                          f"builds for {', '.join(arches)}")


def cmd_chart(args: argparse.Namespace, fetch: Fetch) -> int:
    ref = f"{args.chart}:{args.version}"
    log = Path(args.push_log).read_text(encoding="utf-8")
    print(f"helm push printed:\n{log}")
    digest = pushed_chart(log, ref)
    problems = check_chart(read_by_digest(fetch, args.chart, digest), Path(args.package).read_bytes())
    return _finish(problems, digest, args, f"Pushed {ref} as {digest}, the packaged chart byte for byte")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="the index digest from imagetools create's metadata file")
    chart = commands.add_parser("chart", help="the chart digest from helm push's output")
    for command, flags in ((index, ("--metadata", "--built", "--image", "--matrix")),
                           (chart, ("--push-log", "--chart", "--version", "--package"))):
        for flag in flags:
            command.add_argument(flag, required=True)
        command.add_argument("--outputs")
        command.add_argument("--summary")
    return root


def main(argv: Optional[List[str]] = None, fetch: Optional[Fetch] = None) -> int:
    args = parser().parse_args(argv)
    command = {"index": cmd_index, "chart": cmd_chart}[args.command]
    try:
        return command(args, fetch or _registry_fetch)
    except (DigestError, OSError, ValueError, LookupError, http.client.HTTPException) as error:
        print(f"::error title=Release digest::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
