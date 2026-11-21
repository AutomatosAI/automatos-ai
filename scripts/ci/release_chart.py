#!/usr/bin/env python3
"""The chart a release publishes, naming every image by digest (PRD-279 US-201, US-202).

images.yml's ``release-chart`` job runs this on a ``v*`` tag, after the merge job has
published and signed each image. Three commands:

* ``prepare`` copies ``charts/automatos`` and writes, into the copy's ``values.yaml``,
  each published image's repository and the digest of the index the merge job signed.
  Only those lines change, comments included, and the result is read back as YAML and
  must equal the original with exactly those values set. The repository's own chart is
  never touched: it keeps ``appVersion: "edge"``. ``helm package --version
  --app-version`` then stamps the copy with the release's version. It also writes, for
  each published edition, the images that edition's render must run (``--images``) and
  the values file its render is given (``--render-values``, one ``<edition>.yaml`` each).
* ``check-rendered`` reads ``helm template`` of the packaged chart once per published
  edition (``<edition>.yaml`` in ``--rendered``), each with ``existingSecret`` and that
  edition's values file only, and fails if an edition's render is missing, if any image
  is named by a tag, if an image that edition should run is missing or another one
  appears, or if the chart's labels do not carry the release's version.
* ``digest`` prints the digest a pushed chart's tag resolves to, so it is signed and
  checked by digest, never by tag.
* ``remove`` takes back a chart version whose signature was not made or did not check
  (``helm push`` tags it before it can be signed). It deletes that version of the
  ``charts/automatos`` package through GitHub's packages API with the job's built-in
  token, which may delete a version of a package the repository's workflow published
  (the repository holds the package's admin role by default). When the version is the
  package's only one, GitHub deletes the package instead. If GitHub refuses, it says
  exactly what a maintainer must delete by hand.

The web app has one published image per edition (``WEB_EDITIONS``), each in a slot of
its own that the chart's templates pick by ``edition.authEdition`` (PRD-277 US-505):
``frontend.image`` for ``local``, ``frontend.enterpriseImage`` for ``enterprise``. Every
slot is pinned, so the packaged chart names one web image digest per published edition,
whichever edition an install picks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import yaml

KeyPath = Tuple[str, ...]

# The chart's values path for each published image every edition runs, other than the web app.
SLOTS: Mapping[KeyPath, str] = {("api", "image"): "api", ("worker", "image"): "workspace-worker"}
# Each published edition's web image, and the values path of its slot (PRD-277 US-505).
WEB_EDITIONS: Mapping[str, Tuple[KeyPath, str]] = {
    "local": (("frontend", "image"), "frontend"),
    "enterprise": (("frontend", "enterpriseImage"), "enterprise-frontend"),
}
# What each edition's render is given besides existingSecret: the least it renders with.
# Made-up values of the right shape (example.com is reserved for examples), never an install's.
RENDER_VALUES: Mapping[str, dict] = {
    "local": {},
    # PRD-277 US-701: the values this edition cannot render without.
    "enterprise": {"edition": {"authEdition": "enterprise", "enterprise": {
        "publicUrl": "https://automatos.example.com", "clientIpHeader": "X-Real-IP",
        "oidc": {"issuer": "https://id.example.com/realms/automatos", "clientId": "automatos"}}}},
}
CHART_NAME = "automatos"
PACKAGE = "charts/automatos"
GITHUB_API = "https://api.github.com"
API_TIMEOUT_SECONDS = 30
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REPOSITORY = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+$")
_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z0-9_-]+):(?P<rest>[^\n]*)(?P<newline>\n?)$")
_COMMENT = re.compile(r"\s+#.*$")


class ChartError(ValueError):
    """A chart, a digest or a rendered manifest a release cannot be published with."""


# --------------------------------------------------------------------------- the images


def read_digests(directory: Path, names: Sequence[str]) -> Dict[str, str]:
    """``{image name: sha256:…}`` from one file per image, as the merge job writes them."""
    digests, problems = {}, []
    for name in names:
        path = directory / name
        value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        if not DIGEST.match(value):
            problems.append(f"{name}: no sha256 digest at {path.name} (got {value[:80]!r})")
        digests[name] = value
    if problems:
        raise ChartError("; ".join(problems))
    return digests


def published_names() -> List[str]:
    """Every image the chart runs in some edition, as the merge job names its digest file."""
    return sorted({*SLOTS.values(), *(name for _, name in WEB_EDITIONS.values())})


def plan(editions: Mapping[str, Tuple[KeyPath, str]] = WEB_EDITIONS) -> Dict[KeyPath, str]:
    """``{chart values path of an image: published image name}`` for every slot the chart has."""
    return {**SLOTS, **{slot: name for slot, name in editions.values()}}


def edition_images(edition: str, editions: Mapping[str, Tuple[KeyPath, str]] = WEB_EDITIONS) -> List[str]:
    """The published images one edition's install runs: every shared one and its own web image."""
    if edition not in editions:
        raise ChartError(f"the {edition!r} edition has no published web image")
    return sorted({*SLOTS.values(), editions[edition][1]})


def settings(slots: Mapping[KeyPath, str], digests: Mapping[str, str], prefix: str) -> Dict[KeyPath, str]:
    """The values to write: each slot's repository and digest."""
    written = {}
    for slot, name in slots.items():
        repository = f"{prefix}{name}"
        if not REPOSITORY.match(repository):
            raise ChartError(f"{repository!r} is not a lower-case registry repository")
        written[(*slot, "repository")] = repository
        written[(*slot, "digest")] = digests[name]
    return written


def check_distinct_web_digests(digests: Mapping[str, str],
                               editions: Mapping[str, Tuple[KeyPath, str]] = WEB_EDITIONS) -> None:
    """Refuse a release whose web image slots share a digest: each web image is built for one
    edition, so one build in two slots (the local image in the enterprise slot) runs the wrong
    edition's web app."""
    by_digest: Dict[str, List[str]] = {}
    for edition in sorted(editions):
        slot, name = editions[edition]
        by_digest.setdefault(digests[name], []).append(f"{'.'.join(slot)} ({edition})")
    shared = [slots for slots in by_digest.values() if len(slots) > 1]
    if shared:
        raise ChartError("; ".join(f"one digest in {' and '.join(slots)}: each web image is built for one edition"
                                   for slots in shared))


def references(digests: Mapping[str, str], prefix: str) -> Dict[str, List[str]]:
    """``{edition: the image references its render must run}``, for every published edition."""
    return {edition: [f"{prefix}{name}@{digests[name]}" for name in edition_images(edition)]
            for edition in sorted(WEB_EDITIONS)}


# --------------------------------------------------------------------------- values.yaml


def _replaced(match: re.Match, value: str) -> str:
    rest = match["rest"]
    if not _COMMENT.sub("", rest).strip():
        raise ChartError(f"{match['key']} is a mapping, not a value")
    comment = _COMMENT.search(rest)
    return f'{match["indent"]}{match["key"]}: "{value}"{comment.group(0) if comment else ""}{match["newline"]}'


def _key_paths(lines: Sequence[str]) -> Iterator[Tuple[Optional[KeyPath], Optional[re.Match]]]:
    """Each line's key path (by indentation) and its match; (None, None) for other lines."""
    stack: Tuple[Tuple[int, str], ...] = ()
    for line in lines:
        match = _KEY_LINE.match(line)
        if not match:
            yield None, None
            continue
        indent = len(match["indent"])
        stack = tuple(entry for entry in stack if entry[0] < indent) + ((indent, match["key"]),)
        yield tuple(key for _, key in stack), match


def _with(data: dict, path: KeyPath, value: str) -> dict:
    head, rest = path[0], path[1:]
    return {**data, head: _with(data[head], rest, value) if rest else value}


def set_values(text: str, written: Mapping[KeyPath, str]) -> str:
    """``text`` with each path's value replaced on its own line, comments kept."""
    lines = text.splitlines(keepends=True)
    counts = {path: 0 for path in written}
    out = []
    for line, (path, match) in zip(lines, _key_paths(lines)):
        if path in written:
            counts[path] += 1
            line = _replaced(match, written[path])
        out.append(line)
    unmatched = [".".join(path) for path, count in counts.items() if count != 1]
    if unmatched:
        raise ChartError(f"values.yaml does not hold each of these exactly once: {', '.join(unmatched)}")
    result = "".join(out)
    expected = yaml.safe_load(text)
    for path, value in written.items():
        expected = _with(expected, path, value)
    if yaml.safe_load(result) != expected:
        raise ChartError("values.yaml changed beyond the image lines; nothing is published")
    return result


def prepare(chart: Path, out: Path, digests: Mapping[str, str], prefix: str, version: str) -> Dict[str, List[str]]:
    """Copy the chart to ``out``, pin every image slot, and return each edition's references."""
    values_text = (chart / "values.yaml").read_text(encoding="utf-8")
    default_edition = str(yaml.safe_load(values_text)["edition"]["authEdition"])
    if default_edition not in WEB_EDITIONS:
        raise ChartError(f"the chart defaults to the {default_edition!r} edition, which has no published web image")
    check_distinct_web_digests(digests)
    pinned = set_values(values_text, settings(plan(), digests, prefix))
    header = (f"# Automatos {version}. Every image below is named by the digest this release published\n"
              "# and signed; a digest wins over a tag.\n\n")
    target = out / CHART_NAME
    shutil.copytree(chart, target)
    (target / "values.yaml").write_text(header + pinned, encoding="utf-8")
    return references(digests, prefix)


def write_render_values(directory: Path) -> List[Path]:
    """One ``<edition>.yaml`` per published edition: the values its render is given."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for edition in sorted(WEB_EDITIONS):
        path = directory / f"{edition}.yaml"
        path.write_text(yaml.safe_dump(RENDER_VALUES[edition], sort_keys=True), encoding="utf-8")
        written.append(path)
    return written


# --------------------------------------------------------------------------- the render


def _walk(node: object) -> Iterator[Tuple[str, object]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def tagged(reference: str) -> bool:
    """True unless ``reference`` is ``repository@sha256:<64 hex>`` with no tag."""
    name, _, digest = reference.partition("@")
    return not DIGEST.match(digest) or ":" in name.rsplit("/", 1)[-1]


def check_rendered(documents: Sequence[object], expected: Sequence[str], version: str) -> List[str]:
    """Every problem with a rendered release chart; an empty list passes."""
    found = [(key, value) for document in documents for key, value in _walk(document)]
    images = {value for key, value in found if key == "image" and isinstance(value, str)}
    problems = [f"named by a tag, not a digest: {image}" for image in sorted(images) if tagged(image)]
    problems += [f"published but not run by the chart: {image}" for image in sorted(set(expected) - images)]
    problems += [f"run by the chart but not one of the images this render should run: {image}"
                 for image in sorted(images - set(expected)) if not tagged(image)]
    want = {"helm.sh/chart": f"{CHART_NAME}-{version}".replace("+", "_"), "app.kubernetes.io/version": version}
    for label, value in want.items():
        seen = {str(item) for key, item in found if key == label}
        if seen != {value}:
            problems.append(f"label {label} is {sorted(seen)}, not {value!r}")
    if not images:
        problems.append("the render names no image at all")
    return problems


def check_renders(renders: Mapping[str, Sequence[object]], expected: Mapping[str, Sequence[str]],
                  version: str) -> List[str]:
    """Every problem with the renders of every published edition, each named by its edition."""
    problems = [f"{edition}: not rendered" for edition in sorted(set(expected) - set(renders))]
    problems += [f"{edition}: rendered, but not a published edition" for edition in sorted(set(renders) - set(expected))]
    for edition in sorted(set(expected) & set(renders)):
        problems += [f"{edition}: {problem}" for problem in check_rendered(renders[edition], expected[edition], version)]
    return problems


# --------------------------------------------------------------------------- taking a version back


def package_path(owner: str, owner_type: str) -> str:
    """The packages API path of the chart's package, for an organisation or a user."""
    scope = f"orgs/{owner}" if owner_type == "Organization" else f"users/{owner}"
    return f"/{scope}/packages/container/{urllib.parse.quote(PACKAGE, safe='')}"


def removal(package: str, versions: Sequence[dict], tag: str) -> str:
    """The path to DELETE: the version tagged ``tag``, or the package when it is the only one."""
    ours = [version for version in versions
            if tag in (((version.get("metadata") or {}).get("container") or {}).get("tags") or [])]
    if len(ours) != 1:
        raise ChartError(f"{len(ours)} versions of {PACKAGE} carry the tag {tag!r}, not one")
    return package if len(versions) == 1 else f"{package}/versions/{ours[0]['id']}"


def _github(method: str, path: str, token: str) -> object:
    request = urllib.request.Request(f"{GITHUB_API}{path}", method=method, headers={
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
        body = response.read()
    return json.loads(body) if body else None


def remove(owner: str, owner_type: str, tag: str, token: str, call=None) -> str:
    """Delete the chart version tagged ``tag``; return the path deleted."""
    call = call or _github
    package = package_path(owner, owner_type)
    versions = call("GET", f"{package}/versions?per_page=100", token)
    target = removal(package, versions or [], tag)
    call("DELETE", target, token)
    return target


# --------------------------------------------------------------------------- commands


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as the other CI scripts load their siblings."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(text)


def cmd_prepare(args: argparse.Namespace) -> int:
    digests = read_digests(Path(args.digests), published_names())
    by_edition = prepare(Path(args.chart), Path(args.out), digests, args.prefix, args.version)
    Path(args.images).write_text(json.dumps(by_edition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_render_values(Path(args.render_values))
    lines = [f"- {edition}: `{reference}`\n" for edition, refs in by_edition.items() for reference in refs]
    _append(args.summary, f"### Chart {args.version}: its default images, by edition\n\n" + "".join(lines))
    print("".join(lines), end="")
    return 0


def _renders(directory: Path) -> Dict[str, List[object]]:
    """``{edition: its rendered documents}`` from ``<edition>.yaml`` files in ``directory``."""
    return {path.stem: [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
            for path in sorted(directory.glob("*.yaml"))}


def cmd_check_rendered(args: argparse.Namespace) -> int:
    renders = _renders(Path(args.rendered))
    expected = json.loads(Path(args.images).read_text(encoding="utf-8"))
    problems = check_renders(renders, expected, args.version)
    for problem in problems:
        print(f"::error title=Release chart::{problem}")
    counts = ", ".join(f"{edition} {len(documents)}" for edition, documents in renders.items())
    print(f"manifests by edition: {counts or 'none'}; {len(problems)} problems")
    return 1 if problems else 0


def cmd_digest(args: argparse.Namespace) -> int:
    digest = _sibling("image_registry").resolve(args.ref)
    if not DIGEST.match(digest):
        raise ChartError(f"{args.ref} resolved to {digest!r}")
    _append(args.outputs, f"digest={digest}\n")
    print(digest)
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    chart = f"ghcr.io/{args.owner.lower()}/{PACKAGE}:{args.tag}"
    try:
        target = remove(args.owner, args.owner_type, args.tag, os.environ.get("GH_TOKEN", ""))
    except (urllib.error.URLError, OSError, ChartError, KeyError, ValueError) as error:
        print(f"::error title=Unproven chart left published::{chart} could not be removed ({error}). A maintainer "
              f"must delete that version of the {PACKAGE} package by hand (the organisation's Packages, "
              f"{PACKAGE}, the {args.tag} version's menu), so no unproven chart stays published")
        return 1
    print(f"::warning title=Unproven chart removed::{chart} was pushed but not proven, so it was removed (DELETE {target})")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prepare", help="copy the chart and pin its images by digest")
    for flag in ("--chart", "--out", "--digests", "--prefix", "--version", "--images", "--render-values"):
        one.add_argument(flag, required=True)
    one.add_argument("--summary")
    two = commands.add_parser("check-rendered", help="fail on any image named by a tag, in any edition's render")
    for flag in ("--rendered", "--images", "--version"):
        two.add_argument(flag, required=True)
    three = commands.add_parser("digest", help="the digest a pushed chart's tag resolves to")
    three.add_argument("--ref", required=True)
    three.add_argument("--outputs")
    four = commands.add_parser("remove", help="delete a pushed chart version whose signature was not proven")
    for flag in ("--owner", "--owner-type", "--tag"):
        four.add_argument(flag, required=True)
    return root


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    command = {"prepare": cmd_prepare, "check-rendered": cmd_check_rendered, "digest": cmd_digest,
               "remove": cmd_remove}[args.command]
    try:
        return command(args)
    except (ChartError, OSError, yaml.YAMLError) as error:
        print(f"::error title=Release chart::{error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
