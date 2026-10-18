"""Image size, layers, update delta and budget reports for images.yml (PRD-262 US-001).

Four commands, each run by one images.yml step:

* ``measure`` (each build job, after the push) reads the pushed digest back from its
  registry, picks the job's architecture out of the index, and writes a JSON report:
  compressed size, every layer with the instruction that made it and its heaviest
  paths, and the update delta against the published ``edge``. It appends a
  Markdown table to the step summary.
* ``budget`` (each build job) checks that report against ``.github/image-budgets.json``.
  In the file's ``warn`` mode an over-budget build prints a warning; ``fail`` mode
  (PRD-262 US-004) fails the job.
* ``layer`` (the API's build jobs, PRD-262 US-014) checks that one layer of that
  report holds only files of one suffix below one directory: the stdlib compile's
  layer holds stdlib ``.pyc`` and nothing under site-packages. It fails the job
  otherwise, or when that layer could not be listed.
* ``report`` (the report job) reads every build's JSON and works out, per arch, the
  layers the API and the worker share and what the default stack's app images cost
  to pull. An image this run did not build counts as its current ``edge``.

Workflow annotations (``::warning::``) go to stdout; Markdown goes to ``--summary``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tarfile
import zlib
from pathlib import Path
from typing import List, Optional, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


layers = _sibling("image_layers")
registry = _sibling("image_registry")

REVISION_LABEL = "org.opencontainers.image.revision"
READ_ERRORS = registry.READ_ERRORS
LIST_ERRORS = READ_ERRORS + (tarfile.TarError, zlib.error, EOFError)
SHARED_PAIR = ("api", "workspace-worker")
# The app images `make up` runs from this repository (the datastores are not ours).
DEFAULT_STACK = ("api", "frontend", "workspace-worker")
ARCHES = ("amd64", "arm64")
SUMMARY_LAYER_MIN = layers.MB  # the heaviest-paths list covers layers of 1 MB or more
SUMMARY_PATHS = 8


def _warn(title: str, message: str) -> None:
    print(f"::warning title={title}::{message}")


def _append(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as summary:
            summary.write(text.rstrip("\n") + "\n\n")


def _revision(config: dict) -> str:
    return (config.get("config", {}).get("Labels") or {}).get(REVISION_LABEL, "")


def _listing(session, row: dict) -> dict:
    """``row``'s path listing, or ``{"error": …}`` with a warning when it can't be read."""
    try:
        with registry.blob(session, row["digest"]) as stream:
            return layers.list_paths(stream, row["media_type"])
    except LIST_ERRORS as error:
        _warn("Layer not listed", f"{row['digest']}: {error}")
        return {"error": str(error)}


def read_edge(ref: str, arch: str) -> Tuple[Optional[dict], str]:
    """``edge``'s layers for ``arch``, or None and why (an unpublished image has none)."""
    if not ref:
        return None, "no `edge`: this image is not published, so only its size is reported"
    try:
        manifest, config, _ = registry.image(ref, arch)
    except READ_ERRORS as error:
        _warn("No edge to compare with", f"{ref} ({arch}): {error}")
        return None, f"`edge` could not be read ({error}), so no update delta"
    return {
        "ref": ref,
        "manifest": manifest["digest"],
        "revision": _revision(config),
        "bytes": sum(layer["size"] for layer in manifest["layers"]),
        "layers": [{"digest": layer["digest"], "size": layer["size"]} for layer in manifest["layers"]],
    }, ""


def build_report(args: argparse.Namespace) -> dict:
    """The JSON report for one pushed image and architecture."""
    manifest, config, session = registry.image(f"{args.image}@{args.digest}", args.arch)
    rows = layers.layer_rows(manifest, config)
    edge, note = read_edge(args.edge, args.arch)
    delta = layers.update_delta(rows, (layer["digest"] for layer in edge["layers"])) if edge else None
    fresh = set(delta["new_digests"]) if delta else set()
    listed = [{**row, "new": row["digest"] in fresh if delta else None, "listing": _listing(session, row)}
              for row in rows]
    unpacked = [row["listing"].get("bytes") for row in listed]
    return {
        "name": args.name,
        "arch": args.arch,
        "image": args.image,
        "digest": args.digest,
        "manifest": manifest["digest"],
        "revision": _revision(config),
        "bytes": sum(row["size"] for row in rows),
        "unpacked_bytes": None if None in unpacked else sum(unpacked),
        "layers": listed,
        "edge": edge,
        "edge_note": note,
        "delta": delta,
    }


def _cell(text: str) -> str:
    return "`" + text.replace("`", "'").replace("|", "\\|") + "`"


def _delta_line(report: dict) -> str:
    delta, edge = report["delta"], report["edge"]
    if not delta:
        return f"**Update delta:** {report['edge_note']}."
    against = f"**Update delta** vs `edge` ({edge['revision'][:7] or edge['manifest'][:19]})"
    if delta["change"] == "unchanged":
        return f"{against}: **{layers.mb(0)}**, every layer is already in `edge`."
    parts = " · ".join(f"{kind} {layers.mb(size)}" for kind, size in delta["by_kind"].items())
    return (f"{against}: **{layers.mb(delta['bytes'])}** in {delta['layers']} new layers, "
            f"a **{delta['change']}** change ({parts}).")


def _layer_table(report: dict) -> List[str]:
    lines = ["| # | Compressed | Unpacked | Kind | New | Instruction |", "|--:|--:|--:|---|:-:|---|"]
    for position, row in enumerate(report["layers"], 1):
        unpacked = row["listing"].get("bytes")
        new = {True: "yes", False: "", None: "–"}[row["new"]]
        lines.append(f"| {position} | {layers.mb(row['size'])} | "
                     f"{layers.mb(unpacked) if unpacked is not None else '?'} | {row['kind']} | {new} | "
                     f"{_cell(row['instruction'])} |")
    return lines


def _heaviest(report: dict) -> List[str]:
    lines = ["<details><summary>Heaviest paths per layer (layers of 1 MB or more)</summary>", ""]
    for position, row in enumerate(report["layers"], 1):
        if row["size"] < SUMMARY_LAYER_MIN or "heaviest" not in row["listing"]:
            continue
        lines.append(f"**Layer {position}** {_cell(row['instruction'])}")
        lines += [f"- `{path}` {layers.mb(size)}" for path, size in row["listing"]["heaviest"][:SUMMARY_PATHS]]
        lines.append("")
    return lines + ["</details>"]


def report_markdown(report: dict) -> str:
    """The step-summary section for one image and architecture."""
    unpacked = report["unpacked_bytes"]
    head = [
        f"### {report['name']} ({report['arch']}): {layers.mb(report['bytes'])} compressed, "
        f"{len(report['layers'])} layers",
        "",
        f"Unpacked {layers.mb(unpacked) if unpacked is not None else '(not every layer could be listed)'}. "
        f"Revision `{report['revision'][:7] or '?'}`. Manifest `{report['manifest'][:19]}`.",
        "",
        _delta_line(report),
        "",
    ]
    return "\n".join(head + _layer_table(report) + [""] + _heaviest(report))


def measure(args: argparse.Namespace) -> int:
    report = build_report(args)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    _append(args.summary, report_markdown(report))
    return 0


def budget(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    budgets = json.loads(Path(args.budgets).read_text(encoding="utf-8"))
    verdict = layers.budget_verdict(report["name"], report["arch"], report["bytes"], budgets)
    if verdict.level != "ok":
        print(f"::{verdict.level} title=Image size budget::{verdict.message}")
    _append(args.summary, f"**Budget** ({budgets.get('mode', 'warn')} mode): {verdict.message}.")
    return 1 if verdict.level == "error" else 0


def layer(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    verdict = layers.layer_verdict(report["layers"], args.instruction, args.under, args.not_under, args.suffix)
    if verdict.level != "ok":
        print(f"::{verdict.level} title=Layer contents::{report['name']} ({report['arch']}): {verdict.message}")
    _append(args.summary, f"**Layer check:** {verdict.message}.")
    return 1 if verdict.level == "error" else 0


def _stack_layers(name: str, arch: str, built: dict, edge_prefix: str) -> Tuple[List[dict], str]:
    """This run's layers for ``name``, or its current ``edge``'s when it was not built."""
    if (name, arch) in built:
        return [{"digest": row["digest"], "size": row["size"]} for row in built[(name, arch)]["layers"]], "this build"
    manifest, _, _ = registry.image(f"{edge_prefix}{name}:edge", arch)
    return manifest["layers"], "edge"


def stack_report(built: dict, edge_prefix: str) -> dict:
    """Per arch: the API/worker shared layers and the default stack's first pull.

    The report is report only, so an ``edge`` it cannot read (a registry hiccup, or an
    image never published) leaves that arch unmeasured, with a warning and its
    ``error``, instead of failing the job.
    """
    result = {}
    for arch in ARCHES:
        try:
            stack = {name: _stack_layers(name, arch, built, edge_prefix) for name in DEFAULT_STACK}
        except READ_ERRORS as error:
            _warn("Stack not measured", f"{arch}: an image's `edge` could not be read ({error})")
            result[arch] = {"error": str(error)}
            continue
        first, second = (stack[name][0] for name in SHARED_PAIR)
        result[arch] = {
            "sources": {name: source for name, (_, source) in stack.items()},
            "shared": layers.shared_layers(first, second),
            "first_pull_bytes": layers.unique_bytes(layer_list for layer_list, _ in stack.values()),
        }
    return result


def stack_markdown(built: dict, stack: dict) -> str:
    lines = ["### Images in this run", "", "| Image | Arch | Compressed | Update delta | Change |",
             "|---|---|--:|--:|---|"]
    for (name, arch), report in sorted(built.items()):
        delta = report["delta"]
        lines.append(f"| {name} | {arch} | {layers.mb(report['bytes'])} | "
                     f"{layers.mb(delta['bytes']) if delta else '–'} | {delta['change'] if delta else 'no edge'} |")
    lines += ["", "### API and worker, and the default first pull", "",
              "| Arch | Shared layers | Shared | App images' first pull (unique layers) | From |",
              "|---|--:|--:|--:|---|"]
    for arch, numbers in stack.items():
        if "error" in numbers:
            lines.append(f"| {arch} | – | – | – | not measured: {_cell(numbers['error'])} |")
            continue
        shared = numbers["shared"]
        sources = ", ".join(f"{name}: {source}" for name, source in numbers["sources"].items())
        lines.append(f"| {arch} | {shared['layers']} | {layers.mb(shared['bytes'])} | "
                     f"{layers.mb(numbers['first_pull_bytes'])} | {sources} |")
    lines += ["", "The first pull counts the API, frontend and worker images, each distinct layer once; "
              "the datastore images are not included. Shared layers leave out layers under 1 KiB."]
    return "\n".join(lines)


def report(args: argparse.Namespace) -> int:
    built = {}
    for path in sorted(Path(args.dir).glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        built[(item["name"], item["arch"])] = item
    stack = stack_report(built, args.edge_prefix)
    sizes = [{key: item[key] for key in ("name", "arch", "bytes", "unpacked_bytes", "revision", "delta")}
             for item in built.values()]
    Path(args.out).write_text(json.dumps({"images": sizes, "stack": stack}, indent=1), encoding="utf-8")
    _append(args.summary, stack_markdown(built, stack))
    if args.outputs:
        # An unmeasured arch writes no outputs, so a reader sees them empty: not measured.
        with open(args.outputs, "a", encoding="utf-8") as outputs:
            for arch, numbers in stack.items():
                if "error" in numbers:
                    continue
                outputs.write(f"shared_layers_{arch}={numbers['shared']['layers']}\n")
                outputs.write(f"first_pull_bytes_{arch}={numbers['first_pull_bytes']}\n")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    one = commands.add_parser("measure", help="report one pushed image and arch")
    for flag in ("--name", "--arch", "--image", "--digest", "--out"):
        one.add_argument(flag, required=True)
    one.add_argument("--edge", default="", help="the published edge tag; empty when there is none")
    one.add_argument("--summary", help="append Markdown here ($GITHUB_STEP_SUMMARY)")
    check = commands.add_parser("budget", help="check one report against the budgets")
    check.add_argument("--report", required=True)
    check.add_argument("--budgets", required=True)
    check.add_argument("--summary")
    one_layer = commands.add_parser("layer", help="check what one layer of a report holds")
    for flag in ("--report", "--instruction", "--under", "--not-under", "--suffix"):
        one_layer.add_argument(flag, required=True)
    one_layer.add_argument("--summary")
    every = commands.add_parser("report", help="the shared-layer and first-pull report")
    every.add_argument("--dir", required=True, help="the build jobs' JSON reports")
    every.add_argument("--edge-prefix", required=True, help="ghcr.io/<owner>/automatos-")
    every.add_argument("--out", required=True)
    every.add_argument("--summary")
    every.add_argument("--outputs", help="append job outputs here ($GITHUB_OUTPUT)")
    return root


def main(argv: List[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {"measure": measure, "budget": budget, "layer": layer, "report": report}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
