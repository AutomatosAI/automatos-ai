"""PRD-262 #7: services/media-render/ci/assert_module_trace.py judges a module trace.

media-render.yml runs the image's render, check and snapshot with Node's module
resolution logged (ci/trace_modules.mjs) and passes the log to this script, which
must fail when any process loaded, or tried and failed to load, a path the js stage
prunes, and must not pass a trace that saw no render at all. These tests feed it
traces built here, in the hook's line format.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "services" / "media-render" / "ci" / "assert_module_trace.py"
PRUNE_LIST = ROOT / "services" / "media-render" / "js" / "prune.txt"
NM = "/opt/media-render/js/node_modules"


def _load():
    spec = importlib.util.spec_from_file_location("assert_module_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["assert_module_trace"] = module  # before exec: its dataclass looks itself up
    spec.loader.exec_module(module)
    return module


trace_check = _load()


def _cli(pid: int, command: str) -> list[dict]:
    """One `hyperframes <command>` process that loaded what every render loads."""
    files = (
        "hyperframes/bin/hyperframes.mjs",
        "hyperframes/dist/cli.js",
        "puppeteer-core/lib/puppeteer/puppeteer-core.js",
        "sharp/dist/index.mjs",
        "esbuild/lib/main.js",
    )
    return [{"pid": pid, "argv": [f"{NM}/.bin/hyperframes", command, "/tmp/job"]}] + [
        {"pid": pid, "file": f"{NM}/{path}"} for path in files
    ]


def _clean() -> list[dict]:
    return _cli(1, "check") + _cli(2, "render") + _cli(3, "check") + _cli(4, "snapshot") + [
        {"pid": 2, "argv": [f"{NM}/hyperframes/dist/renderSetupWorker.js", "browser"]},
        {"pid": 2, "failed": "canvas", "parent": f"file://{NM}/hyperframes/dist/cli.js"},
        {"pid": 2, "failed": "bufferutil", "parent": f"file://{NM}/ws/lib/buffer-util.js"},
        {"pid": 2, "file": "/opt/media-render/js/node_modules/ws/lib/buffer-util.js"},
    ]


def _run(tmp_path: Path, entries: list[dict], *extra: str) -> int:
    log = tmp_path / "modules.log"
    log.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return trace_check.main(["--trace", str(log), "--prune", str(PRUNE_LIST), *extra])


def test_a_trace_of_render_check_and_snapshot_that_touched_nothing_pruned_passes(tmp_path):
    assert _run(tmp_path, _clean()) == 0


@pytest.mark.parametrize("path", [
    "chromium-bidi/lib/cjs/bidiMapper/BidiMapper.js",
    "zod/v4/index.js",
    "hyperframes/dist/studio/index.html",
    "hyperframes/dist/fontLocalizeCli.js",
    "puppeteer-core/lib/es5-iife/puppeteer-core-browser.js",
])
def test_a_file_loaded_from_a_pruned_path_fails(tmp_path, path):
    assert _run(tmp_path, _clean() + [{"pid": 2, "file": f"{NM}/{path}"}]) == 1


@pytest.mark.parametrize("path", ["hyperframes/dist/cli.js.map", "sharp/lib/index.d.ts", "gsap/README.md"])
def test_a_file_of_a_type_the_js_stage_deletes_fails(tmp_path, path):
    assert _run(tmp_path, _clean() + [{"pid": 3, "file": f"{NM}/{path}"}]) == 1


@pytest.mark.parametrize(("specifier", "parent"), [
    ("chromium-bidi/lib/cjs/bidiMapper/BidiMapper.js", f"file://{NM}/puppeteer-core/lib/puppeteer/bidi/BidiOverCdp.js"),
    ("zod", f"file://{NM}/chromium-bidi/lib/cjs/protocol/protocol.js"),
    ("prettier", f"file://{NM}/hyperframes/dist/cli.js"),
    ("./studio/index.js", f"file://{NM}/hyperframes/dist/cli.js"),
    ("../../../devtools-protocol/json/browser_protocol.json", f"file://{NM}/puppeteer-core/lib/puppeteer/x.js"),
    (f"file://{NM}/giget/dist/index.mjs", f"file://{NM}/hyperframes/dist/cli.js"),
])
def test_a_pruned_path_tried_and_swallowed_fails(tmp_path, specifier, parent):
    assert _run(tmp_path, _clean() + [{"pid": 2, "failed": specifier, "parent": parent}]) == 1


def test_a_failed_optional_package_that_is_not_pruned_passes(tmp_path):
    extra = [{"pid": 4, "failed": "puppeteer", "parent": f"file://{NM}/hyperframes/dist/cli.js"},
             {"pid": 4, "failed": "node:sqlite", "parent": None}]
    assert _run(tmp_path, _clean() + extra) == 0


def test_a_trace_missing_a_command_fails(tmp_path):
    without_snapshot = [entry for entry in _clean() if entry["pid"] != 4]
    assert _run(tmp_path, without_snapshot) == 1
    assert _run(tmp_path, without_snapshot, "--commands", "check,render") == 0


def test_a_trace_that_saw_no_render_fails(tmp_path):
    # The hook wrote argv lines but no browser driver loaded: nothing was proved.
    hollow = [entry for entry in _clean() if "puppeteer-core" not in entry.get("file", "")]
    assert _run(tmp_path, hollow) == 1


def test_a_line_that_is_not_json_is_an_error(tmp_path):
    log = tmp_path / "modules.log"
    log.write_text('{"pid": 1, "argv": []}\nnot json\n')
    with pytest.raises(ValueError, match="modules.log:2 is not JSON"):
        trace_check.read_trace(log)


def test_the_prune_list_drops_comments_and_refuses_an_empty_list(tmp_path):
    listed = tmp_path / "prune.txt"
    listed.write_text("# a comment\n\nzod  # trailing\nhyperframes/dist/studio\n")
    assert trace_check.read_prune(listed) == ("zod", "hyperframes/dist/studio")
    listed.write_text("# nothing\n\n")
    with pytest.raises(ValueError, match="lists no path"):
        trace_check.read_prune(listed)


def test_pruned_entry_matches_a_path_or_what_lies_under_it_never_a_namesake():
    pruned = ("zod", "hyperframes/dist/studio")
    assert trace_check.pruned_entry("zod/index.js", pruned) == "zod"
    assert trace_check.pruned_entry("hyperframes/dist/studio", pruned) == "hyperframes/dist/studio"
    assert trace_check.pruned_entry("zodiac/index.js", pruned) is None
    assert trace_check.pruned_entry("hyperframes/dist/studio-server.js", pruned) is None
