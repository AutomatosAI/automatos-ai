"""PRD-262 US-022: the API's two pip layers stay ``edge``'s unless what made them moved.

images.yml builds the API reproducibly (a fixed SOURCE_DATE_EPOCH, rewrite-timestamp),
and on a push to main scripts/ci/image_pip_layers.py compares the two layers pybuild's
halves make (``COPY /install/stable /usr/local``, ``COPY /install/rest /usr/local``)
with the published ``edge``'s. This suite feeds it reports and file sets, no registry:

- pybuild's inputs are its stages' instructions (comments aside, base and uv digests
  in) and split_lock.py; each half of the lock is its pins and hashes, comments aside;
- a half whose pins and pybuild are edge's must give edge's layer digest; one that
  moved is a new layer (a notice); an edge from before the split, or no edge, is a
  notice; a digest that should match and does not is judged by the distributions both
  layers install: the same set is a warning, another set or an unreadable layer fails;
- US-001's report gives edge's layers their instruction, which is how the check finds
  them, and files both COPY layers as dependencies;
- scripts/ci/image_site_packages.py imports main under ``python -v`` and fails on any
  ``bytecode is stale`` line: the clamped mtimes must never make a .pyc stale.

Pure functions plus monkeypatched reads; no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / "scripts" / "ci"
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
LOCK = (ROOT / "orchestrator" / "requirements.lock").read_text(encoding="utf-8")
SPLIT = (ROOT / "orchestrator" / "scripts" / "split_lock.py").read_text(encoding="utf-8")
EDGE_FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures" / "prd262_edge_images.json")
                          .read_text(encoding="utf-8"))
GZIP = "application/vnd.oci.image.layer.v1.tar+gzip"
SITE = "usr/local/lib/python3.11/site-packages"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


pip_layers = _load("image_pip_layers", CI / "image_pip_layers.py")
image_report = _load("image_report", CI / "image_report.py")
site_packages = _load("image_site_packages", CI / "image_site_packages.py")
STABLE, REST = pip_layers.HALVES["stable"], pip_layers.HALVES["rest"]
NOW = pip_layers.inputs(DOCKERFILE, LOCK, SPLIT)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _row(instruction: str, digest: str, size: int = 50_000_000) -> dict:
    return {"instruction": instruction, "digest": digest, "size": size, "media_type": GZIP}


def _report(stable: str = _digest("1"), rest: str = _digest("2"), edge_stable: str = _digest("1"),
            edge_rest: str = _digest("2")) -> dict:
    """One API build's report as image_report.py measure writes it, with its edge."""
    return {
        "name": "api", "arch": "amd64", "image": "ghcr.io/o/automatos-api", "digest": _digest("9"),
        "layers": [_row("WORKDIR /app", _digest("0")), _row(STABLE, stable), _row(REST, rest),
                   _row("COPY . .", _digest("3"))],
        "edge": {"ref": "ghcr.io/o/automatos-api:edge", "revision": "a" * 40,
                 "layers": [_row("WORKDIR /app", _digest("0")), _row(STABLE, edge_stable), _row(REST, edge_rest),
                            _row("COPY . .", _digest("4"))]},
        "edge_note": "",
    }


def _levels(verdicts: dict) -> dict:
    return {half: verdict.level for half, verdict in verdicts.items()}


# --------------------------------------------------------------------------- pybuild's inputs


def test_pybuilds_inputs_are_its_stages_with_their_pinned_bases():
    pybuild = NOW.pybuild
    assert pybuild[0].startswith("FROM ghcr.io/astral-sh/uv:") and "@sha256:" in pybuild[0]
    assert any(re.match(r"FROM python:3\.11-slim@sha256:[0-9a-f]{64} as pybuild$", line, re.I) for line in pybuild)
    assert any("uv pip install" in line for line in pybuild) and any("compileall" in line for line in pybuild)
    assert not [line for line in pybuild if line.startswith("#") or re.search(r"\sas base$", line, re.I)]


def test_a_comment_or_a_later_stage_leaves_pybuilds_inputs_alone():
    reworded = DOCKERFILE.replace("# binutils: the strip below", "# binutils, for the strip below")
    assert reworded != DOCKERFILE
    later = DOCKERFILE + "\nLABEL org.example.note=later\n"
    assert pip_layers.pybuild_instructions(reworded) == NOW.pybuild
    assert pip_layers.pybuild_instructions(later) == NOW.pybuild


def test_a_new_base_digest_moves_pybuilds_inputs():
    digest = re.search(r"python:3\.11-slim@(sha256:[0-9a-f]{64})", DOCKERFILE).group(1)
    moved = DOCKERFILE.replace(digest, _digest("f"))
    assert pip_layers.pybuild_instructions(moved) != NOW.pybuild


def test_a_dockerfile_without_a_base_stage_is_refused():
    with pytest.raises(ValueError, match="AS base"):
        pip_layers.pybuild_instructions("FROM python:3.11-slim AS pybuild\nRUN true\n")


def test_each_half_is_its_pins_and_hashes_comments_aside():
    assert NOW.halves["stable"] and NOW.halves["rest"]
    assert not [pin for pins in NOW.halves.values() for pin in pins if "#" in pin]
    assert all("--hash=sha256:" in pin for pins in NOW.halves.values() for pin in pins)
    stable_names = {re.match(r"[^=\s]+", pin).group(0).lower() for pin in NOW.halves["stable"]}
    assert stable_names == set(pip_layers.split_lock.STABLE)
    recommented = re.sub(r"# via", "# via, recompiled,", LOCK)
    assert pip_layers.inputs(DOCKERFILE, recommented, SPLIT) == NOW


def test_a_bump_outside_the_stable_set_moves_only_the_second_half():
    pin = re.search(r"^pysher==(\S+) \\$", LOCK, re.M)
    bumped = pip_layers.inputs(DOCKERFILE, LOCK.replace(pin.group(0), "pysher==9.9.9 \\"), SPLIT)
    assert bumped.halves["stable"] == NOW.halves["stable"]
    assert bumped.halves["rest"] != NOW.halves["rest"]


# --------------------------------------------------------------------------- what each layer must be


def test_no_edge_files_or_moved_pybuild_expect_no_match():
    assert _levels(pip_layers.expected(_report(), NOW, None)) == {"stable": "notice", "rest": "notice"}
    moved = NOW._replace(pybuild=NOW.pybuild + ("RUN true",))
    verdicts = pip_layers.expected(_report(stable=_digest("5")), NOW, moved)
    assert _levels(verdicts) == {"stable": "notice", "rest": "notice"}
    assert "pybuild or split_lock.py changed" in verdicts["stable"].message
    other_split = NOW._replace(split_script=SPLIT + "\n# changed\n")
    assert _levels(pip_layers.expected(_report(), NOW, other_split)) == {"stable": "notice", "rest": "notice"}


def test_unchanged_inputs_must_give_edges_layers():
    assert _levels(pip_layers.expected(_report(), NOW, NOW)) == {"stable": "ok", "rest": "ok"}
    verdicts = pip_layers.expected(_report(stable=_digest("5")), NOW, NOW)
    assert _levels(verdicts) == {"stable": "error", "rest": "ok"}
    assert "but the layer is not" in verdicts["stable"].message


def test_a_moved_half_is_a_new_layer_and_says_what_it_costs():
    edge = NOW._replace(halves={**NOW.halves, "rest": NOW.halves["rest"][1:]})
    verdicts = pip_layers.expected(_report(rest=_digest("6")), NOW, edge)
    assert _levels(verdicts) == {"stable": "ok", "rest": "notice"}
    assert "an update downloads its 50.0 MB" in verdicts["rest"].message


def test_an_edge_from_before_the_split_and_a_build_without_the_layers():
    report = _report()
    report["edge"]["layers"] = [_row("COPY /install /usr/local", _digest("7"))]
    verdicts = pip_layers.expected(report, NOW, NOW)
    assert _levels(verdicts) == {"stable": "notice", "rest": "notice"}
    assert "predates the split" in verdicts["rest"].message
    report = _report()
    report["layers"] = [row for row in report["layers"] if row["instruction"] != REST]
    assert _levels(pip_layers.expected(report, NOW, NOW)) == {"stable": "ok", "rest": "error"}


def test_two_layers_from_one_copy_are_refused():
    report = _report()
    report["layers"].append(_row(STABLE, _digest("8")))
    with pytest.raises(ValueError, match="2 layers"):
        pip_layers.expected(report, NOW, NOW)


# --------------------------------------------------------------------------- a mismatch, by distributions


def test_distributions_are_read_from_dist_info_paths():
    paths = [f"{SITE}/numpy-2.4.6.dist-info/METADATA", f"{SITE}/numpy-2.4.6.dist-info/RECORD",
             f"{SITE}/Pydantic_Core-2.41.5.dist-info/METADATA", f"{SITE}/numpy/__init__.py", "usr/local/bin/f2py"]
    assert pip_layers.distributions(paths) == {"numpy==2.4.6", "pydantic-core==2.41.5"}


def test_the_same_distributions_warn_and_others_fail():
    same = pip_layers.same_distributions(frozenset({"numpy==2.4.6"}), frozenset({"numpy==2.4.6"}))
    assert same.level == "warning" and "moved the bytes" in same.message
    other = pip_layers.same_distributions(frozenset({"numpy==2.4.7"}), frozenset({"numpy==2.4.6"}))
    assert other.level == "error" and "new numpy==2.4.7" in other.message and "gone numpy==2.4.6" in other.message


def test_a_mismatch_is_judged_by_what_both_layers_install(monkeypatch):
    listed = {_digest("5"): [f"{SITE}/numpy-2.4.6.dist-info/METADATA"],
              _digest("1"): [f"{SITE}/numpy-2.4.6.dist-info/METADATA"]}
    monkeypatch.setattr(pip_layers, "layer_paths", lambda ref, digest, media_type: listed[digest])
    assert _levels(pip_layers.judge(_report(stable=_digest("5")), NOW, NOW)) == {"stable": "warning", "rest": "ok"}
    listed[_digest("5")] = [f"{SITE}/numpy-2.4.7.dist-info/METADATA"]
    assert _levels(pip_layers.judge(_report(stable=_digest("5")), NOW, NOW)) == {"stable": "error", "rest": "ok"}


def test_a_layer_that_cannot_be_read_fails(monkeypatch):
    def unreadable(ref, digest, media_type):
        raise OSError("connection reset")

    monkeypatch.setattr(pip_layers, "layer_paths", unreadable)
    verdict = pip_layers.judge(_report(stable=_digest("5")), NOW, NOW)["stable"]
    assert verdict.level == "error" and "could not be read: connection reset" in verdict.message


def test_no_edge_in_the_report_is_a_notice():
    report = {**_report(), "edge": None, "edge_note": "`edge` could not be read"}
    verdicts = pip_layers.judge(report, NOW, NOW)
    assert _levels(verdicts) == {"stable": "notice", "rest": "notice"}
    assert "could not be read" in verdicts["stable"].message


# --------------------------------------------------------------------------- the command


def test_finish_annotates_summarises_and_fails_only_on_an_error(tmp_path, capsys):
    summary = tmp_path / "summary.md"
    ok, error = pip_layers.Verdict("ok", "same"), pip_layers.Verdict("error", "differs")
    assert pip_layers.finish(_report(), {"stable": ok, "rest": error}, str(summary)) == 1
    out = capsys.readouterr().out
    assert "::error title=Pip layer rest::differs" in out and "Pip layer stable" not in out
    assert f"- `{REST}`: error: differs." in summary.read_text()
    warning = pip_layers.Verdict("warning", "bytes moved")
    assert pip_layers.finish(_report(), {"stable": ok, "rest": warning}, None) == 0


def _edge_dir(tmp_path: Path, lock: str = LOCK) -> Path:
    edge = tmp_path / "edge"
    for path, text in ((pip_layers.DOCKERFILE, DOCKERFILE), (pip_layers.LOCK, lock), (pip_layers.SPLIT, SPLIT)):
        (edge / path).parent.mkdir(parents=True, exist_ok=True)
        (edge / path).write_text(text, encoding="utf-8")
    return edge


def test_main_compares_with_edges_files(tmp_path, capsys):
    report = tmp_path / "api-amd64.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    summary = tmp_path / "summary.md"
    args = ["--report", str(report), "--edge-dir", str(_edge_dir(tmp_path)), "--summary", str(summary)]
    assert pip_layers.main(args) == 0
    assert summary.read_text().count(": ok: ") == 2
    report.write_text(json.dumps(_report(rest=_digest("6"))), encoding="utf-8")
    shutil.rmtree(tmp_path / "edge")
    assert pip_layers.main(args) == 0, "an edge whose files were not fetched expects no match"
    assert "predates the split" in capsys.readouterr().out


def test_an_edge_lock_that_does_not_split_expects_no_match(tmp_path, capsys):
    unsplittable = re.sub(r"^lxml==.*?(?=^\S)", "", LOCK, flags=re.M | re.S)
    assert "\nlxml==" not in unsplittable
    assert pip_layers._edge_inputs(_edge_dir(tmp_path, unsplittable)) is None
    assert "does not split" in capsys.readouterr().out


# --------------------------------------------------------------------------- US-001's report


def test_the_report_files_both_pip_layers_as_dependencies():
    layers = pip_layers.layers
    for half in ("stable", "rest"):
        made = layers.instruction({"created_by": f"COPY /install/{half} /usr/local # buildkit"})
        assert made == pip_layers.HALVES[half]
        assert layers.kind_of(made) == "dependencies"


def test_edges_layers_carry_the_instruction_that_made_them(monkeypatch):
    api = EDGE_FIXTURE["api"]
    monkeypatch.setattr(image_report.registry, "image",
                        lambda ref, arch: ({**api["manifest"], "digest": _digest("c")}, api["config"], None))
    edge, note = image_report.read_edge("ghcr.io/o/automatos-api:edge", "amd64")
    assert note == "" and edge["revision"] == "0d9f0ac594c3783e62a30e6c5ce75164028850db"
    assert all(set(row) == {"digest", "size", "media_type", "instruction"} for row in edge["layers"])
    assert "COPY /install /usr/local" in [row["instruction"] for row in edge["layers"]]
    assert pip_layers._layer(edge["layers"], "stable") is None, "the recorded edge predates the split"


# --------------------------------------------------------------------------- no stale bytecode


def test_stale_bytecode_lines_are_picked_out_of_python_v():
    stderr = ("# code object from '/usr/local/lib/python3.11/site-packages/a/__pycache__/x.cpython-311.pyc'\n"
              "# bytecode is stale for 'a.y'\nimport 'a.y' # <_frozen_importlib_external.SourceFileLoader>\n")
    assert site_packages.stale_bytecode(stderr) == ["# bytecode is stale for 'a.y'"]
    assert site_packages.stale_bytecode("import 'a' # loader\n") == []


def test_the_app_import_runs_under_python_v_and_fails_on_stale_bytecode(monkeypatch, capsys):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", run.stderr)

    run.stderr = "# bytecode is stale for 'pydantic.main'\n"
    monkeypatch.setattr(site_packages.subprocess, "run", run)
    problems = site_packages.app_problems()
    assert calls[0][1:] == ["-v", "-c", site_packages.APP_IMPORT]
    assert problems and "1 modules found their bytecode stale" in problems[0]
    run.stderr = "import 'main' # loader\n"
    assert site_packages.app_problems() == []
    assert "0 lines say `bytecode is stale`" in capsys.readouterr().out
