"""PRD-279 US-303: an install reports the version its API image was built as.

* ``config.AUTOMATOS_VERSION`` holds it: the image's AUTOMATOS_VERSION, read through
  config.py like every setting, and ``config.UNVERSIONED_BUILD`` (``dev``) when the build
  was given none (a checkout's own build, this test suite). Never the literal ``1.0.0``.
* Every place the API said ``"version": "1.0.0"`` says that value: the app's own version
  (OpenAPI), ``GET /`` and ``GET /health`` in main.py, and ``GET /api/system/health``
  (api/system.py), which the web app reads to show it beside the edition.
* images.yml builds the API with it: the tag's version on a ``v*`` tag, the full commit id on
  every other run, as one build arg the picker adds to the API alone.
* The arg is declared after the production stage's last filesystem layer, so a value that
  changes on every commit rebuilds no layer, and the cache keys and tree tags are as before.

File reads, the picker and the app's root handler: no Docker, no network.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

import config as config_module
from config import config

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "orchestrator" / "main.py"
SYSTEM = ROOT / "orchestrator" / "api" / "system.py"
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text(encoding="utf-8"))
PICK = next(step for step in WORKFLOW["jobs"]["changes"]["steps"] if step.get("id") == "pick")
IMAGES = json.loads(PICK["env"]["IMAGES"])
RUNNERS = json.loads(PICK["env"]["RUNNERS"])
LOCAL = "registry:2@sha256:" + "0" * 64
OLD_LITERAL = "1.0.0"
COMMIT = "0d9f0ac594c3783e62a30e6c5ce75164028850db"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


picker = _load("image_matrix")


def _dotted(node: ast.AST) -> str:
    """``a.b`` for an attribute read, the constant's repr for a literal, '' otherwise."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return repr(node.value) if isinstance(node, ast.Constant) else ""


def _version_values(path: Path) -> list:
    """Every value the file gives a ``version`` keyword argument or a ``"version"`` dict key."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            found += [_dotted(word.value) for word in node.keywords if word.arg == "version"]
        if isinstance(node, ast.Dict):
            found += [_dotted(value) for key, value in zip(node.keys, node.values)
                      if isinstance(key, ast.Constant) and key.value == "version"]
    return found


# --------------------------------------------------------------------------- the setting


def test_the_version_is_read_through_config_and_is_never_the_old_literal():
    assert config.AUTOMATOS_VERSION and config.AUTOMATOS_VERSION != OLD_LITERAL
    assert config_module.UNVERSIONED_BUILD == "dev" != OLD_LITERAL
    assert "AUTOMATOS_VERSION" in json.loads(
        (ROOT / "orchestrator" / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]


@pytest.mark.parametrize("raw,expected", [
    (None, "dev"), ("", "dev"), ("   ", "dev"), ("0.12.0", "0.12.0"), (" 0.12.0-rc.1\n", "0.12.0-rc.1"),
    (COMMIT, COMMIT),
])
def test_a_build_given_no_version_reports_dev_and_any_other_its_own(raw, expected):
    assert config_module._build_version(raw) == expected


# --------------------------------------------------------------------------- what the API returns


def test_every_version_main_py_returns_is_the_setting():
    """The app's own version (OpenAPI), GET / and GET /health: three places, none a literal."""
    assert _version_values(MAIN) == ["config.AUTOMATOS_VERSION"] * 3
    assert f'"{OLD_LITERAL}"' not in MAIN.read_text(encoding="utf-8")


def test_the_system_health_answer_the_web_app_reads_carries_the_setting():
    assert _version_values(SYSTEM) == ["app_config.AUTOMATOS_VERSION"]
    assert "from config import config as app_config" in SYSTEM.read_text(encoding="utf-8")


def test_the_running_app_reports_the_setting():
    import main

    assert main.app.version == config.AUTOMATOS_VERSION != OLD_LITERAL
    answer = asyncio.run(main.root())
    assert answer["version"] == config.AUTOMATOS_VERSION


# --------------------------------------------------------------------------- the image


def _production() -> list:
    code = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
    lines = [" ".join(line.split()) for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if re.fullmatch(r"FROM \S+ as production", line, re.I))
    return lines[start + 1:]


def test_the_api_image_takes_the_version_after_its_last_layer_and_sets_it_for_the_app():
    production = _production()
    app_copy = production.index("COPY --from=compile --chown=1000:1000 /app /app")
    assert production.index("ARG AUTOMATOS_VERSION") > app_copy, "no layer is rebuilt for a new value"
    assert production.index("ENV AUTOMATOS_VERSION=${AUTOMATOS_VERSION}") == production.index("ARG AUTOMATOS_VERSION") + 1
    assert DOCKERFILE.count("ARG AUTOMATOS_VERSION") == 1, "declared in the production stage only, with no default"
    after = production[app_copy + 1:]
    assert not [line for line in after if line.split()[0].upper() in {"RUN", "COPY", "ADD", "WORKDIR"}]


def _entries(version: str) -> dict:
    built = picker.matrix(IMAGES, [image["name"] for image in IMAGES], RUNNERS, True, "AutomatosAI", LOCAL,
                          version=version)
    return {(entry["name"], entry["arch"]): entry for entry in built["include"]}


@pytest.mark.parametrize("version", ["0.12.0", "0.12.0-rc.1", COMMIT])
def test_the_api_alone_is_built_with_the_version(version):
    with_version, without = _entries(version), _entries("")
    for key, entry in with_version.items():
        args = entry["build_args"].splitlines()
        if key[0] == "api":
            assert args == ["SOURCE_DATE_EPOCH=1792368000", f"AUTOMATOS_VERSION={version}"], args
        else:
            assert not [arg for arg in args if arg.startswith("AUTOMATOS_VERSION")], key
        # Only the build args differ: the cache keys, the image name and the edge it is
        # measured against are the same, so a new value reuses every cached layer.
        assert {**entry, "build_args": ""} == {**without[key], "build_args": ""}, key


@pytest.mark.parametrize("bad", ["0.12.0\nEVIL=1", "v 1", "-rc", "a" * 129, "0.12.0+build"])
def test_a_version_that_could_add_another_build_arg_is_refused(bad):
    with pytest.raises(ValueError, match="is not a version or a commit id"):
        _entries(bad)


def test_the_workflow_passes_the_tags_version_or_the_commit_id():
    assert PICK["env"]["BUILD_VERSION"] == "${{ steps.release.outputs.version || github.sha }}"
    assert '--version "$BUILD_VERSION"' in PICK["run"]
    api = next(image for image in IMAGES if image["name"] == "api")
    assert api["version_arg"] == "AUTOMATOS_VERSION" and api["build_args"] == []
    assert not [image["name"] for image in IMAGES if image.get("version_arg") and image["name"] != "api"]


def test_the_tree_tags_are_still_the_build_contexts_git_tree():
    """tree-<12> names the context's tree, so a version that changes on every commit changes no tree tag."""
    step = next(step for step in WORKFLOW["jobs"]["merge"]["steps"]
                if step.get("name") == "Tag the context's tree (tree-<12>, once per tree)")
    assert 'tag=$(python3 scripts/stack_images.py tree-tag --context "$CONTEXT")' in step["run"]
    assert "VERSION" not in json.dumps(step)
