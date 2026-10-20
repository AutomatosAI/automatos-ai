"""PRD-262 US-028: the code graph's grammars are named packages, checked inside the image off the network.

modules/codegraph/parsers/treesitter_parser.py parses each language with the grammar of
its own tree_sitter_* package (TreeSitterParser.GRAMMARS). This suite checks:

- requirements.in names every one of those packages and requirements.lock pins it, so a
  graphifyy bump that dropped one could not take it out silently (graphifyy 0.8.31
  installs all of them too). tree-sitter-language-pack, which since 1.0.0 downloads its
  grammars on first use, is in neither: test_prd262_unused_packages_go.py forbids
  importing it, keeps it out of every requirements file and looks for it in both images;
- the parser no longer carries the language pack's import, its branch or the
  three-language fallback;
- test.yml's prod-image-parity runs test_prd262_codegraph_languages.py with
  ``--network none`` in the throwaway test image (scripts/ci/image_test_image.py: built
  FROM the image under test plus requirements-dev.lock, hashes required, tests mounted
  read-only, never pushed), after the clone tests built it, and removes the image after;
  pytest itself stays out of the production image (US-014's step);
- a change to the parser, its language tests or their fixtures reruns prod-image-parity.

File reads, plus importing the parser module for GRAMMARS; no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

# CI collection-order safety net, as in the other modules that import modules.* at
# collection time.
import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from modules.codegraph.parsers.treesitter_parser import TreeSitterParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
PARSER = ORCH / "modules" / "codegraph" / "parsers" / "treesitter_parser.py"
TEST_YML = ROOT / ".github" / "workflows" / "test.yml"
LANGUAGE_TESTS = "tests/test_prd262_codegraph_languages.py"
TEST_IMAGE = "parity-be-test"
GRAMMAR_PACKAGES = 13  # 14 languages: tree_sitter_typescript serves typescript and tsx
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_CONTINUATION = re.compile(r"\\[ \t]*\n")


def _requirements(path: Path) -> dict[str, str]:
    """PEP 503 name -> the requirement line as written, comments and blank lines dropped."""
    found = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = _NAME.match(line)
        if match:
            found[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = line
    return found


def _distribution(module: str) -> str:
    """The distribution a tree_sitter_* module comes from (tree_sitter_c_sharp -> tree-sitter-c-sharp)."""
    return module.replace("_", "-")


def _workflow() -> dict:
    return yaml.safe_load(TEST_YML.read_text(encoding="utf-8"))


def _parity_runs() -> list[str]:
    """prod-image-parity's run scripts, continuation lines joined and whitespace collapsed."""
    steps = _workflow()["jobs"]["prod-image-parity"]["steps"]
    return [" ".join(_CONTINUATION.sub(" ", step.get("run", "")).split()) for step in steps]


# --------------------------------------------------------------------------- requirements


def test_requirements_name_every_grammar_package_the_parser_loads():
    requirements = _requirements(ORCH / "requirements.in")
    lock = _requirements(ORCH / "requirements.lock")
    packages = sorted({_distribution(module) for module, _ in TreeSitterParser.GRAMMARS.values()})

    assert len(packages) == GRAMMAR_PACKAGES, packages
    for name in packages:
        assert name in requirements, f"{name} must be named in requirements.in, not left to graphifyy"
        assert lock[name].startswith(f"{name}=="), f"{name} is pinned in requirements.lock"
    assert "tree-sitter" in requirements and lock["tree-sitter"].startswith("tree-sitter==")


def test_the_parser_carries_no_language_pack_and_no_fallback():
    code = "\n".join(line for line in PARSER.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))
    for gone in ("_HAS_LANGUAGE_PACK", "_get_parser_individual", "pkg_map", "import tree_sitter_language_pack",
                 "from tree_sitter_language_pack"):
        assert gone not in code, gone
    assert not re.search(r"(?<![\w.])get_parser\(", code), "the language pack's get_parser() is not called"


# --------------------------------------------------------------------------- CI


def test_ci_runs_the_language_tests_off_the_network_in_the_test_image():
    runs = _parity_runs()
    built = [i for i, run in enumerate(runs) if f"image_test_image.py build parity-be {TEST_IMAGE}" in run]
    at = [i for i, run in enumerate(runs) if LANGUAGE_TESTS in run]

    assert len(built) == 1 and len(at) == 1, (built, at)
    assert built[0] < at[0], "the test image is built before the language tests run in it"
    assert (f"python3 scripts/ci/image_test_image.py pytest {TEST_IMAGE} --network none "
            f"-q --timeout=120 {LANGUAGE_TESTS}") in runs[at[0]]
    assert (ORCH / LANGUAGE_TESTS).is_file()


def test_the_test_image_is_removed_after_its_last_run_and_never_pushed():
    runs = _parity_runs()
    used = [i for i, run in enumerate(runs) if TEST_IMAGE in run]
    removed = [i for i, run in enumerate(runs) if f"docker image rm {TEST_IMAGE}" in run]

    assert removed == [used[-1]], (used, removed)
    assert runs[used[-1]].index(LANGUAGE_TESTS) < runs[used[-1]].index(f"docker image rm {TEST_IMAGE}")
    assert not [run for run in runs if TEST_IMAGE in run and "docker push" in run]


def test_pytest_runs_only_in_the_test_image():
    """US-014: the production image still fails `import pytest`; tests run in the derived image."""
    runs = _parity_runs()
    assert any("if docker run --rm parity-be python -c 'import pytest'; then" in run for run in runs)
    targets = re.findall(r"image_test_image\.py pytest (\S+)", " ".join(runs))
    assert targets and set(targets) == {TEST_IMAGE}, targets


def test_a_change_to_the_parser_or_its_fixtures_reruns_prod_image_parity():
    changes = next(step for step in _workflow()["jobs"]["changes"]["steps"]
                   if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    pattern = changes["env"]["PROD_IMAGE_PATHS"]
    for path in (
        "orchestrator/modules/codegraph/parsers/treesitter_parser.py",
        f"orchestrator/{LANGUAGE_TESTS}",
        "orchestrator/tests/fixtures/codegraph_languages/sample.kt",
        "orchestrator/requirements.in",
        "orchestrator/requirements.lock",
    ):
        assert re.search(pattern, path), path
    assert not re.search(pattern, "orchestrator/modules/codegraph/codegraph_service.py")
