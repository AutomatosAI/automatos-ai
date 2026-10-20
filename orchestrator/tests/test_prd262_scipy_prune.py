"""PRD-262, the 19 Oct breakdown's row #17: scipy keeps the subpackages the code reaches.

pybuild deletes the scipy subpackages no code path loads (stats, io, signal, interpolate,
ndimage, cluster, odr, fftpack, differentiate, datasets, misc; about 29 MB) and fails the
build when one it keeps is missing. scipy is in the lock for datasketch (graphify's
dedup) and graspologic-native; networkx's pagerank reaches scipy.sparse; the app's own
one use, core/math/probability_theory.py, takes Student's t from scipy.special. This
suite checks:

- the Dockerfile: the prune RUN in pybuild, on the first half of the lock (scipy is in
  split_lock.py's stable set), after the install and before the compile, with exactly
  the lists scripts/ci/image_scipy.py checks in the image;
- the guard for future code: no module under orchestrator/ imports a pruned subpackage,
  at any depth, lazily or not, nor names one through ``import scipy`` and an attribute;
- the guard for scipy and datasketch bumps, on the versions the lock installs here: no
  module of a kept subpackage imports a pruned one when it loads (only inside a
  function), and datasketch reaches only kept subpackages;
- core.math's confidence interval for a small sample is Student's t (2.776 for 4 degrees
  of freedom at 95 %), which it now takes from scipy.special;
- scripts/ci/image_scipy.py: the image's bare Python can start it (standard library at
  module level), it imports the app with image_site_packages.py's throwaway settings,
  and both workflows run it inside the image they built.

File reads, ASTs of the installed scipy and datasketch, and one confidence interval; no
Docker, no network.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
SCRIPT = ROOT / "scripts" / "ci" / "image_scipy.py"
SCIPY_DIR = "/install/stable/lib/python3.11/site-packages/scipy"
T_975_DF4 = 2.7764451051977987  # Student's t, 97.5 % quantile, 4 degrees of freedom
_CONTINUATION = re.compile(r"\\[ \t]*\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


image_scipy = _load("image_scipy", SCRIPT)
site_packages = _load("image_site_packages", ROOT / "scripts" / "ci" / "image_site_packages.py")
split_lock = _load("split_lock", ORCH / "scripts" / "split_lock.py")


def _instructions(text: str = DOCKERFILE) -> List[str]:
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _pybuild() -> List[str]:
    instructions = _instructions()
    start = next(i for i, line in enumerate(instructions) if re.match(r"FROM\s.*\sas\s+pybuild$", line, re.I))
    end = next(i for i, line in enumerate(instructions) if i > start and line.upper().startswith("FROM "))
    return instructions[start:end]


def _prune_run() -> str:
    runs = [line for line in _pybuild() if line.startswith("RUN ") and SCIPY_DIR in line]
    assert len(runs) == 1, runs
    return runs[0]


def _loop_words(run: str, variable: str) -> List[str]:
    match = re.search(rf"for {variable} in ([^;]+);", run)
    assert match, f"no `for {variable} in …;` in {run}"
    return match.group(1).split()


# --------------------------------------------------------------------------- the Dockerfile


def test_the_prune_keeps_and_deletes_the_lists_the_image_check_uses():
    run = _prune_run()
    assert _loop_words(run, "kept") == list(image_scipy.KEPT)
    assert _loop_words(run, "dropped") == list(image_scipy.PRUNED)
    assert not set(image_scipy.KEPT) & set(image_scipy.PRUNED)
    assert 'test -d "$scipy/$kept" || { echo "scipy has no $kept" >&2; exit 1; }' in run
    assert 'rm -rf "${scipy:?}/$dropped" || exit 1' in run
    assert "|| true" not in run


def test_the_prune_works_on_the_half_that_holds_scipy_before_the_compile():
    assert "scipy" in split_lock.STABLE, "scipy installs into /install/stable"
    stage = _pybuild()
    at = stage.index(_prune_run())
    assert next(i for i, line in enumerate(stage) if "uv pip install" in line) < at
    assert at < next(i for i, line in enumerate(stage) if "compileall" in line), "no bytecode of what goes"


# --------------------------------------------------------------------------- the code


def _scipy_names(tree: ast.AST) -> Iterator[Tuple[int, str]]:
    """(line, scipy.<name>) for every scipy subpackage a module imports or reaches as ``alias.<name>``."""
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scipy":
                    aliases.add(alias.asname or "scipy")
                elif alias.name.startswith("scipy."):
                    yield node.lineno, ".".join(alias.name.split(".")[:2])
        elif isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "scipy":
            parts = node.module.split(".")
            names = [f"scipy.{alias.name}" for alias in node.names] if len(parts) == 1 else [".".join(parts[:2])]
            yield from ((node.lineno, name) for name in names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            yield node.lineno, f"scipy.{node.attr}"


def _pruned_uses(paths) -> List[str]:
    pruned = {f"scipy.{name}" for name in image_scipy.PRUNED}
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [f"{path}:{line} {name}" for line, name in _scipy_names(tree) if name in pruned]
    return found


def test_no_module_of_the_app_imports_a_pruned_subpackage():
    modules = [path for path in ORCH.rglob("*.py") if "tests" not in path.relative_to(ORCH).parts]
    assert _pruned_uses(modules) == []


def test_the_scan_sees_every_form_of_import():
    source = ("import scipy as sp\nimport scipy.io\nfrom scipy import stats, sparse\nfrom scipy.signal import welch\n"
              "def f():\n    return sp.ndimage\n")
    names = {name for _, name in _scipy_names(ast.parse(source))}
    assert {"scipy.io", "scipy.stats", "scipy.sparse", "scipy.signal", "scipy.ndimage"} <= names


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    assert spec and spec.submodule_search_locations, f"{name} is installed with requirements.lock"
    return Path(next(iter(spec.submodule_search_locations)))


def _module_level_imports(path: Path, package: str) -> Iterator[str]:
    """Absolute names a module imports when it loads: module level, if/try blocks and class bodies
    included, function bodies not."""
    def walk(nodes):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Import):
                yield from (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = package.split(".")[: len(package.split(".")) - node.level + 1] if node.level else []
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
                yield module
                yield from (f"{module}.{alias.name}" for alias in node.names)
            yield from walk(ast.iter_child_nodes(node))
    yield from walk(ast.parse(path.read_text(encoding="utf-8", errors="replace")).body)


def test_no_kept_subpackage_imports_a_pruned_one_when_it_loads():
    """A scipy bump that started importing a pruned subpackage at load time would fail the image."""
    scipy_dir = _package_dir("scipy")
    pruned = {f"scipy.{name}" for name in image_scipy.PRUNED}
    found = []
    for path in scipy_dir.rglob("*.py"):
        relative = path.relative_to(scipy_dir.parent).with_suffix("")
        parts = relative.parts if relative.name != "__init__" else relative.parts[:-1]
        if len(parts) > 1 and parts[1] in image_scipy.PRUNED or "tests" in parts:
            continue
        package = ".".join(parts if relative.name == "__init__" else parts[:-1])
        found += [f"{'.'.join(parts)} -> {name}" for name in _module_level_imports(path, package)
                  if ".".join(name.split(".")[:2]) in pruned]
    assert found == []


def test_datasketch_reaches_only_kept_subpackages():
    paths = sorted(_package_dir("datasketch").rglob("*.py"))
    kept = {f"scipy.{name}" for name in image_scipy.KEPT}
    reached = {name for path in paths for _, name in _scipy_names(ast.parse(path.read_text(encoding="utf-8")))}
    assert reached and reached <= kept, reached - kept


def test_a_small_samples_interval_is_students_t():
    """PRD-262 row #17: the t quantile comes from scipy.special.stdtrit, what scipy.stats.t.ppf
    returns, so the interval is the one it was (2.776 for 4 degrees of freedom, not 1.96)."""
    from core.math.probability_theory import ProbabilityTheory

    sample = [10.0, 12.0, 9.0, 11.0, 13.0]
    interval = ProbabilityTheory.calculate_confidence_interval(sample, confidence_level=0.95)
    margin = (interval.upper_bound - interval.lower_bound) / 2
    assert margin == pytest.approx(T_975_DF4 * interval.standard_error, rel=1e-12)
    assert image_scipy.T_975_DF4 == T_975_DF4 and image_scipy.SAMPLE == tuple(sample)


# --------------------------------------------------------------------------- the image check


def test_the_image_check_starts_on_the_images_bare_python():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert top - {"__future__"} <= set(sys.stdlib_module_names), top


def test_the_image_check_imports_the_app_like_the_site_packages_check():
    assert image_scipy.APP_IMPORT_ENV == site_packages.APP_IMPORT_ENV
    assert image_scipy.APP_DIR == site_packages.APP_DIR
    assert image_scipy.NEVER_LOADED == ("graphify.extract",)


def test_the_image_check_reports_a_pruned_subpackage_that_was_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.stats", object())
    assert image_scipy.loaded_problems() == ["scipy.stats was imported"]


def test_a_crashed_check_is_reported_on_one_line():
    def broken():
        raise ImportError("first line\nsecond line")

    assert image_scipy.run("x", broken) == ["ImportError: first line second line"]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_prod_image_parity_runs_it_inside_the_railway_build():
    steps = _workflow("test.yml")["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "image_scipy.py" in step.get("run", ""))
    assert built < check
    assert steps[check]["run"].strip() == "docker run --rm -i --entrypoint python parity-be - < scripts/ci/image_scipy.py"
    assert "if" not in steps[check] and "continue-on-error" not in steps[check]


def test_images_yml_runs_it_inside_every_api_build():
    steps = _workflow("images.yml")["jobs"]["build"]["steps"]
    built = next(i for i, step in enumerate(steps) if step.get("id") == "build")
    at, step = next((i, step) for i, step in enumerate(steps) if "image_scipy.py" in step.get("run", ""))
    assert built < at
    assert "matrix.name == 'api'" in step["if"] and "steps.build.outcome == 'success'" in step["if"]
    assert "!cancelled()" in step["if"]
    assert ('docker run --rm -i --entrypoint python "${IMAGE}@${DIGEST}" - < scripts/ci/image_scipy.py'
            in step["run"])
    assert "continue-on-error" not in step


def test_a_change_to_the_image_check_reruns_both():
    relative = SCRIPT.relative_to(ROOT).as_posix()
    changes = _workflow("test.yml")["jobs"]["changes"]["steps"]
    match = next(step for step in changes if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(match["env"]["PROD_IMAGE_PATHS"], relative)
    images = _workflow("images.yml")
    paths = images.get("on", images.get(True))["pull_request"]["paths"]  # PyYAML reads `on` as True
    assert any(fnmatch.fnmatch(relative, path) for path in paths)
