"""PRD-262 US-015: packages the code never imports leave the API image.

nltk, passlib, python-jose (with its ecdsa and rsa closure), pgvector (the Python
package; the Postgres extension stays) and scikit-learn left requirements.txt. The
Dockerfile's separate `--no-deps futureagi==0.6.0` install, a PRD-58 leftover, went
too: the API calls agent-opt-worker over HTTP and never imports futureagi (its module
is `fi`). Railway's [leiden] build still gets scikit-learn through graspologic until
US-027. US-018 removed gotenberg-client with the Gotenberg sidecar and its only client,
ConversionService, which nothing called. This suite checks:

- no module under orchestrator/, tests included, imports one of them: an AST walk over
  `import`, `from ... import` and the lazy forms (importlib.import_module, __import__,
  importlib.util.find_spec, pytest.importorskip) with a literal name, at any depth;
- requirements.in, requirements-dev.txt and the locks compiled from them (PRD-262
  US-022) name none of them (the [leiden] lock only scikit-learn), the Dockerfile's one
  install is the hashed lock, never futureagi, and start.sh no longer downloads NLTK data;
- graphifyy is pinned exactly in requirements.in, and with the [leiden] extra at the same
  version in requirements-leiden.in; psutil is pinned at 6 or later, the first release
  with aarch64 wheels;
- pybuild installs wheels only, with `--no-binary pysher,graspologic` after
  `--only-binary :all:` (pip's `:all:` cleared an exception named before it; uv, which
  installs the lock since US-022, takes them either way and the order stays);
- CI looks inside both built variants (scripts/ci/image_dists_absent.py): test.yml's
  prod-image-parity checks the Railway build, smoke-images-pr the GHCR build.

FORBIDDEN_IMPORTS is meant to grow: W3 adds camelot and cv2 (US-023),
google.generativeai (US-024), pandas outside pandas_ai_service.py's methods (US-025,
which can filter on ImportSite.scope), langchain_text_splitters and langchain_core
(US-026) and tree_sitter_language_pack (US-028). A dotted name forbids that module
and everything under it.

Pure file reads, plus the check script run with this Python: no Docker, no network.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
DISTS_SCRIPT = ROOT / "scripts" / "ci" / "image_dists_absent.py"

# Import name -> the distribution it came from.
FORBIDDEN_IMPORTS = {
    "nltk": "nltk",
    "passlib": "passlib",
    "jose": "python-jose",
    "pgvector": "pgvector",
    "sklearn": "scikit-learn",
    "fi": "futureagi",
    "gotenberg_client": "gotenberg-client",  # US-018
}
# Distributions no image installs: the ones above plus python-jose's closure.
REMOVED_DISTS = frozenset(FORBIDDEN_IMPORTS.values()) | {"ecdsa", "rsa"}
# Railway's [leiden] build keeps this one, through graspologic, until US-027.
RAILWAY_KEEPS = frozenset({"scikit-learn"})
WHEEL_EXCEPTIONS = "pysher,graspologic"

SKIP_DIRS = frozenset({"venv", ".venv", "node_modules", "__pycache__", "site-packages"})
LAZY_IMPORTERS = frozenset({"import_module", "__import__", "find_spec", "importorskip"})
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]+\])?==(\d+)(\.\d+)+$")
# An import names its module, so a file that never mentions a forbidden name as a whole
# word cannot import it: only the files that do are parsed (about 45 of 3,000).
_MENTION = re.compile(
    r"\b(?:{})\b".format("|".join(re.escape(name.split(".")[0]) for name in FORBIDDEN_IMPORTS)).encode()
)


@dataclass(frozen=True)
class ImportSite:
    path: str  # relative to orchestrator/
    line: int
    module: str
    scope: str  # the enclosing classes and functions, dotted; "" at module level


# --------------------------------------------------------------------------- the import walk


def _imported(node: ast.AST) -> list[str]:
    """The absolute module names one AST node imports, lazy forms included."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module:
            return []  # relative: inside our own packages
        return [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    if isinstance(node, ast.Call) and node.args:
        func, first = node.func, node.args[0]
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in LAZY_IMPORTERS and isinstance(first, ast.Constant) and isinstance(first.value, str):
            return [first.value]
    return []


def import_sites(source: str | bytes, path: str) -> list[ImportSite]:
    """Every import in ``source``, at any depth, with the scope it sits in."""
    sites, stack = [], [(ast.parse(source, filename=path), ())]
    while stack:  # iterative: a long chained expression would overflow recursion
        node, scope = stack.pop()
        sites += [ImportSite(path, node.lineno, module, ".".join(scope)) for module in _imported(node)]
        inner = (*scope, node.name) if isinstance(node, SCOPES) else scope
        stack += [(child, inner) for child in ast.iter_child_nodes(node)]
    return sorted(sites, key=lambda site: (site.line, site.module))


def forbidden(site: ImportSite) -> str | None:
    """The forbidden name ``site`` imports, if any."""
    return next((name for name in FORBIDDEN_IMPORTS
                 if site.module == name or site.module.startswith(f"{name}.")), None)


def _python_files() -> list[Path]:
    found = []
    for root, dirs, files in os.walk(ORCH):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        found += [Path(root) / name for name in files if name.endswith(".py")]
    return sorted(found)


SAMPLE = '''
import importlib
import nltk.tokenize
from jose import jwt
from . import fi
import fine
import sklearnish
def lazy():
    from sklearn import cluster
    importlib.import_module("fi.evals")
    __import__("passlib")
    importlib.util.find_spec("pgvector")
class Holder:
    async def method(self):
        pytest.importorskip("nltk")
'''


def test_the_walk_sees_every_import_form_at_any_depth():
    hits = {(site.line, site.module, site.scope)
            for site in import_sites(SAMPLE, "sample.py") if forbidden(site)}
    assert hits == {
        (3, "nltk.tokenize", ""),
        (4, "jose", ""),
        (4, "jose.jwt", ""),
        (9, "sklearn", "lazy"),
        (9, "sklearn.cluster", "lazy"),
        (10, "fi.evals", "lazy"),
        (11, "passlib", "lazy"),
        (12, "pgvector", "lazy"),
        (15, "nltk", "Holder.method"),
    }, "a relative `from . import fi`, `fine` and `sklearnish` are not the removed packages"
    lines = SAMPLE.splitlines()
    assert all(_MENTION.search(lines[line - 1].encode()) for line, _, _ in hits), "the pre-filter keeps them"


def _offenders(files: list[Path]) -> list[str]:
    offenders = []
    for path in files:
        source = path.read_bytes()
        if not _MENTION.search(source):
            continue
        rel = path.relative_to(ORCH).as_posix()
        offenders += [f"{rel}:{site.line}: {site.module} ({FORBIDDEN_IMPORTS[forbidden(site)]})"
                      for site in import_sites(source, rel) if forbidden(site)]
    return offenders


def test_no_module_under_orchestrator_imports_a_removed_package():
    files = _python_files()
    relative = {path.relative_to(ORCH).as_posix() for path in files}
    assert "tests/test_prd262_unused_packages_go.py" in relative, "the walk covers tests/"
    assert len(files) > 1000, f"the walk found only {len(files)} files"
    offenders = _offenders(files)
    assert not offenders, "the image no longer installs these:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- requirements


def _requirements(path: Path) -> dict[str, str]:
    """PEP 503 name -> the requirement as written, comments and blank lines dropped."""
    found = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        match = _NAME.match(line)
        if match:
            found[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = line
    return found


def test_no_requirements_file_names_a_removed_distribution():
    for name in ("requirements.in", "requirements-leiden.in", "requirements-dev.txt",
                 "requirements.lock", "requirements-dev.lock"):
        assert not REMOVED_DISTS & set(_requirements(ORCH / name)), name
    railway = REMOVED_DISTS & set(_requirements(ORCH / "requirements-leiden.lock"))
    assert railway <= RAILWAY_KEEPS, railway


def test_graphifyy_is_pinned_exactly_with_and_without_the_leiden_extra():
    """PRD-262 US-022: INSTALL_GRAPH_EXTRAS picks a lock; no sed rewrites a requirement."""
    line = _requirements(ORCH / "requirements.in")["graphifyy"]
    assert _PIN.match(line) and line.startswith("graphifyy=="), line
    leiden = _requirements(ORCH / "requirements-leiden.in")["graphifyy"]
    assert leiden == line.replace("graphifyy==", "graphifyy[leiden]==", 1), leiden
    assert "sed " not in DOCKERFILE and "requirements.build.txt" not in DOCKERFILE
    version = line.split("==", 1)[1]
    for lock in ("requirements.lock", "requirements-leiden.lock"):
        assert _requirements(ORCH / lock)["graphifyy"].startswith(f"graphifyy=={version} "), lock
    assert "graspologic" not in _requirements(ORCH / "requirements.lock"), "the GHCR build has no [leiden]"
    assert "graspologic" in _requirements(ORCH / "requirements-leiden.lock"), "Railway's build keeps it"


def test_psutil_is_pinned_at_6_or_later_for_its_aarch64_wheels():
    line = _requirements(ORCH / "requirements.in")["psutil"]
    pinned = _PIN.match(line)
    assert pinned and int(pinned.group(3)) >= 6, line


# --------------------------------------------------------------------------- Dockerfile and start.sh


def _pip_installs() -> list[list[str]]:
    """The arguments of every `pip install` in the Dockerfile, continuation lines joined."""
    code = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
    joined = _CONTINUATION.sub(" ", code)
    return [shlex.split(match) for match in re.findall(r"pip install([^&;|\n]*)", joined)]


def test_the_dockerfile_never_installs_futureagi():
    assert "futureagi" not in DOCKERFILE.lower()
    # The PRD-58 leftover was a second `--no-deps` install; since PRD-262 US-022 the one
    # install is the hashed lock (with --no-deps, as a lock needs), so nothing else gets in.
    installs = _pip_installs()
    assert len(installs) == 1 and "--require-hashes" in installs[0], installs
    for lock in ("requirements.lock", "requirements-leiden.lock"):
        assert "futureagi" not in _requirements(ORCH / lock), lock


def test_pybuild_installs_wheels_only_with_the_pure_python_exceptions_last():
    installs = _pip_installs()
    assert len(installs) == 1, installs
    args = installs[0]
    assert args[args.index("-r") + 1] == "$half.txt" and args[args.index("--prefix") + 1] == "/install/$half", args
    only = [i for i, arg in enumerate(args) if arg.startswith("--only-binary")]
    no = [i for i, arg in enumerate(args) if arg.startswith("--no-binary")]
    assert len(only) == 1 and args[only[0] + 1] == ":all:", args
    assert len(no) == 1 and args[no[0] + 1] == WHEEL_EXCEPTIONS, args
    assert only[0] < no[0], "pip's --only-binary :all: cleared a --no-binary named before it"


def test_start_sh_no_longer_downloads_nltk_data():
    assert "nltk" not in (ORCH / "start.sh").read_text(encoding="utf-8").lower()


# --------------------------------------------------------------------------- the check script


def _dists_script():
    spec = importlib.util.spec_from_file_location("image_dists_absent", DISTS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_check_script_reports_installed_distributions_only():
    script = _dists_script()
    assert script.installed(["no-such-dist-prd262", "PyYAML"]) == [f"PyYAML {metadata.version('PyYAML')}"]
    assert script.main([]) == 2
    assert script.main(["no-such-dist-prd262"]) == 0
    assert script.main(["pyyaml", "no-such-dist-prd262"]) == 1


def _run_like_ci(*names: str) -> subprocess.CompletedProcess:
    """`python - NAME... < image_dists_absent.py`, as CI runs it inside an image."""
    return subprocess.run([sys.executable, "-", *names], input=DISTS_SCRIPT.read_text(encoding="utf-8"),
                          capture_output=True, text=True, check=False, timeout=60)


def test_the_check_script_runs_from_stdin_as_ci_runs_it():
    clean = _run_like_ci("no-such-dist-prd262")
    assert clean.returncode == 0, clean.stdout + clean.stderr
    dirty = _run_like_ci("no-such-dist-prd262", "PyYAML")
    assert dirty.returncode == 1 and "::error::the image has PyYAML" in dirty.stdout, dirty.stdout


# --------------------------------------------------------------------------- CI looks inside the images


def _step(workflow: str, job: str, predicate) -> tuple[int, dict]:
    steps = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]
    return next((i, step) for i, step in enumerate(steps) if predicate(step))


def _checks_the_image(step: dict, image: str) -> bool:
    return bool(re.search(
        rf'docker run --rm -i --entrypoint python {re.escape(image)} - "\$\{{names\[@\]\}}" '
        r"< scripts/ci/image_dists_absent\.py",
        step["run"],
    ))


def test_prod_image_parity_checks_the_railway_build():
    built, _ = _step("test.yml", "prod-image-parity",
                     lambda step: "docker build --target production -t parity-be orchestrator/" in step.get("run", ""))
    at, check = _step("test.yml", "prod-image-parity", lambda step: "REMOVED" in (step.get("env") or {}))
    assert built < at
    assert set(check["env"]["REMOVED"].split()) == REMOVED_DISTS - RAILWAY_KEEPS
    assert _checks_the_image(check, "parity-be"), check["run"]


def test_smoke_images_pr_checks_the_ghcr_build():
    built, build = _step("smoke-images.yml", "smoke-images-pr", lambda step: step.get("name") == "Build api (amd64)")
    assert build["with"]["build-args"].split() == ["INSTALL_GRAPH_EXTRAS=false"], "the GHCR variant"
    at, check = _step("smoke-images.yml", "smoke-images-pr", lambda step: "REMOVED" in (step.get("env") or {}))
    assert built < at
    assert set(check["env"]["REMOVED"].split()) == REMOVED_DISTS
    assert check["env"]["IMAGE"] == build["with"]["tags"]
    assert _checks_the_image(check, '"$IMAGE"'), check["run"]


def test_a_change_to_the_check_script_runs_both_checks():
    changes = _step("test.yml", "changes", lambda step: "PROD_IMAGE_PATHS" in (step.get("env") or {}))[1]
    assert re.search(changes["env"]["PROD_IMAGE_PATHS"], "scripts/ci/image_dists_absent.py")
    smoke = yaml.safe_load((WORKFLOWS / "smoke-images.yml").read_text(encoding="utf-8"))
    paths = smoke.get("on", smoke.get(True))["pull_request"]["paths"]  # PyYAML reads `on` as True
    assert "scripts/ci/image_*.py" in paths
