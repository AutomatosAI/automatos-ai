"""PRD-262 US-015: packages the code never imports leave the API image.

nltk, passlib, python-jose (with its ecdsa and rsa closure), pgvector (the Python
package; the Postgres extension stays) and scikit-learn left requirements.txt. The
Dockerfile's separate `--no-deps futureagi==0.6.0` install, a PRD-58 leftover, went
too: the API calls agent-opt-worker over HTTP and never imports futureagi (its module
is `fi`). Railway's [leiden] build still gets scikit-learn through graspologic until
US-027. This suite checks:

- no module under orchestrator/, tests included, imports one of them: an AST walk over
  `import`, `from ... import` and the lazy forms (importlib.import_module, __import__,
  importlib.util.find_spec, pytest.importorskip) with a literal name, at any depth;
- requirements.txt and requirements-dev.txt name none of them, the Dockerfile never
  installs futureagi, and start.sh no longer downloads NLTK data;
- graphifyy is pinned exactly, with and without the Dockerfile's [leiden] sed, and
  psutil is pinned at 6 or later, the first release with aarch64 wheels;
- pybuild installs wheels only, with `--no-binary pysher,graspologic` after
  `--only-binary :all:`, because pip's `:all:` clears an exception named before it.

FORBIDDEN_IMPORTS is meant to grow: W3 adds camelot and cv2 (US-023),
google.generativeai (US-024), pandas outside pandas_ai_service.py's methods (US-025,
which can filter on ImportSite.scope), langchain_text_splitters and langchain_core
(US-026) and tree_sitter_language_pack (US-028). A dotted name forbids that module
and everything under it.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import ast
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")

# Import name -> the distribution it came from.
FORBIDDEN_IMPORTS = {
    "nltk": "nltk",
    "passlib": "passlib",
    "jose": "python-jose",
    "pgvector": "pgvector",
    "sklearn": "scikit-learn",
    "fi": "futureagi",
}
# Distributions no image installs: the ones above plus python-jose's closure.
REMOVED_DISTS = frozenset(FORBIDDEN_IMPORTS.values()) | {"ecdsa", "rsa"}
WHEEL_EXCEPTIONS = "pysher,graspologic"

SKIP_DIRS = frozenset({"venv", ".venv", "node_modules", "__pycache__", "site-packages"})
LAZY_IMPORTERS = frozenset({"import_module", "__import__", "find_spec", "importorskip"})
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
_PIN = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]+\])?==(\d+)(\.\d+)+$")


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


def test_no_module_under_orchestrator_imports_a_removed_package():
    files = _python_files()
    relative = {path.relative_to(ORCH).as_posix() for path in files}
    assert "tests/test_prd262_unused_packages_go.py" in relative, "the walk covers tests/"
    assert len(files) > 1000, f"the walk found only {len(files)} files"
    offenders = [
        f"{site.path}:{site.line}: {site.module} ({FORBIDDEN_IMPORTS[forbidden(site)]})"
        for path in files
        for site in import_sites(path.read_bytes(), path.relative_to(ORCH).as_posix())
        if forbidden(site)
    ]
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
    for name in ("requirements.txt", "requirements-dev.txt"):
        assert not REMOVED_DISTS & set(_requirements(ORCH / name)), name


def test_graphifyy_is_pinned_exactly_with_and_without_the_leiden_extra():
    line = _requirements(ORCH / "requirements.txt")["graphifyy"]
    assert _PIN.match(line) and line.startswith("graphifyy[leiden]=="), line
    sed = "sed 's/^graphifyy\\[leiden\\]/graphifyy/' requirements.txt > requirements.build.txt"
    assert sed in DOCKERFILE, "the GHCR and local builds drop [leiden] with this sed"
    slim = re.sub(r"^graphifyy\[leiden\]", "graphifyy", line)
    assert _PIN.match(slim) and slim.startswith("graphifyy=="), slim


def test_psutil_is_pinned_at_6_or_later_for_its_aarch64_wheels():
    line = _requirements(ORCH / "requirements.txt")["psutil"]
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
    assert all("--no-deps" not in args for args in _pip_installs())


def test_pybuild_installs_wheels_only_with_the_pure_python_exceptions_last():
    installs = _pip_installs()
    assert len(installs) == 1, installs
    args = installs[0]
    assert "requirements.build.txt" in args and "--prefix=/install" in args, args
    only = [i for i, arg in enumerate(args) if arg.startswith("--only-binary")]
    no = [i for i, arg in enumerate(args) if arg.startswith("--no-binary")]
    assert len(only) == 1 and args[only[0] + 1] == ":all:", args
    assert len(no) == 1 and args[no[0] + 1] == WHEEL_EXCEPTIONS, args
    assert only[0] < no[0], "pip's --only-binary :all: clears a --no-binary named before it"


def test_start_sh_no_longer_downloads_nltk_data():
    assert "nltk" not in (ORCH / "start.sh").read_text(encoding="utf-8").lower()
