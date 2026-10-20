"""PRD-262, the 19 Oct breakdown's rows #11 and #47: what the lock pins but the image leaves out.

orchestrator/scripts/split_lock.py leaves two sets out of the halves pybuild installs:

- LEFT_OUT, no image (row #11, about 47 MB): graphifyy requires all 25 tree-sitter
  grammars, and only graphify/extract.py imports these 12. The app parses code with the
  13 grammars requirements.in names (modules/codegraph/parsers/treesitter_parser.py) and
  never imports graphify.extract.
- DEVELOPMENT_ONLY, the development stage's own part (row #47): watchfiles, which only
  ``uvicorn --reload`` uses. The production command never reloads.

The lock keeps pinning both: CI installs it with ``pip install --require-hashes`` and no
``--no-deps``, and pip refuses a requirement of a pinned package that the lock does not
pin. This suite checks:

- LEFT_OUT is exactly the grammars the lock pins for graphifyy alone, none of them one
  the code graph parses with; DEVELOPMENT_ONLY is pinned for uvicorn alone;
- no module under orchestrator/, tests included, imports graphify.extract (or reaches it
  through graphify's lazy ``graphify.extract`` / ``graphify.collect_files``), a left-out
  grammar or watchfiles, at any depth, lazily or not;
- on the graphifyy the lock installs here: from every graphify module the app imports,
  following every import graphify makes (in functions too), graphify.extract and the
  left-out grammars are never reached;
- the development stage copies the development part and reloads, the production stage
  does neither;
- both workflows check the built image has none of them installed, with this list.

File reads and ASTs, of the repository and of the installed graphify; no Docker, no
network.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
LOCK = (ORCH / "requirements.lock").read_text(encoding="utf-8")
REQUIREMENTS_IN = (ORCH / "requirements.in").read_text(encoding="utf-8")
GRAPHIFY_LAZY = {"extract", "collect_files"}  # graphify.__getattr__ imports graphify.extract for these
_CONTINUATION = re.compile(r"\\[ \t]*\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


split_lock = _load("split_lock", ORCH / "scripts" / "split_lock.py")
LEFT_OUT_MODULES = {name.replace("-", "_") for name in split_lock.LEFT_OUT}
FORBIDDEN = {"graphify.extract", "watchfiles", *LEFT_OUT_MODULES}


def _via() -> Dict[str, Set[str]]:
    """Each pin of the lock: the sources its ``# via`` lines name."""
    found: Dict[str, Set[str]] = {}
    for block in split_lock.blocks(LOCK):
        lines = [line.strip().lstrip("#").strip() for line in block.text.splitlines() if line.strip().startswith("#")]
        found[block.name] = {word.removeprefix("via").strip() for word in lines} - {"", "via"}
    return found


def _named_in_requirements() -> Set[str]:
    names = set()
    for line in REQUIREMENTS_IN.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line.split("#", 1)[0].strip())
        if match:
            names.add(split_lock.canonical(match.group(1)))
    return names


# --------------------------------------------------------------------------- the lists


def test_left_out_is_every_grammar_the_lock_pins_for_graphifyy_alone():
    via = _via()
    grammars = {name for name in via if name.startswith("tree-sitter-")}
    for_graphify_alone = {name for name in grammars if via[name] == {"graphifyy"}}
    assert set(split_lock.LEFT_OUT) == for_graphify_alone, "a graphifyy bump added or dropped a grammar"
    assert not set(split_lock.LEFT_OUT) & _named_in_requirements(), "the code graph parses with what it names"
    assert len(split_lock.LEFT_OUT) == 12


def test_the_code_graph_parses_with_none_of_them():
    parser = (ORCH / "modules" / "codegraph" / "parsers" / "treesitter_parser.py").read_text(encoding="utf-8")
    used = set(re.findall(r"\btree_sitter_[a-z_]+", parser))
    assert used and not used & LEFT_OUT_MODULES, used


def test_development_only_is_watchfiles_pinned_for_uvicorn_alone():
    assert set(split_lock.DEVELOPMENT_ONLY) == {"watchfiles"}
    assert _via()["watchfiles"] == {"uvicorn"}
    assert "watchfiles" not in _named_in_requirements()


# --------------------------------------------------------------------------- the code


def _imported(tree: ast.AST) -> Iterator[str]:
    """Every module a tree imports, at any depth, and the lazy forms with a literal name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in ("import_module", "__import__", "find_spec") and isinstance(node.args[0].value, str):
                yield node.args[0].value
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "graphify":
            if node.attr in GRAPHIFY_LAZY:
                yield "graphify.extract"


def _forbidden(names) -> List[str]:
    return sorted({name for name in names for bad in FORBIDDEN if name == bad or name.startswith(f"{bad}.")})


def test_no_module_imports_graphify_extract_a_left_out_grammar_or_watchfiles():
    found = []
    for path in sorted(ORCH.rglob("*.py")):
        if path == Path(__file__):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found += [f"{path.relative_to(ROOT)}: {name}" for name in _forbidden(_imported(tree))]
    assert found == []


def test_the_scan_sees_every_form():
    source = ("import tree_sitter_bash\nfrom graphify.extract import extract\nfrom graphify import collect_files\n"
              "def f():\n    import importlib, graphify\n    importlib.import_module('watchfiles')\n"
              "    return graphify.extract\n")
    assert _forbidden(_imported(ast.parse(source))) == [
        "graphify.extract", "graphify.extract.extract", "tree_sitter_bash", "watchfiles"]


def _graphify_dir() -> Path:
    spec = importlib.util.find_spec("graphify")
    assert spec and spec.submodule_search_locations, "graphifyy is installed with requirements.lock"
    return Path(next(iter(spec.submodule_search_locations)))


def _graphify_imports(module: str, root: Path) -> Set[str]:
    path = root / (module.replace(".", "/") + ".py")
    path = path if path.is_file() else root / module.replace(".", "/") / "__init__.py"
    if not path.is_file():
        return set()
    names = set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = ".".join(package.split(".")[: len(package.split(".")) - node.level + 1])
            target = f"{base}.{node.module}" if node.module else base
            names |= {target, *(f"{target}.{alias.name}" for alias in node.names)}
    return names | set(_imported(ast.parse(path.read_text(encoding="utf-8"))))


def test_the_graphify_modules_the_app_imports_never_reach_extract_or_the_grammars():
    root = _graphify_dir().parent
    app = set()
    for path in ORCH.rglob("*.py"):
        if "tests" not in path.relative_to(ORCH).parts:
            app |= {name for name in _imported(ast.parse(path.read_text(encoding="utf-8")))
                    if name.split(".")[0] == "graphify"}
    assert {"graphify.cluster", "graphify.build"} <= app, sorted(app)
    seen, todo, reached = set(), sorted(app), set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        names = _graphify_imports(module, root)
        reached |= names
        todo += [name for name in names if name.split(".")[0] == "graphify"]
    assert _forbidden(reached) == [], "a graphifyy bump reaches what the image leaves out"


# --------------------------------------------------------------------------- the stages


def _stage(name: str) -> List[str]:
    code = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
    instructions = [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start + 1:end]


def test_only_the_development_stage_copies_the_development_part_and_reloads():
    development, production = _stage("development"), _stage("production")
    part = "COPY --from=pybuild /install/development /usr/local"
    assert part in development and part not in production + _stage("base")
    assert any("--reload" in line for line in development if line.startswith("CMD "))
    assert not any("--reload" in line for line in production)


# --------------------------------------------------------------------------- CI checks the image


def _steps(workflow: str, job: str) -> list:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]


def _left_out_step(steps: list) -> dict:
    found = [step for step in steps if "LEFT_OUT" in (step.get("env") or {})]
    assert len(found) == 1, found
    step = found[0]
    assert set(step["env"]["LEFT_OUT"].split()) == set(split_lock.LEFT_OUT | split_lock.DEVELOPMENT_ONLY)
    assert 'read -ra names <<< "$LEFT_OUT"' in step["run"]
    assert '"${names[@]}" < scripts/ci/image_dists_absent.py' in step["run"]
    assert "continue-on-error" not in step
    return step


def test_prod_image_parity_checks_the_railway_build():
    steps = _steps("test.yml", "prod-image-parity")
    step = _left_out_step(steps)
    built = next(i for i, each in enumerate(steps) if "-t parity-be orchestrator/" in each.get("run", ""))
    assert built < steps.index(step)
    assert "docker run --rm -i --entrypoint python parity-be -" in step["run"]


def test_images_yml_checks_every_api_build():
    steps = _steps("images.yml", "build")
    step = _left_out_step(steps)
    assert next(i for i, each in enumerate(steps) if each.get("id") == "build") < steps.index(step)
    assert "matrix.name == 'api'" in step["if"] and "!cancelled()" in step["if"]
    assert 'docker run --rm -i --entrypoint python "${IMAGE}@${DIGEST}" -' in step["run"]
