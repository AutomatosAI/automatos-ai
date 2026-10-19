"""PRD-262 US-032: the API image ships its boot modules' bytecode, compiled in a stage of its own.

US-014 and US-017 left the image with no app bytecode, so every new container compiled
the ~900 modules `import main` loads before it could serve. The Dockerfile's compile
stage, built FROM base so the bytecode is the runtime interpreter's, copies the app and
compiles the modules orchestrator/scripts/boot_modules.py lists, unchecked-hash; the
production stage copies /app from it as its last layer, so code and bytecode stay one
layer (US-016). This suite checks:

- boot_modules.py: the static closure of main, router_manifest's routers and the
  entrypoint's seed check, on a throwaway app tree (module-level, class-body,
  conditional and relative imports count; function bodies and TYPE_CHECKING don't;
  a package wins over a module of the same name), its errors, and on this app;
- the compile stage: FROM base, the context, then one RUN that lists the boot set into
  /tmp and compiles it unchecked-hash with a fixed hash seed; no cache mount; the
  production stage's last layer is /app from it, and production stays last;
- scripts/ci/image_app_bytecode.py, which test.yml's prod-image-parity runs inside the
  built image: each .pyc must be this Python's unchecked-hash bytecode of the source
  beside it, the compiled set the boot set, and `import main` must rewrite none of it;
- the lane runs that check after the build and starts on a change to either script.

Pure file reads and temporary trees: no Docker.
"""
from __future__ import annotations

import importlib.util
import py_compile
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
_CONTINUATION = re.compile(r"\\[ \t]*\n")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


boot = _load("prd262_boot_modules", ORCH / "scripts" / "boot_modules.py")
probe = _load("prd262_image_app_bytecode", ROOT / "scripts" / "ci" / "image_app_bytecode.py")


# --------------------------------------------------------------------------- boot_modules.py

APP = {
    "main.py": (
        "import api.agents\n"
        "from core import helpers\n"
        "from router_manifest import mount\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    import typed.only\n"
        "else:\n"
        "    import plain\n"
        "try:\n"
        "    import optional_mod\n"
        "except ImportError:\n"
        "    import fallback_mod\n"
        "class App:\n"
        "    import in_class\n"
        "def later():\n"
        "    import heavy.lazy\n"
        "import json\n"
    ),
    "router_manifest.py": (
        "class RouterSpec:\n"
        "    def __init__(self, module, attr='router'): ...\n"
        "MANIFEST = (RouterSpec('api.routed'), RouterSpec('api.routed', attr='other'), RouterSpec('api.gone'))\n"
        "def mount(): ...\n"
    ),
    "api/__init__.py": "",
    "api/agents.py": "from . import shared\nfrom ..outside import nothing\n",
    "api/shared.py": "",
    "api/routed.py": "from .sub.deep import thing\n",
    "api/sub/deep.py": "",
    "core/__init__.py": "",
    "core/helpers.py": "",
    "core/seeds/__init__.py": "",
    "core/seeds/fingerprint.py": "import config\n",
    "config.py": "",
    "plain.py": "",
    "optional_mod.py": "",
    "fallback_mod.py": "",
    "in_class.py": "",
    "typed/only.py": "",
    "heavy/lazy.py": "",
    "dup/__init__.py": "",
    "dup.py": "",
    "scripts/boot_modules.py": "",
}
EXPECTED = [
    "api/__init__.py", "api/agents.py", "api/routed.py", "api/shared.py", "api/sub/deep.py",
    "config.py", "core/__init__.py", "core/helpers.py", "core/seeds/__init__.py", "core/seeds/fingerprint.py",
    "fallback_mod.py", "in_class.py", "main.py", "optional_mod.py", "plain.py", "router_manifest.py",
]


def _tree(tmp_path: Path, files: dict) -> Path:
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_the_boot_set_is_the_closure_of_the_roots_as_they_run(tmp_path):
    app = _tree(tmp_path, APP)
    assert [path.relative_to(app).as_posix() for path in boot.boot_files(app)] == EXPECTED


def test_a_package_wins_over_a_module_of_the_same_name(tmp_path):
    app = _tree(tmp_path, APP)
    assert boot.module_file(app, "dup") == app / "dup" / "__init__.py"
    assert boot.module_file(app, "json") is None, "the standard library is not ours to compile"


@pytest.mark.parametrize(("module", "level", "package", "expected"), [
    ("x", 0, "a.b", "x"),
    (None, 1, "a.b", "a.b"),
    ("c", 1, "a.b", "a.b.c"),
    ("c", 2, "a.b", "a.c"),
    ("c", 3, "a.b", None),
    (None, 1, "", None),
])
def test_relative_imports_resolve_against_the_package(module, level, package, expected):
    assert boot.absolute(module, level, package) == expected


def test_a_missing_root_or_an_unparsable_file_fails(tmp_path, capsys):
    app = _tree(tmp_path, APP)
    (app / "core" / "seeds" / "fingerprint.py").unlink()
    with pytest.raises(boot.BootModulesError, match="core.seeds.fingerprint"):
        boot.boot_files(app)
    assert boot.main(["--app", str(app)]) == 1
    assert "boot_modules: no file in" in capsys.readouterr().err
    (app / "core" / "seeds" / "fingerprint.py").write_text("", encoding="utf-8")
    (app / "api" / "shared.py").write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(boot.BootModulesError, match="shared.py"):
        boot.boot_files(app)


def test_the_command_prints_one_absolute_path_per_line(tmp_path, capsys):
    app = _tree(tmp_path, APP)
    assert boot.main(["--app", str(app)]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == [str(app.resolve() / relative) for relative in EXPECTED]
    assert f"boot_modules: {len(EXPECTED)} modules" in out.err


def test_on_this_app_the_set_holds_the_roots_and_the_routers_and_no_tests():
    files = boot.boot_files()
    relative = {path.relative_to(ORCH).as_posix() for path in files}
    assert {"main.py", "router_manifest.py", "core/seeds/fingerprint.py", "api/agents.py", "config.py"} <= relative
    routed = {boot.module_file(ORCH, name) for name in boot.manifest_modules(ORCH)} - {None}
    assert routed and routed <= set(files)
    assert not [path for path in relative if path.startswith(("tests/", "scripts/", "alembic/"))]
    assert all(path.suffix == ".py" and path.is_file() for path in files)


# --------------------------------------------------------------------------- the Dockerfile


def _instructions() -> list[str]:
    code = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    instructions = _instructions()
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start:end]


COMPILE_RUN = ("RUN python scripts/boot_modules.py > /tmp/boot-modules.txt && PYTHONHASHSEED=0 python -m compileall "
               "-q --invalidation-mode unchecked-hash -i /tmp/boot-modules.txt")


def test_the_compile_stage_copies_the_app_and_compiles_the_boot_set():
    assert _stage("compile") == ["FROM base as compile", "COPY . .", COMPILE_RUN]


def test_production_takes_app_and_bytecode_as_its_last_layer_and_stays_last():
    stages = re.findall(r"^FROM\s+(\S+)\s+as\s+(\S+)\s*$", DOCKERFILE, re.M | re.I)
    names = [name for _, name in stages]
    assert names[-1] == "production" and names.index("compile") < names.index("production")
    production = _stage("production")
    layers = [line for line in production if line.split()[0] in ("RUN", "COPY", "ADD", "WORKDIR")]
    assert layers[-1] == "COPY --from=compile --chown=1000:1000 /app /app"
    assert [line for line in production if "--from=compile" in line] == [layers[-1]]


def test_no_cache_mount_in_the_railway_built_dockerfile():
    """Railway fails a cache mount without a hard-coded service id (PRD-262 §10)."""
    assert "--mount=type=cache" not in DOCKERFILE


# --------------------------------------------------------------------------- the check inside the image

MAGIC, TAG = importlib.util.MAGIC_NUMBER, sys.implementation.cache_tag


def _compiled(tmp_path: Path, name: str = "mod", mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
              optimize: int = -1) -> tuple[Path, Path]:
    source = tmp_path / f"{name}.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    pyc = Path(py_compile.compile(str(source), invalidation_mode=mode, optimize=optimize, doraise=True))
    return source, pyc


def test_unchecked_hash_bytecode_of_its_own_source_passes(tmp_path):
    source, pyc = _compiled(tmp_path)
    assert probe.pyc_problem(pyc, MAGIC, TAG) is None
    assert probe.compiled_sources([pyc]) == {source}


def test_timestamp_and_checked_bytecode_fail(tmp_path):
    _, timestamp = _compiled(tmp_path, "stamped", py_compile.PycInvalidationMode.TIMESTAMP)
    _, checked = _compiled(tmp_path, "checked", py_compile.PycInvalidationMode.CHECKED_HASH)
    assert "is not unchecked-hash bytecode (flags 0b00)" in probe.pyc_problem(timestamp, MAGIC, TAG)
    assert "is not unchecked-hash bytecode (flags 0b11)" in probe.pyc_problem(checked, MAGIC, TAG)


def test_bytecode_of_another_version_of_the_source_fails(tmp_path):
    source, pyc = _compiled(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert "was compiled from another version of mod.py" in probe.pyc_problem(pyc, MAGIC, TAG)
    source.unlink()
    assert "has no source" in probe.pyc_problem(pyc, MAGIC, TAG)


def test_another_interpreters_or_an_optimised_or_a_stray_pyc_fails(tmp_path):
    _, pyc = _compiled(tmp_path)
    assert "another interpreter's magic number" in probe.pyc_problem(pyc, b"\x00\x00\r\n", TAG)
    _, optimised = _compiled(tmp_path, "opt", optimize=1)
    assert f"is not plain {TAG} bytecode" in probe.pyc_problem(optimised, MAGIC, TAG)
    stray = tmp_path / "stray.pyc"
    stray.write_bytes(pyc.read_bytes())
    assert "is not a __pycache__ entry" in probe.pyc_problem(stray, MAGIC, TAG)


def test_the_compiled_set_must_be_the_boot_set():
    expected = {Path("/app/main.py"), Path("/app/config.py")}
    assert probe.set_problems(set(expected), expected) == []
    missing = probe.set_problems({Path("/app/main.py")}, expected)
    assert missing == ["1 boot module(s) ship without bytecode: /app/config.py"]
    extra = probe.set_problems(expected | {Path("/app/tests/x.py")}, expected)
    assert extra == ["1 .pyc file(s) are not the compile stage's: /app/tests/x.py"]
    assert probe.set_problems(expected, set()) == ["scripts/boot_modules.py listed no module"]


def test_start_up_may_add_bytecode_but_never_rewrite_what_shipped():
    shipped, other, new = Path("/app/__pycache__/a.pyc"), Path("/app/__pycache__/b.pyc"), Path("/app/x/__pycache__/c.pyc")
    before = {shipped: (1, 10), other: (1, 20)}
    assert probe.runtime_changes(before, {**before, new: (2, 5)}) == ([], [new])
    assert probe.runtime_changes(before, {shipped: (3, 11), other: (1, 20)}) == ([shipped], [])


def test_a_whole_tree_checks_out_end_to_end(tmp_path, monkeypatch):
    """check() over a compiled tree, with the boot list and `import main` stood in for."""
    source, _ = _compiled(tmp_path, "main")
    monkeypatch.setattr(probe, "boot_set", lambda app: {source})
    monkeypatch.setattr(probe, "import_main", lambda app: None)
    assert probe.check(tmp_path) == []
    source.write_text("VALUE = 3\n", encoding="utf-8")
    assert probe.check(tmp_path) == [f"{probe.pyc_files(tmp_path)[0]} was compiled from another version of main.py"]


# --------------------------------------------------------------------------- CI


def _test_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))


def test_prod_image_parity_checks_the_bytecode_inside_the_built_image():
    steps = _test_workflow()["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    (at,) = [i for i, step in enumerate(steps) if "image_app_bytecode.py" in step.get("run", "")]
    assert built < at
    assert steps[at]["run"] == "docker run --rm -i --entrypoint python parity-be - < scripts/ci/image_app_bytecode.py"


@pytest.mark.parametrize("path", ["scripts/ci/image_app_bytecode.py", "orchestrator/scripts/boot_modules.py",
                                  "orchestrator/Dockerfile"])
def test_the_lane_starts_on_a_change_to_what_it_checks(path):
    changes = next(step for step in _test_workflow()["jobs"]["changes"]["steps"]
                   if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(changes["env"]["PROD_IMAGE_PATHS"], path)
