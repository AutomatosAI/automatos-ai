"""PRD-262 US-031: a boot seed stage's fingerprint covers everything the stage seeds from.

``core/seeds/fingerprint.py`` hashes the seed inputs of each boot seed stage (the
entrypoint's loader and the lifespan leader's block): the seed files, the orchestrator
modules the seeders import and the config values they read. A stage whose stored
fingerprint matches is skipped, so an input the hash misses would leave a change
unseeded. These tests hold:

* **The hash moves with its inputs.** Editing a seed fixture, a listed import or a
  listed config value changes the fingerprint; an unrelated file, bytecode or a hidden
  file does not.
* **The seeders import nothing the hash misses.** An AST walk of ``core/seeds/``,
  ``core/database/load_seed_data.py`` and ``modules/documents/seed_templates.py`` (and
  the ``scripts/seed_*`` modules they import) fails when a seeder imports an
  orchestrator module outside ``core.models``, ``core.database`` and ``config`` that is
  not a hashed input, or reads a ``config`` attribute that is not hashed.
* **The entrypoint's check** (``python -m core.seeds.fingerprint loader``) exits 0 and
  logs the skip line only for a matching stored value.
* **A clean stage is one where no section raised or logged an error** on its thread.
* **Seeding no longer imports the LLM stack**: ``core/llm/__init__.py`` exports lazily.
"""
from __future__ import annotations

import ast
import io
import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest

from core.seeds import fingerprint as fp

_ORCH = Path(__file__).resolve().parents[1]
SETTINGS = {name: f"value-of-{name}" for name in fp.CONFIG_NAMES}

# A miniature orchestrator/: one file per kind of input, and files that are not inputs.
_TREE = {
    "core/seeds/seed_x.py": "X = 1\n",
    "core/seeds/skills/manifest.json": "{}\n",
    "core/database/credential_types_seed.json": "[]\n",
    "core/database/load_seed_data.py": "LOADER = 1\n",
    "core/database/database.py": "ENGINE = 1\n",
    "core/llm/defaults.py": "MODEL = 'a'\n",
    "core/llm/manager.py": "MANAGER = 1\n",
    "modules/documents/seed_templates.py": "STARTERS = 1\n",
    "modules/documents/templates/invoice.html": "<p>invoice</p>\n",
    "modules/documents/templates/social/photo-only.html": "<p>photo</p>\n",
    "scripts/seed_starter_agents.py": "AGENTS = 1\n",
    "scripts/run_tools_sync.py": "SYNC = 1\n",
    "core/other.py": "OTHER = 1\n",
}


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "orchestrator"
    for relative, text in _TREE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _fingerprint(root: Path, settings=None, stage: str = fp.LOADER) -> str:
    return fp.seed_fingerprint(stage, root=root, settings=SETTINGS if settings is None else settings)


# --------------------------------------------------------------------------- the inputs


@pytest.mark.parametrize("relative", [
    "core/seeds/seed_x.py",  # a seeder
    "core/seeds/skills/manifest.json",  # a seed fixture under core/seeds
    "core/database/credential_types_seed.json",  # the credential types fixture
    "core/database/load_seed_data.py",  # the loader
    "core/llm/defaults.py",  # a listed import
    "modules/documents/seed_templates.py",  # the starters' seeder
    "modules/documents/templates/social/photo-only.html",  # a starter template
    "scripts/seed_starter_agents.py",  # a seed script
])
def test_editing_a_seed_input_changes_the_fingerprint(tmp_path, relative):
    root = _tree(tmp_path)
    before = _fingerprint(root)
    (root / relative).write_text("edited\n", encoding="utf-8")
    assert _fingerprint(root) != before


@pytest.mark.parametrize("name", fp.CONFIG_NAMES)
def test_changing_a_listed_config_value_changes_the_fingerprint(tmp_path, name):
    root = _tree(tmp_path)
    assert _fingerprint(root, {**SETTINGS, name: "another value"}) != _fingerprint(root)


def test_a_new_seed_file_or_a_deleted_listed_file_changes_the_fingerprint(tmp_path):
    root = _tree(tmp_path)
    before = _fingerprint(root)
    (root / "core/seeds/seed_new.py").write_text("NEW = 1\n", encoding="utf-8")
    added = _fingerprint(root)
    assert added != before
    (root / "core/llm/defaults.py").unlink()
    assert _fingerprint(root) not in (before, added)


def test_files_that_are_not_inputs_leave_the_fingerprint_alone(tmp_path):
    """Unlisted modules, bytecode a run writes, and Finder or editor files on a bind mount."""
    root = _tree(tmp_path)
    before = _fingerprint(root)
    for relative in ("core/other.py", "core/database/database.py", "core/llm/manager.py", "scripts/run_tools_sync.py"):
        (root / relative).write_text("edited\n", encoding="utf-8")
    for relative in ("core/seeds/__pycache__/seed_x.cpython-311.pyc", "core/seeds/stale.pyc",
                     "core/seeds/.DS_Store", "modules/documents/templates/.#invoice.html",
                     "scripts/__pycache__/seed_starter_agents.cpython-311.pyc"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00 written by a run")
    assert _fingerprint(root) == before


def test_the_hash_is_the_same_wherever_the_tree_lives(tmp_path):
    """Paths are hashed relative to orchestrator/, so /app and a checkout agree."""
    assert _fingerprint(_tree(tmp_path / "a")) == _fingerprint(_tree(tmp_path / "b"))


def test_each_stage_has_its_own_fingerprint_and_key(tmp_path):
    root = _tree(tmp_path)
    assert _fingerprint(root, stage=fp.LOADER) != _fingerprint(root, stage=fp.LEADER)
    assert [fp.setting_key(stage) for stage in fp.STAGES] == ["seed_fingerprint.loader", "seed_fingerprint.leader"]
    with pytest.raises(ValueError):
        fp.setting_key("everything")


def test_every_listed_input_exists_in_this_checkout():
    """A renamed module would otherwise drop out of the hash unnoticed."""
    missing = [name for name in fp.INPUT_FILES if not (_ORCH / name).is_file()]
    missing += [tree for tree in fp.INPUT_TREES if not (_ORCH / tree).is_dir()]
    assert not missing, missing
    assert list(_ORCH.glob(fp.INPUT_GLOBS[0])), "no scripts/seed_*.py matched"


def test_the_real_tree_hashes_the_same_twice():
    assert fp.seed_fingerprint(fp.LEADER, settings=SETTINGS) == fp.seed_fingerprint(fp.LEADER, settings=SETTINGS)


def test_the_config_values_are_read_through_config():
    from config import config

    assert fp.config_values() == {name: getattr(config, name) for name in fp.CONFIG_NAMES}


# --------------------------------------------------------------------------- the entrypoint's check


def test_the_check_skips_when_the_stored_fingerprint_matches(monkeypatch, caplog):
    monkeypatch.setattr(fp, "config_values", lambda: dict(SETTINGS))
    current = fp.seed_fingerprint(fp.LOADER)
    with caplog.at_level(logging.INFO, logger=fp.__name__):
        assert fp.main(["loader"], stdin=io.StringIO(f"{current}\n")) == fp.UNCHANGED
    assert f"⏭️  seed_fingerprint.loader: seeds unchanged (fingerprint {current[:12]}) — skipped" in caplog.text


@pytest.mark.parametrize("stored", ["", "\n", "0" * 64 + "\n"])
def test_the_check_runs_the_seeds_on_anything_else(monkeypatch, caplog, stored):
    monkeypatch.setattr(fp, "config_values", lambda: dict(SETTINGS))
    with caplog.at_level(logging.INFO, logger=fp.__name__):
        assert fp.main(["loader"], stdin=io.StringIO(stored)) == fp.CHANGED
    assert "seeds unchanged" not in caplog.text
    assert "the seeds run" in caplog.text


def test_the_check_takes_a_known_stage_only():
    with pytest.raises(SystemExit):
        fp.main(["everything"], stdin=io.StringIO(""))


def test_the_check_imports_config_but_not_the_app():
    """The entrypoint runs it before every compose boot; it must stay light."""
    tree = ast.parse((_ORCH / "core" / "seeds" / "fingerprint.py").read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    local = {module for module in modules if module.split(".")[0] in _orchestrator_tops()}
    assert local == {"config"}, local
    third_party = {module.split(".")[0] for module in modules} - set(sys.stdlib_module_names) - {"__future__"}
    assert third_party <= {"config", "sqlalchemy"}, third_party  # sqlalchemy: inside the storage functions


# --------------------------------------------------------------------------- clean runs


def test_sections_that_run_clean_make_a_clean_stage():
    ran = []
    assert fp.run_sections([("a", lambda: ran.append("a")), ("b", lambda: ran.append("b"))], logging.getLogger("t"))
    assert ran == ["a", "b"]


def test_a_section_that_raises_makes_the_stage_unclean_and_the_rest_still_run(caplog):
    ran = []

    def broken():
        raise RuntimeError("the database went away")

    with caplog.at_level(logging.ERROR):
        clean = fp.run_sections([("broken", broken), ("after", lambda: ran.append("after"))], logging.getLogger("t"))
    assert clean is False
    assert ran == ["after"]
    assert "broken seed failed; the next boot runs this stage again" in caplog.text


def test_a_section_that_only_logs_an_error_makes_the_stage_unclean():
    """A seeder that catches its own failure still logs it (seed_starters_everywhere, the loader's sections)."""
    def swallowed():
        logging.getLogger("core.seeds.some_seeder").exception("one workspace could not be seeded")

    assert fp.run_sections([("swallowed", swallowed)], logging.getLogger("t")) is False


def test_warnings_and_other_threads_errors_do_not_make_the_stage_unclean():
    def warns():
        logging.getLogger("core.database.database").warning("Using environment variables for database")

    def another_threads_error():
        thread = threading.Thread(target=lambda: logging.getLogger("other").error("not this stage's"))
        thread.start()
        thread.join()

    assert fp.run_sections([("warns", warns), ("thread", another_threads_error)], logging.getLogger("t")) is True


def test_the_watch_leaves_no_handler_behind():
    handlers = list(logging.getLogger().handlers)
    with fp.ErrorWatch():
        assert len(logging.getLogger().handlers) == len(handlers) + 1
    assert logging.getLogger().handlers == handlers


# --------------------------------------------------------------------------- the seeders' imports


def _orchestrator_tops() -> Set[str]:
    """The first names of orchestrator modules: its top-level .py files and Python directories."""
    files = {path.stem for path in _ORCH.glob("*.py")}
    packages = {path.name for path in _ORCH.iterdir() if path.is_dir() and any(path.glob("*.py"))}
    return (files | packages) - {"alembic", "tests"}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(_ORCH).with_suffix("").parts)


def _absolute(module_name: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    package = module_name.split(".")[:-node.level]
    return ".".join(package + ([node.module] if node.module else []))


def _is_module(name: str) -> bool:
    relative = Path(*name.split("."))
    return (_ORCH / relative.with_suffix(".py")).is_file() or (_ORCH / relative).is_dir()


def imported_modules(source: str, module_name: str) -> List[Tuple[str, int]]:
    """Every orchestrator module a source imports, at any depth of the code, with its line."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(module_name, node)
            submodules = [f"{base}.{alias.name}" for alias in node.names if _is_module(f"{base}.{alias.name}")]
            found += [(name, node.lineno) for name in (submodules or [base])]
    return [(name, line) for name, line in found if name.split(".")[0] in _orchestrator_tops()]


def _config_names(tree: ast.AST) -> Tuple[Set[str], Set[str], Set[str]]:
    """Names bound to the config object, to the config module, and names imported from it."""
    objects, modules, imported = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "config" and not node.level:
            for alias in node.names:
                (objects if alias.name == "config" else imported).add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            modules |= {alias.asname or alias.name for alias in node.names if alias.name == "config"}
    return objects, modules, imported


def _config_owner(node: ast.expr, objects: Set[str], modules: Set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in objects
    return (isinstance(node, ast.Attribute) and node.attr == "config"
            and isinstance(node.value, ast.Name) and node.value.id in modules)


def config_reads(source: str) -> Set[str]:
    """Every config setting a source reads: ``config.X``, ``getattr(config, "X")`` or ``from config import X``."""
    tree = ast.parse(source)
    objects, modules, reads = _config_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _config_owner(node.value, objects, modules):
            reads.add(node.attr)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules \
                and node.attr != "config":
            reads.add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and node.args and _config_owner(node.args[0], objects, modules)
              and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)):
            reads.add(node.args[1].value)
    return reads


def _hashed_modules() -> Set[str]:
    return {_module_name(path) for path in fp.input_files(_ORCH) if path.suffix == ".py"}


def _allowed(module: str, hashed: Set[str]) -> bool:
    unhashed = any(module == prefix or module.startswith(f"{prefix}.") for prefix in fp.UNHASHED_MODULES)
    return unhashed or module in hashed


def _seeders() -> List[Path]:
    """The AC's walk, plus every scripts.seed_* module those files import (the loader's)."""
    paths = sorted((_ORCH / "core" / "seeds").rglob("*.py"))
    paths += [_ORCH / "core" / "database" / "load_seed_data.py", _ORCH / "modules" / "documents" / "seed_templates.py"]
    queue, seen = list(paths), set(paths)
    while queue:
        path = queue.pop()
        for module, _line in imported_modules(path.read_text(encoding="utf-8"), _module_name(path)):
            script = _ORCH / Path(*module.split(".")).with_suffix(".py")
            if module.startswith("scripts.seed_") and script.is_file() and script not in seen:
                seen.add(script)
                queue.append(script)
    return sorted(seen)


def unhashed_imports(paths: Iterable[Path]) -> List[str]:
    hashed = _hashed_modules()
    return [f"{path.relative_to(_ORCH)}:{line} imports {module}"
            for path in paths
            for module, line in imported_modules(path.read_text(encoding="utf-8"), _module_name(path))
            if not _allowed(module, hashed)]


def test_the_seeders_import_only_hashed_modules_the_orm_the_database_and_config():
    walked = _seeders()
    assert _ORCH / "scripts" / "seed_starter_agents.py" in walked, "the walk must reach the loader's seed scripts"
    offenders = unhashed_imports(walked)
    assert not offenders, (
        "a seeder imports an orchestrator module the seed fingerprint does not hash; add it to "
        f"INPUT_FILES in core/seeds/fingerprint.py (or stop importing it): {offenders}"
    )


def test_the_seeders_read_only_hashed_config_values():
    reads = {(path.relative_to(_ORCH).as_posix(), name)
             for path in _seeders() for name in config_reads(path.read_text(encoding="utf-8"))}
    unhashed = sorted(f"{path} reads config.{name}" for path, name in reads if name not in fp.CONFIG_NAMES)
    assert not unhashed, f"add these to CONFIG_NAMES in core/seeds/fingerprint.py: {unhashed}"
    assert {name for _path, name in reads} == set(fp.CONFIG_NAMES), "a hashed config value no seeder reads"


@pytest.mark.parametrize("source,module", [
    ("from core.composio.client import ComposioClient\n", "core.composio.client"),
    ("def seed():\n    import modules.socials.settings\n", "modules.socials.settings"),
    ("from core.llm import manager\n", "core.llm.manager"),
])
def test_the_walk_catches_an_unhashed_import(source, module):
    found = imported_modules(source, "core.seeds.seed_new")
    assert module in [name for name, _line in found]
    assert not all(_allowed(name, _hashed_modules()) for name, _line in found)


@pytest.mark.parametrize("source,module", [
    ("from core.models.core import Agent\n", "core.models.core"),
    ("from core.database.database import get_db_session\n", "core.database.database"),
    ("from .seed_packages import seed_packages\n", "core.seeds.seed_packages"),
    ("from core.llm.defaults import get_default_model_config\n", "core.llm.defaults"),
])
def test_the_walk_allows_the_orm_the_database_and_hashed_modules(source, module):
    found = imported_modules(source, "core.seeds.seed_new")
    assert [name for name, _line in found] == [module]
    assert _allowed(module, _hashed_modules())


@pytest.mark.parametrize("source,reads", [
    ("from config import config\nx = config.OPENAI_API_KEY\n", {"OPENAI_API_KEY"}),
    ("from config import config as cfg\nx = getattr(cfg, 'SOME_FLAG')\n", {"SOME_FLAG"}),
    ("import config\nx = config.config.AUTH_EDITION\n", {"AUTH_EDITION"}),
    ("from config import PLAN_TIERS\n", {"PLAN_TIERS"}),
    ("from config import config\nx = config.AUTH_EDITION == 'local'\n", {"AUTH_EDITION"}),
])
def test_the_walk_finds_every_config_read(source, reads):
    assert config_reads(source) == reads


# --------------------------------------------------------------------------- the LLM stack


def test_importing_a_light_llm_module_imports_no_llm_stack():
    """core/llm/__init__.py exports lazily (PEP 562): the seeders' core.llm imports stay light."""
    heavy = ("core.llm.manager", "core.llm.embedding_manager", "core.llm.rerank_manager", "core.llm.clients.base")
    code = (
        "import sys\n"
        "import core.llm.list_prices, core.llm.defaults\n"
        f"loaded = [name for name in {heavy!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=_ORCH, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_the_package_still_exports_the_llm_infrastructure():
    import core.llm
    from core.llm.embedding_manager import get_embedding_manager
    from core.llm.manager import create_llm_manager
    from core.llm.rerank_manager import RerankManager

    assert core.llm.create_llm_manager is create_llm_manager
    from core.llm import RerankManager as exported, get_embedding_manager as getter

    assert (exported, getter) == (RerankManager, get_embedding_manager)
    assert {"LLMManager", "EmbeddingManager", "get_rerank_manager"} <= set(dir(core.llm))
    assert set(core.llm.__all__) == set(core.llm._EXPORTS)
    with pytest.raises(AttributeError):
        _ = core.llm.not_an_export
