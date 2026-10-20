"""PRD-262 US-030: a boot at head runs no alembic; the entrypoint reads the head statically.

``orchestrator/scripts/alembic_heads.py`` reads the code's alembic head(s) from the
migration scripts with ``ast``, importing nothing outside the standard library, and
compares them with ``alembic_version`` (on stdin, as ``psql -tA`` prints it). The
entrypoint skips ``alembic upgrade heads`` only when they are the same set.

* The static heads are alembic's own (``ScriptDirectory``, what ``alembic heads``
  prints), and so is the whole revision graph it reads.
* The reader handles merges (tuple ``down_revision``), roots, annotated assignments and
  alembic's file rules, and refuses (exit 2: alembic decides) what it cannot read.
* At head only for exactly the heads: an empty table, a database behind, one ahead of
  the code (a version it does not know) or one with an extra row all go to alembic.
* It runs with no site-packages at all (``python -S``), so it needs nothing installed.
* ``alembic.ini`` keeps alembic reading the same directory: ``script_location`` is
  ``alembic`` and there are no extra or recursive version locations.

The entrypoint's use of it is tested in ``test_entrypoint_modes.py``; the alembic-from-
zero lane checks it against ``alembic heads`` and a freshly built database, and the
smokes check the boots' logs (``scripts/ci/smoke_alembic.py``).
"""
from __future__ import annotations

import ast
import configparser
import io
import subprocess
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from scripts import alembic_heads  # noqa: E402

_SCRIPT = _ORCH / "scripts" / "alembic_heads.py"
_STDLIB_IMPORTS = {"__future__", "argparse", "ast", "re", "sys", "pathlib", "typing"}


def _alembic_script():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH / "alembic"))
    return ScriptDirectory.from_config(cfg)


def _write(versions: Path, name: str, body: str) -> None:
    versions.mkdir(exist_ok=True)
    (versions / name).write_text(body, encoding="utf-8")


def _rev(revision: str, down) -> str:
    return f"from alembic import op\n\nrevision = {revision!r}\ndown_revision = {down!r}\n"


def _forest(tmp_path: Path, revisions: dict) -> Path:
    versions = tmp_path / "versions"
    for revision, down in revisions.items():
        _write(versions, f"{revision}.py", _rev(revision, down))
    return versions


# --------------------------------------------------------------------------- the real forest


def test_the_static_heads_are_alembics():
    script = _alembic_script()
    assert alembic_heads.heads(alembic_heads.read_graph()) == tuple(sorted(script.get_heads()))


def test_the_static_graph_is_alembics():
    """Every revision and what it revises, as alembic loads them (merges included)."""
    def parents(down):
        return () if down is None else (down,) if isinstance(down, str) else tuple(down)

    loaded = {sc.revision: parents(sc.down_revision) for sc in _alembic_script().walk_revisions()}
    assert alembic_heads.read_graph() == loaded
    assert any(len(down) > 1 for down in loaded.values()), "the forest has merge revisions to cover"


def test_alembic_ini_reads_the_directory_the_static_reader_reads():
    ini = configparser.ConfigParser(interpolation=None)  # the logging formats carry %(name)s
    ini.read(_ORCH / "alembic.ini", encoding="utf-8")
    section = ini["alembic"]
    assert section["script_location"] == "alembic"
    assert alembic_heads.VERSIONS_DIR == _ORCH / "alembic" / "versions"
    assert "version_locations" not in section, "alembic would read more than alembic/versions"
    assert section.get("recursive_version_locations", "false").lower() in ("false", "0", "no")


def test_the_reader_imports_only_the_standard_library():
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= _STDLIB_IMPORTS, imported - _STDLIB_IMPORTS


def test_it_runs_with_no_site_packages(tmp_path):
    """``python -S``: no alembic, no SQLAlchemy, no app module can be imported."""
    head = alembic_heads.heads(alembic_heads.read_graph())
    proc = subprocess.run([sys.executable, "-S", str(_SCRIPT)], input="\n".join(head) + "\n",
                          capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proc.returncode == alembic_heads.AT_HEAD, proc.stdout + proc.stderr
    printed = subprocess.run([sys.executable, "-S", str(_SCRIPT), "--print"],
                             capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert printed.returncode == 0 and tuple(printed.stdout.split()) == head, printed.stderr


# --------------------------------------------------------------------------- reading a forest


def test_a_merge_revision_joins_its_parents_into_one_head(tmp_path):
    versions = _forest(tmp_path, {"base": None, "left": "base", "right": "base", "merge": ("left", "right")})
    assert alembic_heads.heads(alembic_heads.read_graph(versions)) == ("merge",)


def test_two_branches_are_two_heads_sorted(tmp_path):
    versions = _forest(tmp_path, {"base": None, "zeta": "base", "alpha": "base", "root2": None})
    assert alembic_heads.heads(alembic_heads.read_graph(versions)) == ("alpha", "root2", "zeta")


def test_a_list_down_revision_is_a_merge_too(tmp_path):
    versions = _forest(tmp_path, {"a": None, "b": None, "m": ["a", "b"]})
    assert alembic_heads.read_graph(versions)["m"] == ("a", "b")


def test_annotated_and_last_assignments_count_and_function_locals_do_not(tmp_path):
    versions = tmp_path / "versions"
    _write(versions, "base.py", _rev("base", None))
    _write(versions, "typed.py", (
        "from typing import Union\n"
        "revision: str = 'draft'\n"
        "revision: str = 'typed'\n"
        "down_revision: Union[str, None] = 'base'\n"
        "def upgrade():\n"
        "    revision = 'not_a_revision'\n"
        "    down_revision = 'typed'\n"
    ))
    assert alembic_heads.read_graph(versions) == {"base": (), "typed": ("base",)}
    assert alembic_heads.heads(alembic_heads.read_graph(versions)) == ("typed",)


def test_only_the_files_alembic_loads_are_read(tmp_path):
    versions = _forest(tmp_path, {"base": None})
    _write(versions, "__init__.py", "revision = 'init'\ndown_revision = 'base'\n")
    _write(versions, ".#base.py", "garbage that is not python (\n")
    _write(versions, "notes.txt", "revision = 'txt'\n")
    (versions / "nested").mkdir()
    _write(versions / "nested", "deeper.py", _rev("deeper", "base"))
    assert alembic_heads.read_graph(versions) == {"base": ()}


@pytest.mark.parametrize("body,problem", [
    ("revision = 'x'\ndown_revision = (\n", "x.py"),
    ("revision = PREVIOUS + '_next'\ndown_revision = None\n", "revision is not a literal"),
    ("down_revision = None\n", "no module-level revision"),
    ("revision = 'x'\n", "no module-level down_revision"),
    ("revision = 'x'\ndown_revision = 5\n", "down_revision is 5"),
    ("revision = ''\ndown_revision = None\n", "not a revision id"),
])
def test_a_script_it_cannot_read_is_refused(tmp_path, body, problem):
    versions = tmp_path / "versions"
    _write(versions, "x.py", body)
    with pytest.raises(alembic_heads.UnreadableHeads, match=problem):
        alembic_heads.heads(alembic_heads.read_graph(versions))


@pytest.mark.parametrize("revisions,problem", [
    ({"a": None, "b": "gone"}, "names no script: gone"),
    ({"a": "b", "b": "a"}, "no head"),
    ({}, "no head"),
])
def test_a_broken_graph_is_refused(tmp_path, revisions, problem):
    versions = _forest(tmp_path, revisions)
    versions.mkdir(exist_ok=True)
    with pytest.raises(alembic_heads.UnreadableHeads, match=problem):
        alembic_heads.heads(alembic_heads.read_graph(versions))


def test_a_revision_declared_twice_is_refused(tmp_path):
    versions = tmp_path / "versions"
    _write(versions, "one.py", _rev("same", None))
    _write(versions, "two.py", _rev("same", None))
    with pytest.raises(alembic_heads.UnreadableHeads, match="declared twice"):
        alembic_heads.read_graph(versions)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(alembic_heads.UnreadableHeads):
        alembic_heads.read_graph(tmp_path / "nowhere")


# --------------------------------------------------------------------------- at head


@pytest.mark.parametrize("stamped,expected", [
    ("head\n", True),
    ("  head  \n\n", True),                     # psql padding and the trailing newline
    ("", False),                                # an empty alembic_version
    ("parent\n", False),                        # behind the code
    ("from_newer_code\n", False),               # ahead of it (a rollback): alembic fails loudly
    ("head\nparent\n", False),                  # an extra row
])
def test_at_head_only_for_exactly_the_heads(stamped, expected):
    stamped_set = alembic_heads.stamped_versions(io.StringIO(stamped))
    assert alembic_heads.at_head(stamped_set, ("head",)) is expected


def test_a_merged_forest_is_at_head_with_every_head_in_any_order():
    assert alembic_heads.at_head(alembic_heads.stamped_versions(["b\n", "a\n"]), ("a", "b"))
    assert not alembic_heads.at_head(alembic_heads.stamped_versions(["a\n"]), ("a", "b"))


def test_no_heads_is_never_at_head():
    assert not alembic_heads.at_head(frozenset(), ())


# --------------------------------------------------------------------------- the command


@pytest.mark.parametrize("stamped,code", [("tip\n", 0), ("base\n", 1), ("", 1), ("newer\n", 1)])
def test_the_command_exits_by_the_comparison(tmp_path, capsys, stamped, code):
    versions = _forest(tmp_path, {"base": None, "tip": "base"})
    assert alembic_heads.main(["--versions-dir", str(versions)], stdin=io.StringIO(stamped)) == code
    assert "the code's head(s): tip" in capsys.readouterr().out


def test_the_command_prints_the_heads(tmp_path, capsys):
    versions = _forest(tmp_path, {"base": None, "left": "base", "right": "base"})
    assert alembic_heads.main(["--print", "--versions-dir", str(versions)]) == 0
    assert capsys.readouterr().out.split() == ["left", "right"]


def test_the_command_lets_alembic_decide_when_it_cannot_read(tmp_path, capsys):
    versions = tmp_path / "versions"
    _write(versions, "x.py", "revision = (\n")
    assert alembic_heads.main(["--versions-dir", str(versions)], stdin=io.StringIO("x\n")) == alembic_heads.UNREADABLE
    assert "alembic decides" in capsys.readouterr().err
