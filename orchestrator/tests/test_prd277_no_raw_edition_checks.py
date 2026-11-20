"""PRD-277 US-102: no code outside the flag's own places asks which edition it is.

Code asks ``core.editions`` for a named capability (``editions.current().billing``), or
for the name when it only reports it (``editions.current_name()``). This test scans every
Python file under ``orchestrator/`` and fails, naming the file and the line, when one:

* reads ``AUTH_EDITION``, ``IS_LOCAL_EDITION`` or ``IS_SAAS_EDITION`` as an attribute or a
  name, on any object (``config.AUTH_EDITION``, ``settings.AUTH_EDITION``, a local alias
  of the config object), or imports one (``from config import AUTH_EDITION``);
* names one in a string outside a docstring (``getattr(config, "AUTH_EDITION")``,
  ``os.environ["AUTH_EDITION"]``);
* compares something whose name says it is an edition (``edition == "local"``,
  ``self._edition != EDITION_SAAS``), or compares ``current_name()`` itself.

Exempt: ``config.py`` (the flag's own file; US-103 rewrites its boot guards),
``core/editions.py``, ``alembic/versions/`` (history is not rewritten) and the tests.
``EXEMPT_READS`` names each other file allowed to read the flag, and why.

Not caught: a name built at run time (``getattr(config, name)``, ``vars(config)``), a
comparison of a value first stored under a name with no "edition" in it
(``name = editions.current_name(); name == "local"``), shell scripts
(``docker-entrypoint.sh``), and code outside ``orchestrator/`` (``services/``,
``frontend/``). Comments and docstrings never trip it.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set, Tuple

import pytest

ORCH = Path(__file__).resolve().parents[1]
FLAG_NAMES = frozenset({"AUTH_EDITION", "IS_LOCAL_EDITION", "IS_SAAS_EDITION"})
NAME_ACCESSOR = "current_name"
EXEMPT_FILES = frozenset({"config.py", "core/editions.py"})
EXEMPT_PREFIXES = ("tests/", "alembic/versions/")
SKIPPED_DIRS = frozenset({"__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"})
EXEMPT_READS = {
    "core/seeds/fingerprint.py": (
        "Row 44 of PRD-277-EDITION-CHECKS.md: AUTH_EDITION is one of the config values the "
        "seed fingerprint hashes by name (CONFIG_NAMES, read generically by getattr). It decides "
        "nothing; test_prd262_seed_fingerprint holds that list to what the seeders read."
    ),
}
# Enough files that a scan rooted in the wrong place cannot pass by finding nothing.
MIN_SCANNED_FILES = 500


def _docstring_ids(tree: ast.AST) -> Set[int]:
    """String constants that stand alone as statements: docstrings and string comments."""
    return {id(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)}


def _says_edition(node: ast.AST) -> bool:
    """An operand whose own name says it holds an edition, or a call of the name accessor."""
    if isinstance(node, ast.Name):
        return "edition" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return "edition" in node.attr.lower()
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        return name == NAME_ACCESSOR
    return False


def _finding(node: ast.AST, docstrings: Set[int]) -> str:
    """What ``node`` does with the edition, or an empty string."""
    if isinstance(node, ast.Attribute) and node.attr in FLAG_NAMES:
        return f"reads {node.attr}"
    if isinstance(node, ast.Name) and node.id in FLAG_NAMES:
        return f"reads {node.id}"
    if isinstance(node, ast.ImportFrom):
        names = sorted(alias.name for alias in node.names if alias.name in FLAG_NAMES)
        return f"imports {', '.join(names)}" if names else ""
    if isinstance(node, ast.Constant) and node.value in FLAG_NAMES and id(node) not in docstrings:
        return f"names {node.value} in a string"
    if isinstance(node, ast.Compare) and any(map(_says_edition, [node.left, *node.comparators])):
        return "compares an edition"
    if isinstance(node, ast.Match) and _says_edition(node.subject):
        return "matches on an edition"
    return ""


def edition_checks(source: str) -> List[Tuple[int, str]]:
    """Every ``(line, what)`` in ``source`` that reads or compares the edition."""
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    found = ((getattr(node, "lineno", 0), _finding(node, docstrings)) for node in ast.walk(tree))
    return sorted((line, what) for line, what in found if what)


def _is_exempt(relative: str) -> bool:
    return relative in EXEMPT_FILES or relative.startswith(EXEMPT_PREFIXES)


def scanned_files() -> List[str]:
    """Every Python file under ``orchestrator/`` the rule covers, relative to it."""
    paths = (path for path in ORCH.rglob("*.py") if not SKIPPED_DIRS & set(path.relative_to(ORCH).parts))
    return sorted(rel for rel in (path.relative_to(ORCH).as_posix() for path in paths) if not _is_exempt(rel))


def _checks_in(relative: str) -> List[Tuple[int, str]]:
    return edition_checks((ORCH / relative).read_text(encoding="utf-8"))


def test_no_code_outside_the_flags_own_places_reads_or_compares_the_edition():
    offenders = [f"orchestrator/{rel}:{line}: {what}"
                 for rel in scanned_files() if rel not in EXEMPT_READS
                 for line, what in _checks_in(rel)]
    assert not offenders, (
        "ask core.editions for a named capability (editions.current().<capability>), or "
        "editions.current_name() to report the name, instead of reading the edition:\n"
        + "\n".join(offenders)
    )


def test_every_exemption_is_still_needed():
    for relative, reason in EXEMPT_READS.items():
        assert reason and _checks_in(relative), f"{relative} no longer reads the edition: drop its exemption"


def test_the_scan_covers_the_backend():
    files = scanned_files()
    assert len(files) >= MIN_SCANNED_FILES
    assert {"main.py", "core/auth/hybrid.py", "services/workspace_purge.py"} <= set(files)
    assert not [rel for rel in files if _is_exempt(rel)]


def test_the_edition_helpers_are_gone():
    from config import config

    assert not hasattr(config, "IS_LOCAL_EDITION") and not hasattr(config, "IS_SAAS_EDITION")


@pytest.mark.parametrize("source", [
    "from config import config\nif config.AUTH_EDITION == 'local':\n    pass\n",
    "from config import config as cfg\nedition_now = cfg.AUTH_EDITION\n",
    "def f(settings):\n    return settings.IS_LOCAL_EDITION\n",
    "from config import AUTH_EDITION\n",
    "from config import config\nx = getattr(config, 'AUTH_EDITION', 'saas')\n",
    "import os\nx = os.environ['AUTH_EDITION']\n",
    "def f(edition):\n    return edition == 'saas'\n",
    "def f(self):\n    return self._edition != EDITION_SAAS\n",
    "from core import editions\nlocal = editions.current_name() == 'local'\n",
    "def f(edition):\n    return edition in ('local', 'saas')\n",
    "def f(edition):\n    match edition:\n        case 'local':\n            return 1\n",
])
def test_the_scan_catches(source):
    assert edition_checks(source)


@pytest.mark.parametrize("source", [
    '"""The local edition (AUTH_EDITION=local) never bills."""\n',
    "def f():\n    \"\"\"Reads IS_LOCAL_EDITION no more.\"\"\"\n    return 1  # was config.AUTH_EDITION\n",
    "from core import editions\nbilled = editions.current().billing\n",
    "from core import editions\nbody = {'edition': editions.current_name()}\n",
    "MESSAGE = 'Pairing codes are a local-edition feature (AUTH_EDITION=local).'\n",
    "from core import editions\nhosted = editions.current().install_description == editions.INSTALL_HOSTED\n",
])
def test_the_scan_leaves_alone(source):
    assert edition_checks(source) == []
