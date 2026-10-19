"""Seed fingerprints: a boot seed stage runs only when what it seeds from has changed.

PRD-262 US-031. Every boot used to redo seed work that could not have changed: the
entrypoint's seed loader (38 s on a warm local boot, most of it imports) and the
lifespan leader's seed block and starters pass. Each of the two stages now hashes its
inputs and stores the hash in ``system_settings`` (category ``boot``), so no migration
is needed:

* ``seed_fingerprint.loader``: the entrypoint's seed loader, ``python -m
  core.database.load_seed_data --boot`` (compose: ``AUTOMATOS_MIGRATE_ON_BOOT``).
  Before it starts the loader, the entrypoint pipes the stored value into ``python -m
  core.seeds.fingerprint loader``, which imports this module and ``config`` (the
  standard library and python-dotenv), not the app.
* ``seed_fingerprint.leader``: the lifespan leader's seed block and the starters pass
  after it (``core/seeds/leader_stage.py``, ``main.py``).

A fingerprint hashes what the boot seeds are made from:

* the seed files: every file under ``core/seeds/`` (this module and the leader's own
  section list included) and ``modules/documents/templates/``, plus
  ``core/database/credential_types_seed.json``, ``core/database/load_seed_data.py`` and
  ``scripts/seed_*.py``;
* the orchestrator modules the seeders import (``INPUT_FILES``). A seeder may also
  import ``core.models``, ``core.database`` and ``config`` without listing them, and
  nothing else: ``tests/test_prd262_seed_fingerprint.py`` walks the seeders' imports;
* the ``config`` values the seeders read (``CONFIG_NAMES``), which the same test holds
  to what the seeders read.

A stage stores its fingerprint only after every one of its sections finished without
raising and without logging an error on its thread (:class:`ErrorWatch`). Otherwise the
stored value stays as it was and the next boot runs the stage again, as every boot did
before. Seeds that depend on the database rather than on these inputs stay outside
both stages and run on every boot: the orphaned-run reaper, the local workspace and
operator rows, the local first-run seed and the Composio rebind.

    printf '%s\\n' "$stored" | python -m core.seeds.fingerprint loader   # exit 0: unchanged
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

# orchestrator/: /app in the image.
ORCHESTRATOR = Path(__file__).resolve().parents[2]

LOADER = "loader"
LEADER = "leader"
STAGES = (LOADER, LEADER)

SETTING_CATEGORY = "boot"
SETTING_PREFIX = "seed_fingerprint."
SETTING_DESCRIPTION = "PRD-262 US-031: the inputs this boot seed stage last ran clean with"

# Every file under these directories...
INPUT_TREES = ("core/seeds", "modules/documents/templates")
# ...these files: the loader, its fixture, and the orchestrator modules the seeders import...
INPUT_FILES = (
    "core/builtin_skills.py",
    "core/cli_runtime.py",
    "core/database/credential_types_seed.json",
    "core/database/load_seed_data.py",
    "core/llm/defaults.py",
    "core/llm/list_prices.py",
    "core/services/anthropic_catalog_sync.py",
    "modules/documents/presets.py",
    "modules/documents/seed_templates.py",
    "modules/documents/social_starters.py",
    "modules/documents/template_summary.py",
)
# ...and the seed scripts.
INPUT_GLOBS = ("scripts/seed_*.py",)
# The orchestrator modules a seeder may import without hashing them.
UNHASHED_MODULES = ("core.models", "core.database", "config")
# The config values the seeders read (seed_local_first_run, seed_system_settings,
# seed_brand_designer).
CONFIG_NAMES = (
    "AUTH_EDITION",
    "CLI_RUNTIME_ENABLED",
    "DEFAULT_WORKSPACE_ID",
    "LOCAL_OPERATOR_EMAIL",
    "NEXT_PUBLIC_API_URL",
)
# Never inputs: bytecode a run writes, and editor or Finder files on a bind mount.
SKIPPED_DIRS = ("__pycache__",)
SKIPPED_SUFFIXES = (".pyc", ".pyo")

SHORT_LENGTH = 12
UNCHANGED, CHANGED = 0, 1

# The newest row wins: two leaders writing at once can leave two rows, and every later
# write updates both.
READ_SQL = (
    "SELECT value FROM system_settings WHERE category = :category AND key = :key "
    "ORDER BY id DESC LIMIT 1"
)
UPDATE_SQL = (
    "UPDATE system_settings SET value = :value, updated_at = now() "
    "WHERE category = :category AND key = :key"
)
# Raw SQL skips the ORM's defaults, so the columns the settings API requires are set
# here (services/trial_ledger.py, 2026-09-02).
INSERT_SQL = (
    "INSERT INTO system_settings "
    "(category, key, value, value_type, description, is_sensitive, is_required, created_by) "
    "VALUES (:category, :key, :value, 'string', :description, false, false, 'system')"
)

Section = Tuple[str, Callable[[], object]]

logger = logging.getLogger(__name__)


def setting_key(stage: str) -> str:
    """The ``system_settings`` key a stage's fingerprint is stored under."""
    if stage not in STAGES:
        raise ValueError(f"unknown seed stage {stage!r}; expected one of {', '.join(STAGES)}")
    return f"{SETTING_PREFIX}{stage}"


def short(fingerprint: Optional[str]) -> str:
    """A fingerprint as the log shows it."""
    return (fingerprint or "")[:SHORT_LENGTH] or "none"


def skipped_line(stage: str, fingerprint: str) -> str:
    """What a boot logs when a stage's fingerprint matches: the smoke looks for it."""
    return f"⏭️  {setting_key(stage)}: seeds unchanged (fingerprint {short(fingerprint)}) — skipped"


def _is_input(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).parts
    if any(part in SKIPPED_DIRS or part.startswith(".") for part in relative):
        return False
    return path.suffix not in SKIPPED_SUFFIXES and path.is_file()


def input_files(root: Path = ORCHESTRATOR) -> Tuple[Path, ...]:
    """Every seed input file under ``root``, sorted. A listed file that is gone still counts."""
    found = {root / name for name in INPUT_FILES}
    for tree in INPUT_TREES:
        found.update(path for path in (root / tree).rglob("*") if _is_input(path, root))
    for pattern in INPUT_GLOBS:
        found.update(path for path in root.glob(pattern) if _is_input(path, root))
    return tuple(sorted(found))


def config_values() -> Dict[str, object]:
    """The values of ``CONFIG_NAMES`` as the seeders see them, read through ``config``."""
    from config import config

    return {name: getattr(config, name) for name in CONFIG_NAMES}


def _feed(digest, *parts: bytes) -> None:
    """Each part with its length first, so no two input sets hash the same bytes."""
    for part in parts:
        digest.update(f"{len(part)}:".encode("ascii") + part)


def _content(path: Path) -> bytes:
    """A file's bytes, or a marker for one that is missing or unreadable."""
    try:
        return path.read_bytes()
    except OSError as exc:
        return f"<unreadable: {type(exc).__name__}>".encode("utf-8")


def seed_fingerprint(stage: str, root: Path = ORCHESTRATOR,
                     settings: Optional[Mapping[str, object]] = None) -> str:
    """The SHA-256 of a stage's name, every seed input file and the seeders' config values."""
    digest = hashlib.sha256()
    _feed(digest, b"stage", setting_key(stage).encode("utf-8"))
    for path in input_files(root):
        _feed(digest, b"file", path.relative_to(root).as_posix().encode("utf-8"), _content(path))
    values = config_values() if settings is None else settings
    for name in CONFIG_NAMES:
        _feed(digest, b"config", name.encode("utf-8"), repr(values.get(name)).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- stored values


def _params(stage: str) -> Dict[str, str]:
    return {"category": SETTING_CATEGORY, "key": setting_key(stage)}


def stored_fingerprint(db, stage: str) -> Optional[str]:
    """The fingerprint a stage last ran clean with, or ``None``. ``db`` is a SQLAlchemy session."""
    from sqlalchemy import text

    row = db.execute(text(READ_SQL), _params(stage)).first()
    return row[0] if row else None


def store_fingerprint(db, stage: str, fingerprint: str) -> None:
    """Record that a stage ran clean with these inputs; the caller's session commits."""
    from sqlalchemy import text

    params = {**_params(stage), "value": fingerprint, "description": SETTING_DESCRIPTION}
    if db.execute(text(UPDATE_SQL), params).rowcount == 0:
        db.execute(text(INSERT_SQL), params)


# --------------------------------------------------------------------------- clean runs


class ErrorWatch(logging.Handler):
    """Counts the ERROR (and worse) records its own thread logs while it is installed.

    A seeder that catches its own failure still logs it; this is how a stage tells a
    clean run from one that only did not raise. Records from other threads (another
    task's, a worker thread's) are not counted.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self._thread = threading.get_ident()
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread:
            self.errors += 1

    def __enter__(self) -> "ErrorWatch":
        logging.getLogger().addHandler(self)
        return self

    def __exit__(self, *_exc) -> bool:
        logging.getLogger().removeHandler(self)
        return False


def _ran(name: str, section: Callable[[], object], log: logging.Logger) -> bool:
    try:
        section()
    except Exception:
        log.exception("%s seed failed; the next boot runs this stage again", name)
        return False
    return True


def run_sections(sections: Sequence[Section], log: logging.Logger) -> bool:
    """Run every section in order, whatever the others do; True when all ran clean."""
    with ErrorWatch() as watch:
        failed = [name for name, section in sections if not _ran(name, section, log)]
    return not failed and watch.errors == 0


# --------------------------------------------------------------------------- the entrypoint's check


def stored_from(lines: Iterable[str]) -> Optional[str]:
    """The stored value as ``psql -tA`` prints it: the first non-empty line, if any."""
    return next((line.strip() for line in lines if line.strip()), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Has a boot seed stage's fingerprint changed? Reads the stored value on stdin.")
    parser.add_argument("stage", choices=STAGES)
    return parser


def main(argv: Optional[Sequence[str]] = None, stdin: Optional[TextIO] = None) -> int:
    """Exit 0 (and log the skip line) when the stored fingerprint matches, else 1."""
    args = build_parser().parse_args(argv)
    current = seed_fingerprint(args.stage)
    stored = stored_from(stdin if stdin is not None else sys.stdin)
    if stored == current:
        logger.info(skipped_line(args.stage, current))
        return UNCHANGED
    logger.info("   %s: fingerprint %s, stored %s: the seeds run", setting_key(args.stage), short(current), short(stored))
    return CHANGED


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    sys.exit(main())
