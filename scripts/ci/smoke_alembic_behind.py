"""Stamp the backend's database one revision behind head (PRD-262 US-029).

The smoke's behind-head boot starts from a database whose alembic_version is one
revision behind the code's head. Run inside the backend container, from /app, with
the container's own Python, alembic and database settings:

    docker compose exec -T backend python - < scripts/ci/smoke_alembic_behind.py

It reads the single head from the migration scripts and stamps alembic_version at
that head's parent (both parents when the head is a merge revision), running no
migration: alembic's `stamp --purge`, which empties the table first, so the stamp
needs no path from the head. The schema itself stays at head, which every revision
must survive (AGENTS.md: a migration is create_all-safe and uses IF NOT EXISTS), so
the next boot's `alembic upgrade heads` applies the head revision again: one real
alembic run, which scripts/ci/smoke_alembic.py counts.

Its alembic output goes to `docker compose exec`, never to the container's own log,
so the boots' counts are not touched. Exit 0 = stamped, 1 = not.
"""
from __future__ import annotations

import sys
from typing import List

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

ALEMBIC_INI = "alembic.ini"  # the image's WORKDIR, /app, as the entrypoint runs alembic


def head_parents(down_revision) -> List[str]:
    """A revision's ``down_revision`` as a list: none, one, or a merge revision's several."""
    if down_revision is None:
        return []
    if isinstance(down_revision, str):
        return [down_revision]
    return list(down_revision)


def main() -> int:
    config = Config(ALEMBIC_INI)
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    if len(heads) != 1:
        print(f"expected exactly one alembic head, found {list(heads)}", file=sys.stderr)
        return 1
    parents = head_parents(script.get_revision(heads[0]).down_revision)
    if not parents:
        print(f"the head {heads[0]} is the first revision: there is nothing behind it", file=sys.stderr)
        return 1
    command.stamp(config, parents, purge=True)
    print(f"alembic_version stamped at {', '.join(parents)}, one revision behind the head {heads[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
