#!/usr/bin/env python3
"""
Seed Data Loader
================

Loads essential seed data (credential types) and the platform defaults (system
settings, models, skills, personas, marketplace catalog, plugin categories and
local-edition first-run content) into the database. Every section is
idempotent, and a failing platform-default section never stops the others.

Two command lines (from /app, as a module):

    python -m core.database.load_seed_data          # everything: `docker-entrypoint.sh migrate`
    python -m core.database.load_seed_data --boot   # the boot loader (AUTOMATOS_MIGRATE_ON_BOOT)
    python -m core.database.load_seed_data --credentials-only

PRD-262 US-031: the boot loader leaves out what the lifespan's boot leader seeds on
every boot that runs its stage (credential types, system settings, the create-only
packages, the local first-run content; core/seeds/leader_stage.py), so each is seeded
once per boot, and the full package refresh never runs on boot. The entrypoint starts
it only when ``seed_fingerprint.loader`` changed, and it stores that fingerprint once
every section ran without an error (core/seeds/fingerprint.py).
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import psycopg2
from dotenv import load_dotenv

from core.seeds.credential_type_overrides import with_overrides
from core.seeds.fingerprint import LOADER, ErrorWatch, seed_fingerprint, short, store_fingerprint

load_dotenv()

logger = logging.getLogger(__name__)

_CREDENTIAL_TYPES_FILE = Path(__file__).parent / "credential_types_seed.json"

# created_at/updated_at are set here: the model's default=func.now() is applied by
# the ORM, not the database, so raw SQL that omits them stores NULL, and the
# credential-types API (whose response requires datetimes) then answers 500. An
# update also backfills a NULL created_at left by earlier seeds.
# PRD-262 US-031: a row is updated only when the seed changes it, so an unchanged row
# keeps its updated_at. The JSON columns compare as jsonb (``json`` has no equality).
_UPSERT_CREDENTIAL_TYPE = """
    INSERT INTO credential_types
    (id, name, display_name, category, icon, description,
     schema_definition, test_endpoint, documentation_url, is_system, is_active,
     created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    ON CONFLICT (name) DO UPDATE SET
        display_name = EXCLUDED.display_name,
        category = EXCLUDED.category,
        icon = EXCLUDED.icon,
        description = EXCLUDED.description,
        schema_definition = EXCLUDED.schema_definition,
        test_endpoint = EXCLUDED.test_endpoint,
        documentation_url = EXCLUDED.documentation_url,
        created_at = COALESCE(credential_types.created_at, NOW()),
        updated_at = NOW()
    WHERE credential_types.created_at IS NULL
       OR (credential_types.display_name, credential_types.category, credential_types.icon,
           credential_types.description, credential_types.schema_definition::jsonb,
           credential_types.test_endpoint::jsonb, credential_types.documentation_url)
          IS DISTINCT FROM
          (EXCLUDED.display_name, EXCLUDED.category, EXCLUDED.icon,
           EXCLUDED.description, EXCLUDED.schema_definition::jsonb,
           EXCLUDED.test_endpoint::jsonb, EXCLUDED.documentation_url)
"""

# What one credential-type upsert did.
WRITTEN, UNCHANGED, FAILED = "written", "unchanged", "failed"

# Tables whose seeders print their own failure and return: they must hold rows after.
_SEEDED_TABLE_COUNTS = (
    ("skills", "SELECT count(*) FROM skills"),
    ("patterns", "SELECT count(*) FROM patterns"),
)


def _connect():
    """Connect to the database the app serves, resolved the way the app resolves it.

    ``get_database_url()`` tries the credential system, then DATABASE_URL, then the
    POSTGRES_* parts. Railway and Kubernetes provide only DATABASE_URL; reading only
    POSTGRES_* (and falling back to localhost) meant nothing was seeded there.
    """
    from core.database.database import get_database_url

    return psycopg2.connect(get_database_url())


def _credential_row(cred: dict) -> tuple:
    """The upsert parameters for one credential type from the seed file."""
    test_endpoint = cred.get('test_endpoint')
    return (
        cred['id'],
        cred['name'],
        cred['display_name'],
        cred['category'],
        cred.get('icon'),
        cred.get('description'),
        json.dumps(cred['schema_definition']),
        json.dumps(test_endpoint) if test_endpoint else None,
        cred.get('documentation_url'),
        cred.get('is_system', True),
        cred.get('is_active', True),
    )


def _upsert_credential_type(cursor, cred: dict) -> str:
    """Upsert one credential type inside a savepoint: WRITTEN, UNCHANGED, or FAILED (logged).

    A failed statement aborts the whole PostgreSQL transaction, so without the
    savepoint one bad row would fail every later upsert and the final commit would
    silently roll them all back. rowcount is read before RELEASE, which resets it;
    it is 0 when the row already matched the seed.
    """
    cursor.execute("SAVEPOINT credential_type")
    try:
        cursor.execute(_UPSERT_CREDENTIAL_TYPE, _credential_row(cred))
        outcome = WRITTEN if cursor.rowcount > 0 else UNCHANGED
    except Exception:
        logger.exception("Error inserting credential type %s", cred.get('name'))
        cursor.execute("ROLLBACK TO SAVEPOINT credential_type")
        return FAILED
    cursor.execute("RELEASE SAVEPOINT credential_type")
    return outcome


def _load_credential_types(conn) -> None:
    """Upsert every credential type in the seed file, overrides applied, in one transaction."""
    if not _CREDENTIAL_TYPES_FILE.exists():
        logger.warning("Credential types file not found: %s", _CREDENTIAL_TYPES_FILE)
        return
    credential_types = json.loads(_CREDENTIAL_TYPES_FILE.read_text(encoding='utf-8'))
    with conn.cursor() as cursor:
        outcomes = Counter(_upsert_credential_type(cursor, with_overrides(cred)) for cred in credential_types)
    conn.commit()
    logger.info(
        "Credential types: %d written, %d unchanged, %d failed",
        outcomes[WRITTEN], outcomes[UNCHANGED], outcomes[FAILED],
    )


def _count_credential_types(conn) -> int:
    """How many credential types the database holds after loading."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM credential_types")
        return cursor.fetchone()[0]


def _seed_system_settings() -> None:
    """System settings (PRD-25)."""
    try:
        from core.seeds.seed_system_settings import seed_system_settings
        from core.database.database import get_db_session

        with get_db_session() as db:
            created, updated = seed_system_settings(db)
        logger.info("System settings: %d created, %d updated", created, updated)
    except Exception:
        logger.exception("Error loading system settings")


def _seed_models() -> None:
    """LLM models."""
    try:
        from core.seeds.seed_models import seed_models

        seed_models()
        logger.info("LLM models seeded")
    except Exception:
        logger.exception("Error loading LLM models")


def _seed_claude_list_prices() -> None:
    """The Claude 5.5 routes' list prices: insert if absent, price if NULL (P256-FIX-T2)."""
    try:
        from core.database.database import get_db_session
        from core.seeds.seed_claude_list_prices import seed_claude_list_prices

        with get_db_session() as db:
            seed_claude_list_prices(db)
    except Exception:
        logger.exception("Error seeding Claude list prices")


def _require_seeded_tables() -> None:
    """Raise when a table whose seeder only prints its failure is still empty."""
    from core.database.database import get_db_session
    from sqlalchemy import text

    with get_db_session() as db:
        empty = [table for table, sql in _SEEDED_TABLE_COUNTS if not db.execute(text(sql)).scalar()]
    if empty:
        raise RuntimeError(f"still empty after their seeders ran: {', '.join(empty)}")


def _seed_skills_and_patterns() -> None:
    """Skills and patterns. Both seeders print a failure and return, so the tables are checked after."""
    try:
        from core.seeds.seed_skills import seed_skills, seed_patterns

        seed_skills()
        seed_patterns()
        _require_seeded_tables()
        logger.info("Skills and patterns seeded")
    except Exception:
        logger.exception("Error loading skills/patterns")


def _seed_personas() -> None:
    """Personas."""
    try:
        from core.seeds.seed_personas import seed_personas
        from core.database.database import get_db_session

        with get_db_session() as db:
            created, updated = seed_personas(db)
        logger.info("Personas: %d created, %d updated", created, updated)
    except Exception:
        logger.exception("Error loading personas")


def _seed_marketplace_agents() -> None:
    """Marketplace catalog (PRD-209 local first-run; PRD-233 S3 owns the curated refresh).

    v2 agents check by name. Starter agents DELETE+reinsert the 'Automatos Team'
    items (which would churn ids that marketplace_installs reference), so they run
    only into an EMPTY catalog.
    """
    try:
        from core.database.database import get_db_session
        from sqlalchemy import text

        with get_db_session() as db:
            catalog_rows = db.execute(text("SELECT count(*) FROM marketplace_items")).scalar() or 0
        if catalog_rows == 0:
            from scripts.seed_starter_agents import seed_starter_agents

            seed_starter_agents()
            logger.info("Starter agents seeded (empty catalog)")
        from scripts.seed_marketplace_agents_v2 import seed_marketplace_agents_v2

        seed_marketplace_agents_v2()
        logger.info("Marketplace agents v2 seeded")
    except Exception:
        logger.exception("Error loading marketplace agents")


def _seed_shopify_agents() -> None:
    """Shopify agents upsert by slug."""
    try:
        from core.seeds.seed_shopify_agents import seed_shopify_agents

        seed_shopify_agents()
        logger.info("Shopify agents seeded")
    except Exception:
        logger.exception("Error loading Shopify agents")


def _seed_packages() -> None:
    """Packages upsert by slug."""
    try:
        from core.seeds.seed_packages import seed_packages

        created, updated = seed_packages()
        logger.info("Packages: %d created, %d updated", created, updated)
    except Exception:
        logger.exception("Error loading packages")


def _seed_plugin_categories() -> None:
    """Plugin categories."""
    try:
        from core.seeds.seed_plugin_categories import seed_plugin_categories
        from core.database.database import get_db_session

        with get_db_session() as db:
            created, updated = seed_plugin_categories(db)
        logger.info("Plugin categories: %d created, %d updated", created, updated)
    except Exception:
        logger.exception("Error loading plugin categories")


def _seed_local_edition_first_run() -> None:
    """PRD-233 S3: local-edition first-run content.

    The workspace + operator rows, Auto, a starter roster, one demo Playbook and a
    welcome Deliverable. Gated INSIDE the seed on AUTH_EDITION=local +
    DEFAULT_WORKSPACE_ID (saas ⇒ no-op); idempotent-refresh, never overwrites
    edits. Package-qualified imports so the seed shares the app's ORM session and
    model objects (module mode: python -m ...). Here for `migrate`; a boot runs it
    at the end of the lifespan leader block instead (PRD-262 US-031).
    """
    try:
        from core.seeds.seed_local_first_run import seed_local_first_run
        from core.database.database import get_db_session

        with get_db_session() as db:
            outcome = seed_local_first_run(db)
        logger.info("Local first-run: %s", outcome)
    except Exception:
        logger.exception("Error loading local-edition first-run content")


# Every platform-default section, in order: `migrate` and a manual run.
_ALL_SECTIONS = (
    _seed_system_settings,
    _seed_models,
    _seed_claude_list_prices,
    _seed_skills_and_patterns,
    _seed_personas,
    _seed_marketplace_agents,
    _seed_shopify_agents,
    _seed_packages,
    _seed_plugin_categories,
    _seed_local_edition_first_run,
)
# The lifespan leader block seeds these on a boot (and the packages create-only), so the
# boot loader leaves them out (PRD-262 US-031).
_LEADER_SECTIONS = (_seed_system_settings, _seed_packages, _seed_local_edition_first_run)
BOOT_SECTIONS = tuple(section for section in _ALL_SECTIONS if section not in _LEADER_SECTIONS)


def _load_platform_defaults(sections: Sequence = _ALL_SECTIONS) -> None:
    """Every platform-default section given, in order; each one handles its own failure."""
    for section in sections:
        section()


def load_seed_data(load_credentials: bool = True, load_platform_defaults: bool = True,
                   boot: bool = False) -> bool:
    """Load the seed data; return False when the database or credential types fail.

    A failing platform-default section is logged and skipped, as before: one
    broken seeder must not stop the others. ``boot`` runs the boot loader's sections.
    """
    try:
        conn = _connect()
    except Exception:
        logger.exception("Seed data: could not connect to the database")
        return False
    try:
        if load_credentials:
            _load_credential_types(conn)
        logger.info("Credential types in the database: %d", _count_credential_types(conn))
    except Exception:
        logger.exception("Seed data: loading credential types failed")
        return False
    finally:
        conn.close()

    if load_platform_defaults:
        if boot:
            _load_platform_defaults(BOOT_SECTIONS)
        else:
            _load_platform_defaults()
    logger.info("Seed data loaded")
    return True


def _store_loader_fingerprint(fingerprint: str) -> None:
    """Record the inputs the loader ran clean with: the entrypoint skips it until they change."""
    try:
        from core.database.database import get_db_session

        with get_db_session() as db:
            store_fingerprint(db, LOADER, fingerprint)
        logger.info("seed_fingerprint.loader stored (fingerprint %s)", short(fingerprint))
    except Exception:
        logger.exception("seed_fingerprint.loader could not be stored; the next boot runs the loader again")


def _close_database_pool() -> None:
    """Close the app engine's pooled connections, so os._exit leaves none half-open."""
    database = sys.modules.get("core.database.database")
    if database is not None:
        database.engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Load seed data into database')
    parser.add_argument('--credentials-only', action='store_true', help='Load only credential types')
    parser.add_argument('--boot', action='store_true',
                        help="The boot loader: leave out what the lifespan's boot leader seeds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Load the seeds, then store seed_fingerprint.loader when every section ran clean.

    The fingerprint is taken before any seed runs, so a seed input edited during the
    run is seen by the next boot.
    """
    args = _parser().parse_args(argv)
    fingerprint = seed_fingerprint(LOADER)
    with ErrorWatch() as watch:
        loaded = load_seed_data(load_credentials=not args.boot,
                                load_platform_defaults=not args.credentials_only, boot=args.boot)
    if not loaded or watch.errors:
        logger.warning("seed_fingerprint.loader not stored: a seed section failed; the next boot runs the loader again")
    elif not args.credentials_only:
        _store_loader_fingerprint(fingerprint)
    _close_database_pool()
    return 0 if loaded else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from core.monitoring.request_log_guard import guard_request_logs
    guard_request_logs()  # no request URL or SDK request body in this script's logs (#1178)
    status = main()
    # PRD-262 US-031: the interpreter's own teardown took about 6 s on the Mac after the
    # last line was written (module and extension finalisation). Nothing it does matters
    # once the logs and the seeders' prints are flushed, so the process ends here.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
