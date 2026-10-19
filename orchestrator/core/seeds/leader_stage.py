"""The boot leader's seeds: run when ``seed_fingerprint.leader`` changed (PRD-262 US-031).

The lifespan's boot leader (``main._boot_phase_1_core``, under the boot advisory lock,
so one worker of N) used to run every seed below on every boot. They now run only
when the leader stage's fingerprint (``core/seeds/fingerprint.py``) differs from the
stored one. The stage is these sections plus the starters pass (``main._run_starters``,
which the background boot runs once the server listens, PRD-262 US-034); its
fingerprint is stored by :func:`finish_leader_stage` once all of it ran clean, so a
failed section runs again on the next boot, as before.

This is the only place Railway seeds: credential types and system settings are
seeded here, once per boot that runs the stage, and the entrypoint's boot loader no
longer seeds them. Packages are create-only here, so live curation of package content
survives redeploys; a full refresh stays a manual run of
``core/seeds/seed_packages.py`` (or ``docker-entrypoint.sh migrate``).

Whatever the fingerprint, the leader also runs the local first-run seed after these
sections (:func:`seed_local_first_run_on_boot`) and then the orphaned-run reaper
(``main.py``), because both depend on the database, not on seed files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from core.seeds.fingerprint import (
    LEADER,
    Section,
    run_sections,
    seed_fingerprint,
    short,
    skipped_line,
    store_fingerprint,
    stored_fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderSeeds:
    """What the boot leader's gated seeds did on this boot."""

    fingerprint: str
    ran: bool  # False: the stored fingerprint matched, so nothing ran
    clean: bool  # every section ran without raising or logging an error


def _seed_system_prompts() -> None:
    """New prompts may ship with code changes."""
    from core.database.database import get_db_session
    from core.seeds.seed_system_prompts import seed_system_prompts

    with get_db_session() as db:
        seed_system_prompts(db)


def _seed_system_settings() -> None:
    """New setting keys may ship with code changes; a user's value is never overwritten."""
    from core.database.database import get_db_session
    from core.seeds.seed_system_settings import seed_system_settings

    with get_db_session() as db:
        created, updated = seed_system_settings(db)
    if created or updated:
        logger.info("System settings seeded: %d created, %d updated", created, updated)


def _seed_credential_types() -> None:
    """Credential types, LinkedIn's override included (core/seeds/credential_type_overrides.py).

    The loader logs its own failure, which makes the stage unclean (ErrorWatch).
    """
    from core.database.load_seed_data import load_seed_data

    load_seed_data(load_credentials=True, load_platform_defaults=False)


def _sync_auto_personas() -> None:
    """PRD-226: bring existing workspaces' Auto persona up to the current doctrine.

    Hash-guarded: a customized soul is left untouched and the skip is reported;
    brand-new workspaces already carry the doctrine via the seed CREATE path. This is
    the ONLY reachable production home for the backfill: every lazy-seed caller
    (chat.py, workspaces.py, hybrid.py) reaches seed_auto_agent only when the Auto row
    is ABSENT, so its existing-row backfill branch never fires for the rows that need
    it. A new doctrine version is a change to core/seeds/seed_auto_agent.py, so the
    fingerprint brings the stage back for it.
    """
    from core.database.database import get_db_session
    from core.seeds.seed_auto_agent import sync_auto_personas

    with get_db_session() as db:
        counts = sync_auto_personas(db)  # get_db_session commits on exit
    logger.info(
        "Auto doctrine backfill: %s updated, %s skipped (customized), %s current",
        counts["updated"], counts["skipped"], counts["current"],
    )


def _seed_builtin_skills() -> None:
    """PRD-251 US-119: the built-in skills (core/seeds/skills/manifest.json).

    CREATE-only: a skill the owner synced (scripts/sync-skills.py) changes
    core/seeds/skills/, so it lands on the next deploy; an existing row is refreshed by
    content hash when it is loaded (SkillLoader), never here. Runs before the
    marketplace seeds, which resolve their agents' skills by name.
    """
    from core.database.database import get_db_session
    from core.seeds.seed_builtin_skills import seed_builtin_skills

    with get_db_session() as db:
        skills_seeded = seed_builtin_skills(db)
    logger.info(
        "Built-in skills: created %s; not synced yet %s; name held by another source %s; invalid %s",
        skills_seeded["created"], skills_seeded["not_synced"],
        skills_seeded["left_alone"], skills_seeded["invalid"],
    )


def _seed_socials_package() -> None:
    """PRD-251 US-120: the Socials package's marketplace agents and Playbooks.

    Creates what is missing and leaves existing rows alone, except the agents' skill
    links: a run links the built-in skills that exist now. After the built-in skills;
    the package row itself is the packages section below.
    """
    from core.database.database import get_db_session
    from core.seeds.seed_socials_package import seed_socials_marketplace

    with get_db_session() as db:
        socials_seeded = seed_socials_marketplace(db)
    logger.info("Socials package seed: %s", socials_seeded)


def _seed_packages_create_only() -> None:
    """PRD-230: missing packages land on the next deploy; existing rows are never touched."""
    from core.database.database import get_db_session
    from core.seeds.seed_packages import seed_packages

    with get_db_session() as db:
        pkg_created, _ = seed_packages(db, create_only=True)
    if pkg_created:
        logger.info("Marketplace packages seeded: %d created", pkg_created)


# The leader's gated sections, in order.
SECTIONS: Tuple[Section, ...] = (
    ("System prompts", _seed_system_prompts),
    ("System settings", _seed_system_settings),
    ("Credential types", _seed_credential_types),
    ("Auto persona doctrine backfill", _sync_auto_personas),
    ("Built-in skills", _seed_builtin_skills),
    ("Socials package", _seed_socials_package),
    ("Marketplace packages", _seed_packages_create_only),
)


def _stored() -> Optional[str]:
    """The stored leader fingerprint; ``None`` (so the seeds run) when it cannot be read."""
    from core.database.database import get_db_session

    try:
        with get_db_session() as db:
            return stored_fingerprint(db, LEADER)
    except Exception:
        logger.exception("seed_fingerprint.leader could not be read; the leader's seeds run")
        return None


def run_leader_seeds() -> LeaderSeeds:
    """The leader's sections when its fingerprint changed; skipped (and said so) when not.

    The fingerprint is taken before any seed runs, so an input edited during the run is
    seen by the next boot.
    """
    fingerprint = seed_fingerprint(LEADER)
    if _stored() == fingerprint:
        logger.info(skipped_line(LEADER, fingerprint))
        return LeaderSeeds(fingerprint=fingerprint, ran=False, clean=True)
    clean = run_sections(SECTIONS, logger)
    logger.info("Boot seeds completed (leader worker)")
    return LeaderSeeds(fingerprint=fingerprint, ran=True, clean=clean)


def seed_local_first_run_on_boot() -> None:
    """PRD-233 S3: the local edition's first-run content, on every boot.

    It depends on the database (a seeded row the user deleted, a new operator), not
    only on seed files, so it runs whatever the fingerprint, after the gated sections.
    Gated inside the seed on AUTH_EDITION=local (saas: a no-op); idempotent, never
    overwrites edits. ``docker-entrypoint.sh migrate`` runs it too, in the full loader.
    """
    from core.database.database import get_db_session
    from core.seeds.seed_local_first_run import seed_local_first_run

    try:
        with get_db_session() as db:
            outcome = seed_local_first_run(db)
        logger.info("Local first-run: %s", outcome)
    except Exception:
        logger.exception("Local first-run seed failed; the next boot tries again")


def finish_leader_stage(seeds: LeaderSeeds, starters_clean: bool) -> bool:
    """Store the leader stage's fingerprint once its sections and the starters ran clean.

    Returns True when it was stored. Run after the starters pass, in that pass's thread.
    """
    if not seeds.ran:
        return False
    if not (seeds.clean and starters_clean):
        logger.warning("seed_fingerprint.leader not stored: a seed section failed; the next boot runs the stage again")
        return False
    from core.database.database import get_db_session

    try:
        with get_db_session() as db:
            store_fingerprint(db, LEADER, seeds.fingerprint)
    except Exception:
        logger.exception("seed_fingerprint.leader could not be stored; the next boot runs the stage again")
        return False
    logger.info("seed_fingerprint.leader stored (fingerprint %s)", short(seeds.fingerprint))
    return True
