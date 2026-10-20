"""One-shot field memory cleanup — both legacy collections and orphan points.

Two passes:

  1. Drop any per-mission ``field_<uuid>`` Qdrant collection (the legacy
     one-collection-per-mission layout). After the PRD-108 single-collection
     refactor, all of these are dead weight and load on every restart.

  2. Delete points in the shared ``field_memory`` collection whose
     ``field_id`` is not referenced by any orchestration_runs row.

Run via Railway shell on automatos-ai-api:

    python scripts/nuke_orphan_field_collections.py
    python scripts/nuke_orphan_field_collections.py --dry-run

This is the script you want to run once after the refactor lands, to
clear the OOM-causing collection backlog.
"""
import argparse
import asyncio
import logging
import sys

from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
from core.monitoring.request_log_guard import guard_request_logs
guard_request_logs()  # no request URL or SDK request body in this script's logs (#1178)
logger = logging.getLogger("nuke-orphan-fields")


# The Qdrant this sweeps when config names none, and how many legacy drops between progress lines.
DEFAULT_QDRANT_URL = "http://localhost:6333"
PROGRESS_EVERY = 25
# Exit codes: the node could not list its collections.
EXIT_NO_LISTING = 2


async def _drop_collections(client, names: list) -> None:
    """Drop each legacy collection; a failed one is logged and the rest still go."""
    deleted = failures = 0
    for name in names:
        try:
            await client.delete_collection(name)
        except Exception as exc:  # one failed drop is counted and logged; the rest still go
            failures += 1
            logger.exception("Failed to delete %s: %s", name, exc)
            continue
        deleted += 1
        if deleted % PROGRESS_EVERY == 0:
            logger.info("Legacy drop progress: %d / %d", deleted, len(names))
    logger.info("Legacy drop done — deleted=%d failures=%d", deleted, failures)


async def _drop_legacy_collections(client, shared: str, dry_run: bool) -> bool:
    """Pass 1: the per-mission ``field_<uuid>`` collections (listed on a dry run, dropped
    otherwise). False when the node can't list its collections."""
    try:
        collections_resp = await client.get_collections()
    except Exception as exc:  # the run stops with EXIT_NO_LISTING and says why
        logger.exception("Failed to list Qdrant collections: %s", exc)
        return False
    legacy = sorted(
        c.name for c in collections_resp.collections
        if c.name.startswith("field_") and c.name != shared
    )
    logger.info("Legacy field_<uuid> collections: %d", len(legacy))
    if dry_run:
        for name in legacy:
            print(name)
    elif legacy:
        await _drop_collections(client, legacy)
    return True


def _referenced_field_ids() -> set:
    """Every field_id an orchestration run still names."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        return {
            row[0] for row in db.execute(
                text(
                    "SELECT config->>'field_id' FROM orchestration_runs "
                    "WHERE config->>'field_id' IS NOT NULL"
                )
            ).fetchall()
            if row[0]
        }
    finally:
        db.close()


async def _present_field_ids(client, shared: str) -> set:
    """Every field_id with a point in the shared collection; none when the scroll fails."""
    try:
        scrolled, _ = await client.scroll(
            collection_name=shared,
            limit=100000,
            with_payload=["field_id"],
            with_vectors=False,
        )
    except Exception as exc:  # logged; the sweep then finds no orphan and deletes nothing
        logger.exception("Scroll on %s failed: %s", shared, exc)
        scrolled = []
    return {
        p.payload.get("field_id") for p in scrolled
        if p.payload and p.payload.get("field_id")
    }


async def _destroy_fields(orphan_ids: list) -> None:
    """Delete every point of each orphan field through the field adapter."""
    from modules.context.adapters.vector_field import VectorFieldSharedContext

    adapter = VectorFieldSharedContext()
    for fid in orphan_ids:
        try:
            await adapter.destroy_context(fid)
        except Exception as exc:  # one failed field is logged; the sweep goes on
            logger.exception("Failed to destroy orphan %s: %s", fid, exc)
    logger.info("Orphan point sweep done — %d field_ids cleared", len(orphan_ids))


async def _sweep_orphan_points(client, shared: str, dry_run: bool) -> None:
    """Pass 2: the points in the shared collection whose field no orchestration run names."""
    try:
        exists = await client.collection_exists(shared)
    except Exception:  # an unreachable collection is swept as an absent one, as before, and logged
        logger.exception("collection_exists failed on %s", shared)
        exists = False
    if not exists:
        logger.info("Shared collection %s not present yet — nothing to sweep", shared)
        return
    referenced_ids = _referenced_field_ids()
    logger.info("Referenced field_ids: %d", len(referenced_ids))
    orphan_ids = sorted(await _present_field_ids(client, shared) - referenced_ids)
    logger.info("Orphan field_ids in shared collection: %d", len(orphan_ids))
    if dry_run:
        for fid in orphan_ids:
            print(f"orphan_field_id: {fid}")
    elif orphan_ids:
        await _destroy_fields(orphan_ids)


async def main(dry_run: bool) -> int:
    from config import config  # noqa: F401 — ensures env validation
    from core.qdrant import QdrantRestClient
    from modules.context.adapters.vector_field import SHARED_COLLECTION

    qdrant_url = getattr(config, "QDRANT_URL", None) or DEFAULT_QDRANT_URL
    qdrant_api_key = getattr(config, "QDRANT_API_KEY", None) or None
    client = QdrantRestClient(url=qdrant_url, api_key=qdrant_api_key)
    if not await _drop_legacy_collections(client, SHARED_COLLECTION, dry_run):
        return EXIT_NO_LISTING
    await _sweep_orphan_points(client, SHARED_COLLECTION, dry_run)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List orphans without deleting",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(dry_run=args.dry_run)))
