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
logger = logging.getLogger("nuke-orphan-fields")


async def main(dry_run: bool) -> int:
    from config import config  # noqa: F401 — ensures env validation
    from core.database.database import SessionLocal
    from modules.context.adapters.vector_field import SHARED_COLLECTION
    from qdrant_client import AsyncQdrantClient

    qdrant_url = getattr(config, "QDRANT_URL", None) or "http://localhost:6333"
    qdrant_api_key = getattr(config, "QDRANT_API_KEY", None) or None

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # ── Pass 1: legacy per-mission collections ────────────────────
    try:
        collections_resp = await client.get_collections()
    except Exception as exc:
        logger.error("Failed to list Qdrant collections: %s", exc)
        return 2

    legacy = sorted(
        c.name for c in collections_resp.collections
        if c.name.startswith("field_") and c.name != SHARED_COLLECTION
    )
    logger.info("Legacy field_<uuid> collections: %d", len(legacy))

    if legacy:
        if dry_run:
            for name in legacy:
                print(name)
        else:
            deleted = 0
            failures = 0
            for name in legacy:
                try:
                    await client.delete_collection(name)
                    deleted += 1
                    if deleted % 25 == 0:
                        logger.info("Legacy drop progress: %d / %d", deleted, len(legacy))
                except Exception as exc:
                    failures += 1
                    logger.warning("Failed to delete %s: %s", name, exc)
            logger.info("Legacy drop done — deleted=%d failures=%d", deleted, failures)

    # ── Pass 2: orphan points in the shared collection ────────────
    try:
        exists = await client.collection_exists(SHARED_COLLECTION)
    except Exception:
        exists = False

    if not exists:
        logger.info("Shared collection %s not present yet — nothing to sweep", SHARED_COLLECTION)
        return 0

    db = SessionLocal()
    try:
        referenced_ids = {
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

    logger.info("Referenced field_ids: %d", len(referenced_ids))

    try:
        scrolled, _ = await client.scroll(
            collection_name=SHARED_COLLECTION,
            limit=100000,
            with_payload=["field_id"],
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning("Scroll on %s failed: %s", SHARED_COLLECTION, exc)
        scrolled = []

    present_ids = {
        p.payload.get("field_id") for p in scrolled
        if p.payload and p.payload.get("field_id")
    }
    orphan_ids = sorted(present_ids - referenced_ids)
    logger.info("Orphan field_ids in shared collection: %d", len(orphan_ids))

    if dry_run:
        for fid in orphan_ids:
            print(f"orphan_field_id: {fid}")
        return 0

    if orphan_ids:
        from modules.context.adapters.vector_field import VectorFieldSharedContext
        adapter = VectorFieldSharedContext()
        for fid in orphan_ids:
            try:
                await adapter.destroy_context(fid)
            except Exception as exc:
                logger.warning("Failed to destroy orphan %s: %s", fid, exc)
        logger.info("Orphan point sweep done — %d field_ids cleared", len(orphan_ids))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List orphans without deleting",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(dry_run=args.dry_run)))
