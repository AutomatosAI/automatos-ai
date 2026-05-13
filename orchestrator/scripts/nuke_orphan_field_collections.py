"""One-shot orphan field collection sweep.

Lists every Qdrant ``field_*`` collection, cross-references against
``orchestration_runs.config->>'field_id'``, and deletes any collection that
no live mission run still references.

Run via Railway shell on automatos-ai-api:

    python scripts/nuke_orphan_field_collections.py
    python scripts/nuke_orphan_field_collections.py --dry-run

Each field is its own Qdrant collection. Qdrant loads every collection's HNSW
into memory on startup, so a backlog of orphans crashes the pod with OOM.
The coordinator's tick drains 20 per loop, this script clears the whole
backlog in one pass.
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
    from config import config  # noqa: F401  — ensures env validation
    from core.database.database import SessionLocal
    from qdrant_client import AsyncQdrantClient

    qdrant_url = getattr(config, "QDRANT_URL", None) or "http://localhost:6333"
    qdrant_api_key = getattr(config, "QDRANT_API_KEY", None) or None

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    try:
        collections_resp = await client.get_collections()
    except Exception as exc:
        logger.error("Failed to list Qdrant collections: %s", exc)
        return 2

    field_collections = sorted(
        c.name for c in collections_resp.collections if c.name.startswith("field_")
    )
    logger.info("Found %d field_* collections in Qdrant", len(field_collections))

    if not field_collections:
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

    logger.info("Found %d referenced field_ids in orchestration_runs", len(referenced_ids))

    orphans = [
        name for name in field_collections
        if name[len("field_"):] not in referenced_ids
    ]
    logger.info("Orphan count: %d (will %s)", len(orphans), "DRY-RUN list" if dry_run else "delete")

    if dry_run:
        for name in orphans:
            print(name)
        return 0

    deleted = 0
    failures = 0
    for name in orphans:
        try:
            await client.delete_collection(name)
            deleted += 1
            if deleted % 25 == 0:
                logger.info("Progress: %d / %d", deleted, len(orphans))
        except Exception as exc:
            failures += 1
            logger.warning("Failed to delete %s: %s", name, exc)

    logger.info("Done — deleted=%d failures=%d", deleted, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List orphans without deleting",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(dry_run=args.dry_run)))
