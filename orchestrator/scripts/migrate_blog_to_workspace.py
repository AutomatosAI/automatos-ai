"""
Migrate existing blog post content from DB column to workspace .md files.

One-shot, idempotent. Skips posts that already have a file_path set.

Usage:
    python scripts/migrate_blog_to_workspace.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url
from core.workspace_client import WorkspaceClient


async def migrate():
    engine = create_engine(get_database_url())

    with engine.connect() as db:
        rows = db.execute(text(
            "SELECT id, slug, content, file_path, workspace_id "
            "FROM blog_posts WHERE content IS NOT NULL AND content != '' "
            "ORDER BY created_at"
        )).fetchall()

        print(f"Found {len(rows)} blog posts with content in DB")

        migrated = 0
        skipped = 0

        for row in rows:
            post_id, slug, content, file_path, workspace_id = row

            if file_path:
                print(f"  SKIP {slug} — already has file_path: {file_path}")
                skipped += 1
                continue

            if not workspace_id:
                print(f"  SKIP {slug} — no workspace_id")
                skipped += 1
                continue

            target_path = f"content/blog/{slug}.md"
            print(f"  Migrating {slug} → {target_path} ...", end=" ")

            ws_client = WorkspaceClient(str(workspace_id))
            result = await ws_client.write_file(target_path, content)

            if result.get("success"):
                # Update DB: set file_path, truncate content to 500 char excerpt
                trans = db.begin()
                db.execute(text(
                    "UPDATE blog_posts SET file_path = :fp, content = :excerpt "
                    "WHERE id = :id"
                ), {
                    "fp": target_path,
                    "excerpt": content[:500],
                    "id": post_id,
                })
                trans.commit()
                print("OK")
                migrated += 1
            else:
                print(f"FAILED — {result}")

        print(f"\nDone: {migrated} migrated, {skipped} skipped")


if __name__ == "__main__":
    asyncio.run(migrate())
