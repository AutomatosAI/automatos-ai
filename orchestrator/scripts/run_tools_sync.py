#!/usr/bin/env python3
"""
Run the same full sync as POST /api/tools/sync.
Uses services.metadata_sync_service (DB cache + Composio API).
Loads .env via core.database.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def main():
    from core.database.database import SessionLocal
    from services.metadata_sync_service import MetadataSyncService

    db = SessionLocal()
    try:
        service = MetadataSyncService(db)
        print("Running full sync (apps + triggers + actions, orphan cleanup)...")
        result = service.run_full_sync()
        print("Done:", result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
