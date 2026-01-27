#!/usr/bin/env python3
"""
Diagnose composio cache: triggers, action counts, duplicates.
Uses .env for DB connection (via core.database).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text, func
from core.database.database import SessionLocal
from core.models.composio_cache import ComposioAppCache, ComposioActionCache, ComposioStatsCache

def main():
    db = SessionLocal()
    try:
        print("=" * 70)
        print("COMPOSIO CACHE DIAGNOSTIC")
        print("=" * 70)

        # ---- Stats ----
        total_actions = db.query(func.count(ComposioActionCache.id)).scalar() or 0
        total_apps = db.query(func.count(ComposioAppCache.id)).scalar() or 0
        stats = {r.stat_key: r.stat_value for r in db.query(ComposioStatsCache).all()}
        cached_total = (stats.get("total_actions") or {}).get("count")

        print("\n1. TOTALS")
        print("-" * 70)
        print(f"  composio_actions_cache rows:  {total_actions}")
        print(f"  composio_apps_cache rows:     {total_apps}")
        print(f"  stats cache total_actions:    {cached_total}")

        # ---- Duplicates: (app_name, action_name) ----
        dupes = db.execute(text("""
            SELECT app_name, action_name, COUNT(*) AS cnt
            FROM composio_actions_cache
            GROUP BY app_name, action_name
            HAVING COUNT(*) > 1
        """)).fetchall()
        print(f"\n2. DUPLICATE (app_name, action_name) ROWS")
        print("-" * 70)
        if dupes:
            print(f"  Found {len(dupes)} duplicate keys (should be 0 - unique constraint).")
            for r in dupes[:10]:
                print(f"    {r[0]} / {r[1]} -> count {r[2]}")
            if len(dupes) > 10:
                print(f"    ... and {len(dupes) - 10} more")
        else:
            print("  None (unique constraint holds).")

        # ---- Distinct (app, action) vs total ----
        distinct = db.execute(text("""
            SELECT COUNT(DISTINCT (app_name, action_name)) FROM composio_actions_cache
        """)).scalar() or 0
        print(f"\n3. DISTINCT (app_name, action_name) vs TOTAL ROWS")
        print("-" * 70)
        print(f"  Distinct keys: {distinct}")
        print(f"  Total rows:    {total_actions}")
        if total_actions > distinct:
            print(f"  -> {total_actions - distinct} duplicate rows!")

        # ---- GITHUB ----
        gh = db.query(ComposioAppCache).filter(ComposioAppCache.app_name == "GITHUB").first()
        gh_actions = db.query(func.count(ComposioActionCache.id)).filter(
            ComposioActionCache.app_name == "GITHUB"
        ).scalar() or 0
        gh_meta = (gh.app_metadata or {}) if gh else {}
        gh_triggers = gh_meta.get("triggers") or []

        print(f"\n4. GITHUB APP")
        print("-" * 70)
        if gh:
            print(f"  action_count (column):   {gh.action_count}")
            print(f"  trigger_count (column):  {gh.trigger_count}")
            print(f"  DB actions for GITHUB:   {gh_actions}")
            print(f"  metadata.triggers len:   {len(gh_triggers)}")
            if gh_triggers:
                print(f"  First trigger:          {gh_triggers[0]}")
            else:
                print("  metadata.triggers:      [] (EMPTY)")
        else:
            print("  GITHUB not found in composio_apps_cache.")

        # ---- Sample action names for GITHUB (slug vs name patterns) ----
        sample = db.execute(text("""
            SELECT action_name FROM composio_actions_cache
            WHERE app_name = 'GITHUB'
            ORDER BY action_name
            LIMIT 20
        """)).fetchall()
        print(f"\n5. GITHUB ACTION NAMES (sample)")
        print("-" * 70)
        for r in sample:
            print(f"  {r[0]}")

        # ---- Apps with 0 triggers ----
        zero_triggers = db.query(ComposioAppCache).filter(
            ComposioAppCache.trigger_count == 0,
            ComposioAppCache.status == "ACTIVE",
        ).limit(15).all()
        print(f"\n6. APPS WITH trigger_count = 0 (sample)")
        print("-" * 70)
        for a in zero_triggers:
            tl = (a.app_metadata or {}).get("triggers")
            n = len(tl) if isinstance(tl, list) else 0
            print(f"  {a.app_name}: trigger_count={a.trigger_count}, metadata.triggers len={n}")

        # ---- Top apps by action count ----
        top = db.execute(text("""
            SELECT app_name, action_count, trigger_count
            FROM composio_apps_cache
            WHERE status = 'ACTIVE'
            ORDER BY action_count DESC
            LIMIT 10
        """)).fetchall()
        print(f"\n7. TOP 10 APPS BY action_count")
        print("-" * 70)
        for r in top:
            print(f"  {r[0]}: actions={r[1]}, triggers={r[2]}")

        print("\n" + "=" * 70)
        print("HOW TO FIX")
        print("=" * 70)
        if gh and len(gh_triggers) == 0:
            print("  1. Re-sync to populate triggers:")
            print("       curl -X POST http://localhost:8000/api/tools/sync")
            print("     Then hard-refresh the Tools UI (Ctrl+Shift+R / Cmd+Shift+R).")
        elif gh and len(gh_triggers) > 0:
            print("  1. Triggers are in DB. If UI shows 0: hard-refresh Tools, ensure you use")
            print("     Marketplace/Enabled (not a cached session).")
        if total_actions > 0 and gh_actions != (gh.action_count if gh else 0):
            print("  2. action_count vs DB mismatch → run re-sync (sync now cleans orphans and")
            print("     sets action_count from actual DB).")
        print("  3. Run re-sync to reduce inflated tool count (removes orphaned actions):")
        print("       POST /api/tools/sync")
        print()
    finally:
        db.close()


if __name__ == "__main__":
    main()
