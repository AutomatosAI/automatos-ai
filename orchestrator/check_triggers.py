#!/usr/bin/env python3
"""
Quick diagnostic script to check trigger metadata in the database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.database.database import SessionLocal
from core.models.composio_cache import ComposioAppCache
from core.composio.client import get_composio_client

db = SessionLocal()

print("=" * 80)
print("TRIGGER METADATA DIAGNOSTIC")
print("=" * 80)

# Check a few apps in the database
test_apps = ["SLACK", "GMAIL", "GITHUB"]
print("\n1. Checking database for trigger data:")
print("-" * 80)

for app_name in test_apps:
    app = db.query(ComposioAppCache).filter(ComposioAppCache.app_name == app_name).first()
    if app:
        print(f"\n{app_name}:")
        print(f"  Trigger Count (column): {app.trigger_count}")
        print(f"  Metadata keys: {list((app.app_metadata or {}).keys())}")
        triggers = (app.app_metadata or {}).get("triggers", [])
        print(f"  Triggers in metadata: {len(triggers)}")
        if triggers:
            print(f"  First trigger: {triggers[0]}")
        else:
            print(f"  ⚠️  No triggers in metadata!")
    else:
        print(f"\n{app_name}: NOT FOUND in database")

# Check what Composio API returns
print("\n\n2. Checking what Composio API returns:")
print("-" * 80)

try:
    client = get_composio_client()
    apps = client.get_available_apps()
    
    for app_name in test_apps:
        app_data = next((a for a in apps if a.get("name", "").upper() == app_name), None)
        if app_data:
            print(f"\n{app_name} (from Composio API):")
            print(f"  Trigger Count: {app_data.get('trigger_count', 0)}")
            triggers = app_data.get("triggers", [])
            print(f"  Triggers array length: {len(triggers)}")
            if triggers:
                print(f"  First trigger: {triggers[0]}")
            else:
                print(f"  ⚠️  No triggers returned from API!")
        else:
            print(f"\n{app_name}: NOT FOUND in Composio API response")
    
    # Check trigger types API
    print("\n\n3. Checking trigger types API directly:")
    print("-" * 80)
    trigger_types = client.get_trigger_types()
    print(f"Total trigger types fetched: {len(trigger_types)}")
    if trigger_types:
        print(f"First trigger type: {trigger_types[0]}")
        # Check if any are for Slack
        slack_triggers = [t for t in trigger_types if t.get("toolkit", {}).get("slug", "").lower() == "slack"]
        print(f"Slack triggers found: {len(slack_triggers)}")
        if slack_triggers:
            print(f"First Slack trigger: {slack_triggers[0]}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

db.close()
