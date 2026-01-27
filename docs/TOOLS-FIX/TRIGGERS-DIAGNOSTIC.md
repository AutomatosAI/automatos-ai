# Triggers Metadata Diagnostic Report

## Issue
All apps showing `trigger_count: 0` and empty `triggers: []` arrays in the database, even though Composio apps should have triggers.

**Resolution:** See **Fixes Applied** below. Main gaps were (1) `/api/tools/connected` not returning triggers, and (2) frontend not passing `metadata.triggers` for connected apps. Both fixed; re-sync + hard-refresh UI if needed.

## Code Flow Analysis

### 1. Trigger Fetching (`core/composio/client.py`)

**Method: `get_available_apps()`** (lines 362-412)
- Calls `_build_trigger_map()` to get triggers from Composio API
- For each app, tries three sources:
  1. `app.meta.triggers` (from Composio SDK toolkit meta)
  2. `app.triggers` (direct attribute)
  3. `trigger_map.get(app.slug.lower(), [])` (fallback from API)

**Method: `_build_trigger_map()`** (lines 414-428)
- Calls `get_trigger_types()` to fetch all triggers
- Builds a map: `toolkit_slug -> [triggers]`
- **Potential Issue**: If `get_trigger_types()` fails or returns empty, map will be empty

**Method: `get_trigger_types()`** (lines 448-486)
- Makes API call to: `https://backend.composio.dev/api/v3/triggers_types`
- Uses API key authentication
- Paginates through results
- **Potential Issues**:
  - API call might be failing silently
  - API might require different authentication
  - Response format might have changed
  - Error handling returns empty list on failure

### 2. Trigger Storage (`services/metadata_sync_service.py`)

**Method: `run_full_sync()`** (lines 42-182)
- Calls `client.get_available_apps()` which should include triggers
- Calls `_upsert_app_only()` for each app
- Stores triggers in `app_metadata["triggers"]` (lines 219, 234, 273, 288)

**Method: `_upsert_app_only()`** (lines 253-292)
- Extracts triggers from app dict: `app.get("triggers") or []`
- Stores in: `app_metadata["triggers"] = app.get("triggers") or []`
- **Potential Issue**: If `get_available_apps()` returns empty triggers, they'll be stored as empty

### 3. Trigger Retrieval (`api/tools.py`)

**Endpoint: `GET /api/tools/marketplace`** (lines 69-203)
- Reads triggers from: `app_metadata.get("triggers") or []`
- Logs warning if `trigger_count > 0` but `triggers` is empty (lines 175-179)
- **This warning should appear if triggers are missing!**

## Root Cause Analysis

### Most Likely Issues:

1. **`get_trigger_types()` API Call Failing**
   - API endpoint might have changed
   - Authentication might be incorrect
   - API might be rate-limited or blocked
   - Error is caught and returns empty list (line 478-479)

2. **Trigger Map Not Being Built**
   - If `get_trigger_types()` returns empty, `trigger_map` will be empty
   - Fallback to `trigger_map.get(app.slug.lower(), [])` will return `[]`

3. **Composio SDK Not Providing Triggers**
   - `app.meta.triggers` might not exist in SDK response
   - `app.triggers` might not exist
   - SDK version might not support triggers yet

4. **Sync Not Run After Trigger Code Added**
   - If sync was run before trigger code was added, database won't have triggers
   - Need to re-run sync to populate triggers

## Diagnostic Steps

### Step 1: Check if `get_trigger_types()` is working
```python
from core.composio.client import get_composio_client
client = get_composio_client()
triggers = client.get_trigger_types()
print(f"Total triggers fetched: {len(triggers)}")
if triggers:
    print(f"Sample trigger: {triggers[0]}")
```

### Step 2: Check if trigger_map is being built
```python
client = get_composio_client()
trigger_map = client._build_trigger_map()
print(f"Trigger map size: {len(trigger_map)}")
print(f"Slack triggers: {trigger_map.get('slack', [])}")
```

### Step 3: Check what `get_available_apps()` returns
```python
client = get_composio_client()
apps = client.get_available_apps()
slack = next((a for a in apps if a.get("name", "").lower() == "slack"), None)
if slack:
    print(f"Slack trigger_count: {slack.get('trigger_count', 0)}")
    print(f"Slack triggers: {slack.get('triggers', [])}")
```

### Step 4: Check database
```sql
SELECT app_name, trigger_count, 
       app_metadata->>'triggers' as triggers_json,
       jsonb_array_length(app_metadata->'triggers') as triggers_count
FROM composio_apps_cache 
WHERE app_name = 'SLACK';
```

## Recommended Fixes

### Fix 1: Add Error Logging
Add logging to `get_trigger_types()` to see if API call is failing:
```python
if resp.status_code != 200:
    logger.error(f"Failed to fetch trigger types: {resp.status_code} {resp.text}")
    # Log the actual error response
```

### Fix 2: Verify API Endpoint
Check if the API endpoint is correct:
- Current: `https://backend.composio.dev/api/v3/triggers_types`
- Verify this is the correct endpoint in Composio docs

### Fix 3: Add Fallback Trigger Fetching
If `get_trigger_types()` fails, try alternative methods:
- Use Composio SDK's trigger methods if available
- Fetch triggers per-app instead of bulk

### Fix 4: Re-run Sync
If triggers code was added after last sync, re-run sync:
```bash
POST /api/tools/sync
```

### Fix 5: Check API Response Format
Verify the API response structure matches what the code expects:
- Check if `toolkit.slug` exists in response
- Check if trigger structure matches normalization logic

## Fixes Applied (Resolution)

### 1. **Connected endpoint not returning triggers**
- **`GET /api/tools/connected`** previously omitted `trigger_count` and `triggers`. The Tools UI uses this for the **Enabled** tab; app details showed "0 Triggers" for connected apps.
- **Fix:** `api/tools.py` now includes `trigger_count`, `triggers`, and `description` from `ComposioAppCache` when building each connected app.

### 2. **Frontend not passing triggers for connected apps**
- **`getTools({ status: 'active' })`** (api-client) normalized connected apps but did not set `metadata.triggers`.
- **Fix:** `lib/api-client.ts` now adds `triggers: a.triggers || []` to metadata for the active/connected path.

### 3. **Pagination fallback**
- **`get_trigger_types()`** only read `next_cursor`; some APIs use `nextCursor`.
- **Fix:** `core/composio/client.py` now uses `data.get("next_cursor") or data.get("nextCursor")` when paginating.

### 4. **Sync and tool-count fixes (separate but related)**
- Orphan cleanup: sync deletes actions not in the current Composio bulk response before upserting.
- Dedup: bulk actions are deduplicated by `(app_name, action_name)` before upsert to avoid `UniqueViolation`.
- `action_count` is set from actual DB count per app after sync, not from bulk length.
- Sync logs when storing triggers for SLACK, GMAIL, GITHUB.

### 5. **Diagnostic and run script**
- **`scripts/db_composio_diagnostic.py`** – inspects cache (totals, GITHUB, triggers, duplicates) and prints **HOW TO FIX**.
- **`scripts/run_tools_sync.py`** – runs the same full sync as `POST /api/tools/sync` (uses `.env`).

## Immediate Action Items

1. **Re-run sync** to refresh apps, triggers, and actions (and clean orphans):
   ```bash
   cd automatos-ai/orchestrator && source venv/bin/activate && python scripts/run_tools_sync.py
   ```
   Or `POST /api/tools/sync` with the API running.

2. **Run diagnostic** to verify DB state:
   ```bash
   python scripts/db_composio_diagnostic.py
   ```

3. **If UI still shows 0 Triggers:** hard-refresh the Tools page (Cmd+Shift+R / Ctrl+Shift+R), open app details from **Marketplace** or **Enabled** (not a stale cached view).

4. **Check logs** for `get_trigger_types()` errors if triggers remain missing; **verify** `COMPOSIO_API_KEY` and API permissions.
