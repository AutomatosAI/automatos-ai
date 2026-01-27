# Triggers Metadata Issue - Summary & Fix Plan

## Problem
All apps showing `trigger_count: 0` and empty `triggers: []` arrays in the database, even though Composio apps should have triggers (e.g., Slack shows "9 Triggers" in marketplace but details show "0 Triggers").

**Status:** Addressed. Connected endpoint and frontend now return/pass triggers; sync stores them. Re-sync + hard-refresh UI if still seeing 0.

## Root Cause Analysis

### Code Flow
1. **`get_available_apps()`** (client.py:362-412)
   - Calls `_build_trigger_map()` to fetch triggers from Composio API
   - For each app, tries 3 sources:
     1. `app.meta.triggers` (from Composio SDK)
     2. `app.triggers` (direct attribute)  
     3. `trigger_map.get(app.slug.lower(), [])` (fallback from API)

2. **`_build_trigger_map()`** (client.py:414-428)
   - Calls `get_trigger_types()` to fetch all triggers
   - Builds map: `toolkit_slug -> [triggers]`
   - **If this returns empty, all apps get 0 triggers**

3. **`get_trigger_types()`** (client.py:448-486)
   - Makes API call to: `https://backend.composio.dev/api/v3/triggers_types`
   - Uses `x-api-key` header
   - Paginates through results
   - **Silently returns empty list on failure** (line 478-479)

4. **Sync Service** (metadata_sync_service.py:253-292)
   - Stores triggers in: `app_metadata["triggers"] = app.get("triggers") or []`
   - If `get_available_apps()` returns empty triggers, they're stored as empty

### Most Likely Issues

1. **API Call Failing Silently**
   - `get_trigger_types()` returns empty list on error
   - Error is logged as warning but not investigated
   - API endpoint might have changed or require different auth

2. **Trigger Map Empty**
   - If `get_trigger_types()` fails, `trigger_map` is empty
   - Fallback `trigger_map.get(app.slug.lower(), [])` returns `[]`
   - All apps end up with 0 triggers

3. **Composio SDK Not Providing Triggers**
   - `app.meta.triggers` might not exist in SDK response
   - SDK version might not support triggers yet

4. **Sync Not Run After Trigger Code Added**
   - Database might have old data from before trigger code
   - Need to re-run sync to populate triggers

## Changes Made

### 1. **Connected endpoint returns triggers** (`api/tools.py`)
- `GET /api/tools/connected` now includes `trigger_count`, `triggers`, and `description` from `ComposioAppCache` for each app. Previously these were omitted, so the Enabled tab and app details showed "0 Triggers" for connected apps.

### 2. **Frontend passes triggers for connected apps** (`frontend/lib/api-client.ts`)
- `getTools({ status: 'active' })` normalization now sets `metadata.triggers = a.triggers || []`. The details modal reads `tool.metadata.triggers`; without this, connected-app details showed no triggers.

### 3. **Composio client** (`core/composio/client.py`)
- Pagination: `get_trigger_types()` now uses `data.get("next_cursor") or data.get("nextCursor")` when advancing pages.
- Existing logging in `_build_trigger_map()` and `get_trigger_types()` (trigger counts, errors) retained.

### 4. **Sync service** (`services/metadata_sync_service.py`)
- **Orphan cleanup:** before upserting actions per app, deletes actions not in the current Composio bulk response so DB matches Composio and counts don’t accumulate.
- **Deduplication:** dedupes bulk actions by `(app_name, action_name)` before upsert to avoid `UniqueViolation` from duplicate bulk entries.
- **`action_count`:** set from actual DB count per app after sync, not from bulk length.
- Logs when storing triggers for SLACK, GMAIL, GITHUB; warns if those apps have 0 triggers.

### 5. **Scripts**
- **`scripts/db_composio_diagnostic.py`** – DB diagnostic for cache totals, GITHUB, triggers, duplicates; prints **HOW TO FIX**.
- **`scripts/run_tools_sync.py`** – runs the same full sync as `POST /api/tools/sync` (uses `.env`).

## Diagnostic Steps

### Step 1: Check API Call
Run a sync and check logs for:
```
Fetched X trigger types from Composio API
Built trigger map with X toolkits
```

If you see:
- `Fetched 0 trigger types` → API call is failing
- `Built trigger map with 0 toolkits` → No triggers in response

### Step 2: Check Error Logs
Look for:
```
❌ Failed to fetch trigger types from Composio API: status=XXX
⚠️  Authentication failed - check COMPOSIO_API_KEY
⚠️  API key doesn't have permission to fetch triggers
```

### Step 3: Test API Directly
```bash
curl -H "x-api-key: YOUR_API_KEY" \
  "https://backend.composio.dev/api/v3/triggers_types?toolkit_versions=latest&limit=10"
```

### Step 4: Check Database
```sql
-- Check if triggers are in metadata
SELECT app_name, trigger_count, 
       jsonb_array_length(COALESCE(app_metadata->'triggers', '[]'::jsonb)) as triggers_in_metadata
FROM composio_apps_cache 
WHERE app_name IN ('SLACK', 'GMAIL', 'GITHUB')
ORDER BY app_name;
```

## Recommended Fixes

### Immediate Actions

1. **Re-run Sync**
   ```bash
   POST /api/tools/sync
   ```
   Check logs to see if triggers are being fetched

2. **Check Logs**
   - Look for trigger-related warnings/errors
   - Check if `get_trigger_types()` is succeeding
   - Verify trigger_map is being populated

3. **Verify API Key**
   - Ensure `COMPOSIO_API_KEY` is set correctly
   - Verify API key has permissions to fetch triggers

### If API Call is Failing

1. **Check API Endpoint**
   - Verify `https://backend.composio.dev/api/v3/triggers_types` is correct
   - Check Composio API docs for current endpoint

2. **Check Authentication**
   - Verify API key format
   - Check if endpoint requires different auth method

3. **Add Fallback**
   - If API fails, try fetching triggers per-app
   - Use Composio SDK methods if available

### If Trigger Map is Empty

1. **Check Response Format**
   - Verify API response structure
   - Check if `toolkit.slug` exists in response
   - Ensure trigger structure matches code expectations

2. **Add Validation**
   - Log sample trigger response
   - Validate trigger structure before building map

## Testing

After fixes, verify:

1. **Sync runs successfully**
   ```bash
   POST /api/tools/sync
   # Check logs for trigger counts
   ```

2. **Database has triggers**
   ```sql
   SELECT app_name, trigger_count, 
          jsonb_array_length(app_metadata->'triggers') as triggers_count
   FROM composio_apps_cache 
   WHERE app_name = 'SLACK';
   -- Should show trigger_count > 0 and triggers_count > 0
   ```

3. **API returns triggers**
   ```bash
   GET /api/tools/marketplace?search=slack
   # Check response for triggers array
   ```

4. **UI shows triggers**
   - Open Slack app details
   - Should show triggers count > 0
   - Should list actual triggers

## Files Modified

1. **`automatos-ai/orchestrator/api/tools.py`** – Connected endpoint returns `trigger_count`, `triggers`, `description`.
2. **`automatos-ai/frontend/lib/api-client.ts`** – Connected tools include `metadata.triggers`.
3. **`automatos-ai/orchestrator/core/composio/client.py`** – `nextCursor` pagination fallback; existing trigger logging kept.
4. **`automatos-ai/orchestrator/services/metadata_sync_service.py`** – Orphan cleanup, dedup, `action_count` from DB, trigger logging.
5. **`automatos-ai/orchestrator/scripts/db_composio_diagnostic.py`** – New diagnostic script.
6. **`automatos-ai/orchestrator/scripts/run_tools_sync.py`** – New script to run full sync.

## Next Steps

1. **Re-run sync** (`python scripts/run_tools_sync.py` or `POST /api/tools/sync`) to refresh apps, triggers, actions, and clean orphans.
2. **Run diagnostic** (`python scripts/db_composio_diagnostic.py`) to confirm DB state.
3. **Hard-refresh Tools UI** (Cmd+Shift+R) if triggers still show 0; use Marketplace or Enabled (not a cached session).
4. If triggers remain missing, **check logs** for `get_trigger_types()` errors and **verify** `COMPOSIO_API_KEY`.
