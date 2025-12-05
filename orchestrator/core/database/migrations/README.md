# Database Migrations

This directory contains database migration scripts for the Automatos AI Platform.

## How to Run Migrations

### Option 1: Using psql (Recommended for Development)
```bash
# Run migration on local database
psql -h localhost -U postgres -d orchestrator_db -f migrations/001_add_9_stage_workflow_columns.sql

# Or if using Docker:
docker exec -i automatos_postgres psql -U postgres -d orchestrator_db < migrations/001_add_9_stage_workflow_columns.sql
```

### Option 2: Using Docker Compose
```bash
# Copy migration file into container and run
docker cp migrations/001_add_9_stage_workflow_columns.sql automatos_postgres:/tmp/
docker exec automatos_postgres psql -U postgres -d orchestrator_db -f /tmp/001_add_9_stage_workflow_columns.sql
```

### Option 3: Via Python Script
```bash
cd orchestrator
python scripts/run_migration.py --file migrations/001_add_9_stage_workflow_columns.sql
```

## Available Migrations

### `001_add_9_stage_workflow_columns.sql`
**Purpose:** Add 9-stage workflow enhancement columns to existing database

**Tables Modified:**
- `agents` - Adds quality metrics (quality_score, emergence_score, eci, etc.)
- `workflows` - Adds goal, context, priority, complexity tracking
- `workflow_executions` - Adds model tracking and execution metadata

**Safe to run:** ✅ Yes - Uses `IF NOT EXISTS` checks, can run multiple times safely

**Affects data:** ❌ No - Only adds new columns with nullable or default values

**Created:** 2025-11-24  
**Required for:** 9-stage workflow system, quality scoring, emergence detection

## Migration Naming Convention

Migrations are numbered sequentially:
- `001_description.sql` - First migration
- `002_description.sql` - Second migration
- etc.

Always create new migrations with the next available number.

## Testing Migrations

Before running on production:
1. **Test on development database** first
2. **Backup your data** (`pg_dump`)
3. **Review the migration SQL** carefully
4. **Check verification queries** at the end of migration

## Rollback

If you need to rollback a migration, drop the columns manually:
```sql
-- Example rollback for 001
ALTER TABLE agents DROP COLUMN IF EXISTS quality_score;
ALTER TABLE agents DROP COLUMN IF EXISTS emergence_score;
-- ... etc for all columns
```

## Notes

- All migrations use PostgreSQL-specific syntax
- Migrations are idempotent (safe to run multiple times)
- Always test migrations on a copy of production data first
- Document any manual steps required before/after migration
