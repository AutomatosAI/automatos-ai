# 🗄️ Database Setup - CRITICAL STEP!

## ⚠️ YOU MUST RUN THIS BEFORE TESTING!

The database needs all the new tables for orchestration, agents, memory, and context engineering.

## Option 1: Python Script (Recommended)

```bash
cd automatos-ai

# Install dependency if needed
pip install psycopg2-binary

# Run initialization
python init_database.py
```

You'll see:
```
✅ Created table: agents
✅ Created table: task_decompositions
✅ Created table: agent_runtimes
✅ Created table: memory_items
...
✅ DATABASE INITIALIZATION SUCCESSFUL!
```

## Option 2: Shell Script

```bash
cd automatos-ai

# Make executable
chmod +x init_database.sh

# Run initialization
./init_database.sh
```

## Option 3: Direct PostgreSQL

```bash
cd automatos-ai

# Using psql directly
psql -U postgres -d orchestrator_db < orchestrator/database/init_complete_schema.sql
```

## 📋 What Gets Created

### Core Tables (17 tables)
- `agents` - AI agent definitions
- `workflows` - Workflow configurations
- `task_decompositions` - Task breakdown structures
- `task_assignments` - Agent-task mappings
- `execution_contexts` - Workflow execution state

### Agent Runtime Tables (3 tables)
- `agent_runtimes` - LLM configurations per agent
- `agent_tools` - Tool access permissions
- `agent_performance` - Performance tracking

### Context Engineering Tables (4 tables)
- `context_templates` - Prompt templates
- `context_examples` - Few-shot examples
- `context_patterns` - Reusable patterns
- `context_optimizations` - Optimization logs

### Communication Tables (4 tables)
- `agent_messages` - Inter-agent messages
- `shared_contexts` - Shared knowledge spaces
- `context_permissions` - Access control
- `collaboration_sessions` - Multi-agent sessions

### Memory Tables (4 tables)
- `memory_items` - Agent memories
- `knowledge_nodes` - Knowledge graph nodes
- `knowledge_edges` - Knowledge relationships
- `learning_outcomes` - Learning tracking

### Monitoring Tables (4 tables)
- `dashboard_configs` - Dashboard settings
- `analytics_snapshots` - Metrics snapshots
- `alert_configs` - Alert rules
- `custom_metrics` - User-defined metrics

## 🔍 Verify Installation

After running initialization, verify with:

```bash
# Using Python
python -c "
from orchestrator.config import config
import psycopg2
conn = psycopg2.connect(
    dbname=config.POSTGRES_DB,
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST
)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = \'public\'')
print(f'Total tables: {cursor.fetchone()[0]}')
"
```

Should show: `Total tables: 33+`

## ❌ Common Issues

### pgvector not installed
```
ERROR: extension "pgvector" does not exist
```

**Solution**: Install pgvector extension (optional for now)
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# macOS
brew install pgvector

# Then in PostgreSQL
psql -U postgres -c "CREATE EXTENSION vector;"
```

### Permission denied
```
ERROR: permission denied to create extension
```

**Solution**: Use superuser or grant permissions
```bash
psql -U postgres -c "ALTER USER your_user CREATEDB;"
```

### Database doesn't exist
```
FATAL: database "orchestrator_db" does not exist
```

**Solution**: Create the database first
```bash
psql -U postgres -c "CREATE DATABASE orchestrator_db;"
```

## ✅ Success Indicators

After successful initialization:
1. 33+ tables exist in the database
2. No error messages during execution
3. `init_database.py` shows all critical tables with ✅
4. You can run `test_real_decomposition.py` without database errors

## 🚀 Next Steps

Once database is initialized:

1. **Test configuration**:
   ```bash
   python test_config.py
   ```

2. **Test real task decomposition**:
   ```bash
   python test_real_decomposition.py
   ```

3. **Start the API server**:
   ```bash
   python orchestrator/main.py
   ```

## 📝 Remember

**This is a ONE-TIME setup** unless you:
- Drop the database
- Need to add new tables
- Want to reset to clean state

The schema uses `CREATE TABLE IF NOT EXISTS` so it's safe to run multiple times.
