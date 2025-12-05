# 🛠️ Tools Module - The Tool Registry Revolution

> **"One tool, infinite uses. Zero duplication."**

---

## 💡 The Problem

**Every AI framework does this:**
```python
# In agent file
def web_search(...): ...

# In workflow file  
def web_search(...): ...  # Copy-pasted!

# In API file
def web_search(...): ...  # Again!?
```

**Result:** 3x the code, 3x the bugs, 3x the maintenance. 😤

---

## ✨ The Automatos Solution

### **The Tool Registry Pattern**

**Define once. Use everywhere. Forever.**

```python
# Define in ONE place
@tool_registry.register(
    category=ToolCategory.RESEARCH,
    name="web_search",
    description="Search the web for information"
)
async def search_web(query: str, max_results: int = 10) -> dict:
    """Your implementation"""
    return search_results
```

**Use ANYWHERE:**
```python
# In agents
agent.use_tool("web_search", {"query": "AI news"})

# In workflows
workflow.add_step("web_search", {"query": "...", "max_results": 5})

# In API endpoints
POST /api/tools/execute {"tool": "web_search", "args": {...}}

# In chat
# LLM automatically has access via function calling!
```

**ONE source of truth. ZERO duplication.** 🎯

---

## 🏗️ Architecture

```
modules/tools/
├── __init__.py              # Tool registry export
├── registry.py              # ToolRegistry class
├── executor.py              # UnifiedToolExecutor
├── base.py                  # BaseTool interface
├── categories.py            # Tool categories enum
├── implementations/         # Where tools live
│   ├── research/            # Web search, knowledge lookup
│   ├── database/            # SQL query, data analysis
│   ├── file_ops/            # File read/write, git ops
│   ├── communication/       # Email, Slack, notifications
│   └── ai_services/         # LLM calls, embeddings
└── formatting/              # Result formatters
    ├── result_formatter.py  # Standardize tool outputs
    └── ui_formatter.py      # Format for frontend
```

---

## 🎯 Tool Categories

We organize tools by purpose:

| Category | Purpose | Examples |
|----------|---------|----------|
| **RESEARCH** | Find information | `web_search`, `wikipedia`, `arxiv` |
| **DATABASE_TOOLS** | Data operations | `query_database`, `run_sql`, `data_analysis` |
| **FILE_OPERATIONS** | File management | `read_file`, `write_file`, `git_commit` |
| **COMMUNICATION** | Send messages | `send_email`, `slack_message`, `notify` |
| **AI_SERVICES** | AI capabilities | `generate_image`, `transcribe_audio` |
| **CODE_OPS** | Code manipulation | `format_code`, `run_tests`, `lint_check` |
| **PRODUCTIVITY** | Workflow tools | `create_task`, `set_reminder`, `calendar` |

---

## 🚀 Creating Your First Tool

### **Step 1: Create the Tool File**

```python
# modules/tools/implementations/productivity/notion_sync.py

from typing import Dict, Any
from modules.tools import tool_registry, ToolCategory

@tool_registry.register(
    category=ToolCategory.PRODUCTIVITY,
    name="notion_sync",
    description="Sync content to a Notion page",
    parameters={
        "page_id": {
            "type": "string",
            "description": "Notion page ID",
            "required": True
        },
        "content": {
            "type": "object",
            "description": "Content to sync",
            "required": True
        },
        "merge_mode": {
            "type": "string",
            "description": "How to merge: 'replace' or 'append'",
            "default": "replace"
        }
    }
)
async def notion_sync(
    page_id: str,
    content: Dict[str, Any],
    merge_mode: str = "replace"
) -> Dict[str, Any]:
    """Sync content to Notion"""
    
    # Your implementation
    notion_client = get_notion_client()
    
    if merge_mode == "replace":
        result = notion_client.pages.update(page_id, content)
    else:
        result = notion_client.blocks.append(page_id, content)
    
    return {
        "success": True,
        "page_id": page_id,
        "url": f"https://notion.so/{page_id}",
        "synced_at": datetime.now().isoformat()
    }
```

### **Step 2: That's It!**

Your tool is now available:
- ✅ In all agents
- ✅ In all workflows  
- ✅ In the API (`/api/tools`)
- ✅ In LLM function calling
- ✅ In the UI tool picker

**No registration code. No config files. Just works.**

---

## 🎨 Advanced Features

### **1. Tool Dependencies**

Tools can use other tools:

```python
@tool_registry.register(cat category=ToolCategory.RESEARCH)
async def research_and_summarize(topic: str) -> dict:
    # Use existing tools
    search_results = await executor.execute_tool(
        "web_search", 
        {"query": topic}
    )
    
    summary = await executor.execute_tool(
        "llm_summarize",
        {"text": search_results["content"]}
    )
    
    return {"topic": topic, "summary": summary}
```

### **2. Credential Integration**

Tools automatically get credentials:

```python
@tool_registry.register(
    category=ToolCategory.COMMUNICATION,
    required_credentials=["slack_token"]  # Auto-injected!
)
async def slack_message(channel: str, message: str) -> dict:
    # Credentials resolved automatically
    slack_token = get_credential("slack_token")
    # Send message...
```

### **3. Result Formatting**

Standardized output for LLMs and UIs:

```python
return {
    "success": True,
    "results": [...],      # Main data
    "metadata": {...},     # Context about execution
    "ui_display": {...}    # How to show in frontend
}
```

---

## 🔥 Why This Is Revolutionary

### **Before Tool Registry**

❌ Tools scattered across files  
❌ Duplicate implementations  
❌ Inconsistent interfaces  
❌ Hard to discover  
❌ Manual registration  
❌ No validation  

### **With Tool Registry**

✅ **Single source of truth**  
✅ **Zero duplication**  
✅ **Automatic registration**  
✅ **Type-safe parameters**  
✅ **Built-in validation**  
✅ **Self-documenting**  
✅ **Instant availability**  

---

## 🤝 Contributing Tools

### **High-Impact Tools We Need**

| Tool | Category | Difficulty | Impact |
|------|----------|-----------|---------|
| `jira_integration` | PRODUCTIVITY | 🟢 Easy | 🔥 High |
| `figma_export` | DESIGN | 🟡 Medium | ⭐ Medium |
| `stripe_payments` | FINANCE | 🟡 Medium | 🔥 High |
| `aws_deploy` | DEPLOYMENT | 🔴 Hard | 🔥 High |
| `linear_tasks` | PRODUCTIVITY | 🟢 Easy | ⭐ Medium |
| `github_actions` | CI/CD | 🟡 Medium | 🔥 High |

### **Tool Contribution Checklist**

- [ ] Implements `BaseTool` or uses `@tool_registry.register`
- [ ] Clear parameter descriptions
- [ ] Type hints everywhere
- [ ] Async implementation (if I/O)
- [ ] Error handling with helpful messages
- [ ] Return standardized result format
- [ ] Add example in docstring
- [ ] Test file in `tests/tools/`

---

## 📚 Examples

### **Simple Tool**
```python
@tool_registry.register(category=ToolCategory.RESEARCH)
async def coin_flip() -> dict:
    """Flip a coin"""
    return {
        "success": True,
        "result": random.choice(["heads", "tails"])
    }
```

### **Complex Tool**
```python
@tool_registry.register(
    category=ToolCategory.DATABASE_TOOLS,
    required_credentials=["db_connection"]
)
async def complex_query(
    query: str,
    params: dict,
    timeout: int = 30
) -> dict:
    """Execute complex database query with retries"""
    
    conn = get_db_connection()
    
    try:
        async with timeout_context(timeout):
            result = await conn.execute(query, params)
            
        return {
            "success": True,
            "rows": result.fetchall(),
            "row_count": len(result),
            "execution_time": result.execution_time
        }
    except TimeoutError:
        return {"success": False, "error": "Query timeout"}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"success": False, "error": str(e)}
```

---

## 🎯 Tool Execution Flow

```
1. Tool Request
   ↓
2. Registry Lookup (by name)
   ↓
3. Parameter Validation (Pydantic)
   ↓
4. Credential Resolution (if needed)
   ↓
5. Tool Execution (async)
   ↓
6. Result Formatting
   ↓
7. Return Standardized Output
```

---

## 🔍 Discovering Tools

### **Programmatically**
```python
from modules.tools import ToolRegistry

registry = ToolRegistry()

# Get all tools
all_tools = registry.get_all_tools()

# Get by category
research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)

# Search by name
search_tool = registry.get_tool("web_search")
```

### **Via API**
```bash
# List all tools
GET /api/tools

# Get tool details
GET /api/tools/{tool_id}

# Execute tool
POST /api/tools/execute
{
  "tool": "web_search",
  "args": {"query": "AI news"}
}
```

---

## ⚡ Performance

- **Lazy loading:** Tools loaded on first use
- **Caching:** Registry cached after initialization
- **Async:** All tools support async execution
- **Parallel:** Execute multiple tools concurrently

---

## 🌟 Pro Tips

1. **Make tools composable** - Small, focused tools > monolithic ones
2. **Return rich metadata** - Help LLMs understand results
3. **Handle errors gracefully** - Return `success: false` with helpful messages
4. **Use type hints** - Enables automatic validation
5. **Add examples** - In docstrings, help users understand usage

---

## 🚀 Get Started

```python
# 1. Import the registry
from modules.tools import tool_registry, ToolCategory

# 2. Register your tool
@tool_registry.register(category=ToolCategory.YOUR_CATEGORY)
async def your_amazing_tool(param: str) -> dict:
    return {"success": True, "result": "🎉"}

# 3. That's it! Tool is now available everywhere!
```

---

**Ready to build the tool that changes everything?** 🛠️

Start in `modules/tools/implementations/` and make your mark!
