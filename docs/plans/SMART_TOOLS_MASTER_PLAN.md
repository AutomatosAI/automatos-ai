# Smart Tools & Skills Master Plan (v2)

## Vision
Make Automatos tools/skills dynamic, context-aware, and self-documenting.
No hardcoding. Everything schema-driven. Leverage existing infrastructure.

---

## What We Already Have (Don't Reinvent)

### Agents System (`/agents`)
- Agent creation wizard: Type → Config → Model → Tools → Skills
- Agent types exist but don't do much currently
- Each agent has Mem0 memory ID (learns over time)
- Agents assigned to consumers (chatbot, workflows, APIs)

### Workflows System (`/workflows`)
- Workflow templates (currently just templated prompts)
- Can be scheduled
- Multi-step execution

### Memory System (Mem0)
- User memory (personal preferences)
- Agent memory (per-agent learning)
- Already handles "John's email", "default Slack channel", etc.

### Tool Assignment
- Tools assigned per-agent
- LLM only sees assigned tools (not all 800+)
- Already optimized for context size

---

## Phase 1: Dynamic Tool Suggestions ⭐ PRIORITY

### Goal
When user clicks a tool icon (Gmail, Slack), show tool-specific suggestions.
Add suggestions to BOTH welcome screen AND chat screen.

### Current State
- **Welcome screen**: Has 4 generic suggestions (workflow, document, CPU, CodeGraph)
- **Chat screen**: No suggestions at all
- **Tool icons**: Visible next to agent selector, but clicking does nothing special

### Target State
```
User clicks Gmail icon →
┌──────────────────────────────────────────────────────────┐
│ ✉️ Gmail Suggestions                                      │
├──────────────────────────────────────────────────────────┤
│ • Summarize unread emails from this morning              │
│ • Draft replies to urgent messages                       │
│ • Find emails with attachments from last week            │
│ • Show emails from [contact name]                        │
└──────────────────────────────────────────────────────────┘
```

### Implementation

#### 1. Database: Add suggestions to tool cache
```sql
ALTER TABLE composio_actions_cache
ADD COLUMN app_suggestions JSONB DEFAULT '[]';

-- Example data
UPDATE composio_actions_cache
SET app_suggestions = '[
  "Summarize unread emails from this morning",
  "Draft replies to urgent messages",
  "Find emails with attachments from last week",
  "Show emails from {{contact}}"
]'
WHERE app_name = 'GMAIL';
```

#### 2. API: Tool suggestions endpoint
```python
# api/tools.py

@router.get("/api/tools/{app_name}/suggestions")
async def get_tool_suggestions(
    app_name: str,
    db: Session = Depends(get_db)
) -> dict:
    """Get suggestions for a specific tool/app"""

    # Get from cache
    cached = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name.upper()
    ).first()

    if cached and cached.app_suggestions:
        return {
            "app": app_name,
            "suggestions": cached.app_suggestions,
            "source": "curated"
        }

    # Fallback: Generate from schema
    suggestions = generate_suggestions_from_schema(app_name, db)
    return {
        "app": app_name,
        "suggestions": suggestions,
        "source": "generated"
    }


def generate_suggestions_from_schema(app_name: str, db: Session) -> list:
    """Generate suggestions from action descriptions"""
    actions = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name.upper()
    ).limit(10).all()

    suggestions = []
    for action in actions:
        desc = action.description or ""
        name = action.display_name or action.action_name

        # Extract verb from description
        if "list" in desc.lower() or "fetch" in desc.lower():
            suggestions.append(f"Show me my {app_name.lower()} items")
        if "send" in desc.lower() or "create" in desc.lower():
            suggestions.append(f"Send/create a new {app_name.lower()} item")
        if "search" in desc.lower():
            suggestions.append(f"Search {app_name.lower()} for...")

    return list(set(suggestions))[:4]
```

#### 3. Frontend: Tool icon click handler

**multimodal-input.tsx changes:**
```typescript
// State for tool suggestions
const [toolSuggestions, setToolSuggestions] = useState<string[]>([])
const [activeToolApp, setActiveToolApp] = useState<string | null>(null)

// Handler when tool icon is clicked
const handleToolIconClick = async (appName: string) => {
  setActiveToolApp(appName)

  // Fetch suggestions for this tool
  const res = await fetch(`/api/tools/${appName}/suggestions`)
  const data = await res.json()
  setToolSuggestions(data.suggestions || [])
}

// Render suggestions (replaces generic ones when tool selected)
{activeToolApp && toolSuggestions.length > 0 ? (
  <ToolSuggestions
    appName={activeToolApp}
    suggestions={toolSuggestions}
    onSelect={(suggestion) => {
      sendMessage(suggestion)
      setActiveToolApp(null)
    }}
    onClose={() => setActiveToolApp(null)}
  />
) : (
  <GenericSuggestions suggestions={defaultSuggestions} />
)}
```

#### 4. Add suggestions to chat screen
Currently chat screen has no suggestions. Add them above input:

```typescript
// In chat message area (when no messages or at bottom)
<SuggestionBar
  agentId={selectedAgentId}
  onSelect={sendMessage}
/>
```

### Curated Suggestions (Initial Set)

```json
{
  "GMAIL": [
    "Summarize unread emails from this morning",
    "Draft replies to urgent messages",
    "Find emails with attachments from last week",
    "Show emails from {{contact}}"
  ],
  "SLACK": [
    "Send a message to #{{channel}}",
    "Summarize today's messages in #general",
    "Find messages mentioning {{keyword}}",
    "Check my unread DMs"
  ],
  "GITHUB": [
    "Show my open pull requests",
    "List issues assigned to me",
    "Check CI status for {{repo}}",
    "Create a new issue in {{repo}}"
  ],
  "GOOGLE_CALENDAR": [
    "What's on my calendar today?",
    "Schedule a meeting with {{person}}",
    "Find my next free slot this week",
    "Show meetings for tomorrow"
  ],
  "NOTION": [
    "Search my Notion for {{topic}}",
    "Create a new page in {{database}}",
    "Show recent updates to my workspace",
    "Find notes about {{subject}}"
  ]
}
```

### Files to Modify

| File | Change |
|------|--------|
| `composio_actions_cache` | Add `app_suggestions` column |
| `api/tools.py` | New `/suggestions` endpoint |
| `frontend/multimodal-input.tsx` | Tool icon click → fetch suggestions |
| `frontend/components/chatbot/chat.tsx` | Add suggestions to chat view |
| `frontend/components/suggestions/` | New suggestion components |

### Migration Script
```python
# migrations/add_tool_suggestions.py

INITIAL_SUGGESTIONS = {
    "GMAIL": [...],
    "SLACK": [...],
    # etc.
}

def upgrade():
    # Add column
    op.add_column('composio_actions_cache',
        sa.Column('app_suggestions', sa.JSON, default=[])
    )

    # Populate initial suggestions
    for app_name, suggestions in INITIAL_SUGGESTIONS.items():
        op.execute(f"""
            UPDATE composio_actions_cache
            SET app_suggestions = '{json.dumps(suggestions)}'
            WHERE app_name = '{app_name}'
        """)
```

---

## Phase 2: Tool Context Memory (Mem0 Integration)

### Goal
"Reply to the urgent one" works without re-fetching.

### Approach
Leverage existing Mem0 system - store tool discoveries in agent memory.

### Implementation
```python
# After tool execution, store key entities
async def store_tool_context(
    agent_id: int,
    tool_name: str,
    result: dict,
    mem0_client: Mem0Client
):
    """Store discovered entities in agent memory for later reference"""

    entities = extract_key_entities(tool_name, result)
    if not entities:
        return

    # Store as agent memory
    context_text = f"Recent {tool_name} results:\n"
    for entity in entities[:5]:
        context_text += f"- {entity['label']}: ID={entity['id']}\n"

    await mem0_client.add_memory(
        user_id=f"agent_{agent_id}_context",
        text=context_text,
        metadata={"type": "tool_context", "tool": tool_name}
    )


def extract_key_entities(tool_name: str, result: dict) -> list:
    """Extract referenceable entities from tool result"""
    entities = []

    if "emails" in result:
        for email in result["emails"][:5]:
            entities.append({
                "type": "email",
                "id": email["id"],
                "label": email.get("subject", "")[:50]
            })

    if "messages" in result:
        for msg in result["messages"][:5]:
            entities.append({
                "type": "message",
                "id": msg.get("ts") or msg.get("id"),
                "label": msg.get("text", "")[:50]
            })

    return entities
```

### LLM Context Enhancement
```python
# In service.py, before LLM call
tool_context = await mem0_client.search_memory(
    user_id=f"agent_{agent_id}_context",
    query="recent tool results",
    limit=3
)

if tool_context:
    context_hint = """
[Recent Tool Context]
{context}

Use these IDs when user refers to "the first one", "that email", etc.
""".format(context=tool_context)

    messages.insert(2, {"role": "system", "content": context_hint})
```

---

## Phase 3: Smart Result Summarization

### Goal
50 emails → summary to LLM → details on-demand (token optimization)

### Implementation
```python
class SmartResultProcessor:
    MAX_DIRECT_ITEMS = 5
    MAX_BODY_CHARS = 500

    @classmethod
    def process_for_llm(cls, tool_name: str, result: dict) -> dict:
        """Summarize large results for LLM efficiency"""

        # Email results
        if "emails" in result:
            emails = result["emails"]
            if len(emails) > cls.MAX_DIRECT_ITEMS:
                return {
                    "summary": {
                        "total": len(emails),
                        "unread": sum(1 for e in emails if not e.get("isRead")),
                        "with_attachments": sum(1 for e in emails if e.get("hasAttachments")),
                        "top_senders": cls._get_top_senders(emails),
                    },
                    "preview": [cls._summarize_email(e) for e in emails[:5]],
                    "hint": "Ask which emails to see in detail, or filter by sender/subject"
                }

        return result

    @classmethod
    def _summarize_email(cls, email: dict) -> dict:
        return {
            "id": email["id"],
            "subject": email.get("subject", "")[:80],
            "from": email.get("from", {}).get("email"),
            "date": email.get("date"),
            "has_attachment": email.get("hasAttachments", False)
        }
```

---

## Phase 4: Recipes (Leverage Existing Infrastructure)

### Current State
- **Agent Types**: Code Architect, Security Expert, etc. (do nothing currently)
- **Workflow Templates**: Just templated prompts

### New Vision
- **Agent Types → Recipe Templates**: Pre-configured agent setups
- **Workflow Templates → Executable Recipes**: Multi-tool automated flows
- **Marketplace**: Share agents/recipes publicly

### Recipe = Agent Type + Workflow Template

```yaml
# Recipe: Daily Work Review
id: daily-work-review
name: "Daily Work Review"
description: "Start your day with email, calendar, and Slack summary"
category: "productivity"
icon: "coffee"
author: "automatos"
public: true

# Agent configuration (pre-filled when creating agent)
agent_template:
  type: "daily_reviewer"
  suggested_name: "Morning Assistant"
  suggested_model: "gpt-4o-mini"
  required_tools:
    - GMAIL_FETCH_EMAILS
    - GOOGLE_CALENDAR_LIST_EVENTS
    - SLACK_LIST_MESSAGES
  suggested_skills:
    - email_summarization
    - meeting_prep

# Workflow template (executable flow)
workflow:
  trigger:
    type: "schedule"
    cron: "0 8 * * 1-5"  # 8am weekdays

  steps:
    - id: "fetch_emails"
      tool: "GMAIL_FETCH_EMAILS"
      params:
        query: "is:unread"
        max_results: 20

    - id: "fetch_calendar"
      tool: "GOOGLE_CALENDAR_LIST_EVENTS"
      params:
        time_min: "{{today_start}}"
        time_max: "{{today_end}}"

    - id: "summarize"
      type: "llm"
      prompt: |
        Create a morning briefing:
        - Emails: {{fetch_emails.summary}}
        - Meetings: {{fetch_calendar.events}}

        Highlight urgent items and suggest actions.

# What user sees in wizard
wizard_hints:
  config: "Name your morning assistant"
  model: "GPT-4o-mini recommended for cost efficiency"
  tools: "Gmail, Calendar, Slack required"
  skills: "Email summarization skill auto-added"
```

### UI Flow
```
/agents → Create Agent → Select Recipe
                              ↓
┌─────────────────────────────────────────────────────┐
│ Choose a Recipe (or start custom)                   │
├─────────────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│ │ ☕      │ │ 💰      │ │ 🔒      │ │ 📊      │    │
│ │ Daily   │ │ Invoice │ │ Security│ │ Data    │    │
│ │ Review  │ │ Process │ │ Expert  │ │ Analyst │    │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
│                                                     │
│ ┌─────────┐                                        │
│ │ ✨      │                                        │
│ │ Custom  │                                        │
│ │ Agent   │                                        │
│ └─────────┘                                        │
└─────────────────────────────────────────────────────┘
                              ↓
        Recipe selected → Pre-fills wizard steps
        Custom selected → Empty wizard (current flow)
```

### Marketplace (Future)
- Users can publish their agents/recipes
- Browse community recipes
- One-click install
- Ratings & reviews

---

## Phase 5: MCP Integration (BACKLOG)

### Notes for Future
- Model Context Protocol for cleaner Composio integration
- Standardized tool format
- Built-in context management
- Would require significant refactor

### Keep on Radar
- Monitor Composio MCP developments
- Evaluate when current approach hits limits
- Plan migration path

---

## Phase 6: Parallel Tool Execution

### Goal
"Check email AND Slack" runs both simultaneously

### Implementation
```python
async def execute_tools_parallel(tool_calls: list) -> list:
    """Execute independent tools in parallel"""

    # Identify dependencies
    independent = [t for t in tool_calls if not t.get("depends_on")]
    dependent = [t for t in tool_calls if t.get("depends_on")]

    # Run independent in parallel
    results = await asyncio.gather(*[
        execute_single_tool(t) for t in independent
    ], return_exceptions=True)

    # Run dependent sequentially
    for tool in dependent:
        dep_result = next(
            r for r in results
            if r.get("tool_id") == tool["depends_on"]
        )
        tool["params"].update(extract_context(dep_result))
        results.append(await execute_single_tool(tool))

    return results
```

### LLM Instruction Update
```
You can request multiple tools in one response when they don't depend
on each other. They'll execute in parallel for speed.

Example - GOOD (parallel):
- Tool 1: GMAIL_FETCH_EMAILS
- Tool 2: SLACK_LIST_MESSAGES

Example - BAD (must be sequential):
- Tool 1: GMAIL_FETCH_EMAILS
- Tool 2: GMAIL_REPLY (needs email ID from Tool 1)
```

---

## Implementation Roadmap

| Phase | Description | Effort | Timeline |
|-------|-------------|--------|----------|
| **1** | Tool Suggestions | Low | Week 1 |
| **2** | Tool Context (Mem0) | Medium | Week 2 |
| **3** | Smart Summarization | Medium | Week 3 |
| **4** | Recipes | Medium | Week 4-5 |
| **5** | MCP Integration | High | Backlog |
| **6** | Parallel Execution | Medium | Week 6 |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Tool suggestion click rate | 30%+ of tool interactions |
| "Which one?" clarifications | 50% reduction |
| Token usage for lists | 60% reduction |
| Recipe adoption | 40% of new agents from recipes |
| Community recipes | 20+ shared recipes in 3 months |

---

## Next Steps

1. ✅ Review and finalize this plan
2. 🔜 Create detailed PRD for Phase 1 (Tool Suggestions)
3. 🔜 Implement Phase 1
4. 🔜 Test and iterate
5. 🔜 Move to Phase 2
