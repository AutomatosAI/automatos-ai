# Context-Aware Tool Suggestions (PRD-41 Phase 2)

**Version:** 1.0
**Status:** ✅ Implemented
**Date:** 2026-01-29

---

## Overview

Phase 2 builds on PRD-40's dynamic tool suggestions by adding **context awareness**. When users interact with tools (Gmail, Slack, GitHub), the system remembers what those tools showed and generates personalized suggestions based on recent activity.

**Example:**
```
User: "Check my email"
Agent: [Runs Gmail tool, finds urgent email from Sarah]

[Suggestions dynamically update to:]
✨ "Reply to Sarah's email" (context-aware)
✨ "Reply to the urgent email" (context-aware)
📋 "Summarize unread emails" (curated)
📋 "Draft replies to urgent messages" (curated)
```

---

## Architecture

### Data Flow

```
Tool Execution → Entity Extraction → Mem0 Storage
                                          ↓
User Clicks Tool Icon → Context Retrieval → Suggestion Generation
```

### Components

1. **Entity Extractors** (`orchestrator/core/composio/entity_extractors.py`)
   - Extract meaningful data from tool results
   - Gmail: senders, subjects, labels, email IDs
   - Slack: channels, mentions, message IDs
   - GitHub: PR numbers, issue numbers, repos, authors

2. **Tool Executor Integration** (`orchestrator/core/composio/tool_executor.py`)
   - Automatically extracts entities after successful execution
   - Stores in Mem0 with workspace_id as user_id
   - Graceful error handling (logs but doesn't fail execution)

3. **Context Manager** (`orchestrator/modules/memory/context_manager.py`)
   - Retrieves recent tool results from Mem0
   - Filters by tool name, max age (10 min), and limit
   - Returns structured context with entities

4. **Suggestion Generator** (`orchestrator/api/tools.py`)
   - Generates natural language suggestions from entities
   - Tool-specific templates (Gmail, Slack, GitHub)
   - Truncates long names/subjects for readability

5. **API Endpoint** (`GET /api/tools/{app_name}/suggestions`)
   - Accepts optional `user_id` and `session_id` query params
   - Merges 2 context suggestions + 2 curated suggestions
   - Returns `has_context` flag

6. **Frontend Integration** (`frontend/components/chatbot/chat.tsx`)
   - Passes user_id (Clerk) and session_id (chat ID) to API
   - Displays context hint: "Based on recent {tool} activity"
   - Tracks analytics for context suggestions

---

## Entity Extraction Formats

### Gmail

**Input (Gmail API Response):**
```json
{
  "messages": [{
    "id": "msg_123",
    "labelIds": ["INBOX", "IMPORTANT"],
    "payload": {
      "headers": [
        {"name": "From", "value": "Sarah Johnson <sarah@example.com>"},
        {"name": "Subject", "value": "Urgent: Deadline Update"}
      ]
    }
  }]
}
```

**Output (Extracted Entities):**
```json
{
  "senders": ["Sarah Johnson"],
  "subjects": ["Urgent: Deadline Update"],
  "labels": ["INBOX", "IMPORTANT"],
  "email_ids": ["msg_123"]
}
```

### Slack

**Input (Slack API Response):**
```json
{
  "messages": [{
    "ts": "1234567890.123456",
    "text": "Hey <@U123|john> can you check #general?",
    "channel_name": "engineering"
  }]
}
```

**Output (Extracted Entities):**
```json
{
  "channels": ["#engineering", "#general"],
  "mentions": ["@john"],
  "message_ids": ["1234567890.123456"]
}
```

### GitHub

**Input (GitHub API Response):**
```json
{
  "pull_requests": [{
    "number": 123,
    "user": {"login": "johndoe"},
    "base": {"repo": {"full_name": "org/myrepo"}}
  }]
}
```

**Output (Extracted Entities):**
```json
{
  "pr_numbers": [123],
  "issue_numbers": [],
  "repos": ["org/myrepo"],
  "authors": ["johndoe"]
}
```

---

## Mem0 Storage Schema

**Memory Entry Format:**
```json
{
  "type": "tool_result",
  "tool": "GMAIL",
  "action": "GMAIL_GET_EMAILS",
  "entities": {
    "senders": ["Sarah Johnson"],
    "subjects": ["Urgent: Deadline Update"],
    "labels": ["IMPORTANT"],
    "email_ids": ["msg_123"]
  },
  "timestamp": "2026-01-29T10:30:00Z",
  "user_id": "workspace_abc123",
  "session_id": "chat_xyz789",
  "agent_id": 42
}
```

**Natural Language Summary (for Mem0 processing):**
```
GMAIL GMAIL_GET_EMAILS: emails from Sarah Johnson; subjects: Urgent: Deadline Update
```

---

## Context → Suggestion Generation

### Gmail Examples

| Entities | Generated Suggestion |
|----------|---------------------|
| `senders: ["Sarah Johnson"]` | "Reply to Sarah Johnson's email" |
| `labels: ["IMPORTANT"]` | "Reply to the urgent email" |
| `subjects: ["Project deadline"]` | "Show email about 'Project deadline'" |

### Slack Examples

| Entities | Generated Suggestion |
|----------|---------------------|
| `channels: ["#general"]` | "Send message to #general" |
| `mentions: ["@john"]` | "Reply to @john" |

### GitHub Examples

| Entities | Generated Suggestion |
|----------|---------------------|
| `pr_numbers: [123]` | "Review PR #123" |
| `issue_numbers: [456]` | "Comment on issue #456" |
| `repos: ["org/myrepo"]` | "Show activity in org/myrepo" |

---

## Context Expiry & Limits

- **Max Age:** 10 minutes (configurable via `max_age_minutes` parameter)
- **Limit:** Up to 5 most recent contexts retrieved, use most recent only
- **Fallback:** If no context or expired, return generic Phase 1 suggestions

**Rationale:** 10 minutes balances relevance with memory efficiency. Recent tool results are most likely to be relevant to user's current task.

---

## Error Handling

### Graceful Degradation

1. **Entity Extraction Fails**
   - Logged as warning
   - Tool execution continues normally
   - No context stored, but execution succeeds

2. **Mem0 Storage Fails**
   - Logged as error
   - Tool execution continues
   - Context not available for future suggestions

3. **Mem0 Retrieval Fails**
   - Returns empty context list
   - Falls back to Phase 1 generic suggestions
   - User sees no difference in UX

4. **No Extractor for Tool**
   - Skips entity extraction
   - Only Gmail, Slack, GitHub have extractors currently
   - Other tools still work with generic suggestions

**Principle:** Context-aware suggestions are an enhancement, not a requirement. System degrades gracefully to Phase 1 behavior on any failure.

---

## Analytics Events

### tool_icon_clicked
```typescript
{
  app: "GMAIL",
  location: "chat" | "welcome"
}
```

### suggestions_loaded
```typescript
{
  app: "GMAIL",
  source: "curated" | "generated",
  count: 4,
  has_context: true | false
}
```

### suggestion_clicked
```typescript
{
  app: "GMAIL",
  suggestion: "Reply to Sarah's email",
  source: "curated" | "generated",
  has_context: true | false,
  location: "chat" | "welcome"
}
```

---

## Adding Support for New Tools

To add context-aware suggestions for a new tool (e.g., Google Calendar):

### 1. Create Entity Extractor

```python
# In orchestrator/core/composio/entity_extractors.py

class CalendarExtractor(EntityExtractor):
    def extract(self, tool_result: Dict[str, Any]) -> Dict[str, List[str]]:
        entities = {
            "events": [],
            "attendees": [],
            "event_ids": []
        }

        events = tool_result.get("items", [])
        for event in events:
            entities["events"].append(event.get("summary"))
            entities["event_ids"].append(event.get("id"))
            # Extract attendees...

        return entities

# Add to get_extractor factory
extractors = {
    "GMAIL": GmailExtractor(),
    "SLACK": SlackExtractor(),
    "GITHUB": GitHubExtractor(),
    "GOOGLECALENDAR": CalendarExtractor(),  # NEW
}
```

### 2. Add Suggestion Templates

```python
# In orchestrator/api/tools.py, in generate_context_suggestions()

elif tool_name == "GOOGLECALENDAR":
    events = entities.get("events", [])
    if events:
        event = events[0][:30]  # Truncate
        suggestions.append(f"Show details for '{event}'")

    attendees = entities.get("attendees", [])
    if attendees and len(suggestions) < limit:
        suggestions.append(f"Email {attendees[0]}")
```

### 3. Write Tests

```python
# In orchestrator/tests/core/test_entity_extractors.py

def test_calendar_extractor():
    sample_response = {
        "items": [{
            "id": "event_123",
            "summary": "Team Meeting",
            "attendees": [{"email": "john@example.com"}]
        }]
    }

    extractor = CalendarExtractor()
    entities = extractor.extract(sample_response)

    assert "Team Meeting" in entities["events"]
    assert "event_123" in entities["event_ids"]
```

---

## Performance Considerations

### Mem0 Query Latency
- **Target:** <100ms p95
- **Mitigation:** Mem0 returns up to 50 results, we filter client-side
- **Monitoring:** Track `context_retrieval` latency in analytics

### Entity Extraction
- **Impact:** Negligible (~1-5ms)
- **Runs asynchronously** after tool execution completes
- **Doesn't block** tool response to user

### Suggestion Generation
- **Impact:** <10ms (simple template logic)
- **No external calls** - all in-memory string formatting

---

## Future Enhancements

### Phase 3: Cross-Tool Context
- "Check email and Slack for messages from Sarah"
- Combine entities from multiple tools
- Multi-step workflow suggestions

### Phase 4: User Preference Learning
- Learn which suggestions user clicks most
- Personalize suggestion order
- Adaptive templates based on behavior

### Phase 5: LLM-Generated Suggestions
- Use LLM to generate creative suggestions from context
- More natural language
- Better entity understanding (e.g., "the urgent one" → specific email)

---

## Troubleshooting

### Suggestions not context-aware

**Check:**
1. Is user logged in? (userId must be available from Clerk)
2. Has tool been executed recently? (within 10 minutes)
3. Check Mem0 storage: Does entity have `type: "tool_result"`?
4. Check browser console for API errors

**Debug:**
```python
# In orchestrator/modules/memory/context_manager.py
# Add logging:
logger.info(f"Retrieved {len(results)} contexts for {tool_name}")
logger.debug(f"Context entities: {results}")
```

### Entities not being extracted

**Check:**
1. Does tool have an extractor? (Gmail, Slack, GitHub only currently)
2. Check tool executor logs for extraction errors
3. Verify tool result format matches expected schema

**Debug:**
```python
# In orchestrator/core/composio/tool_executor.py
# The _extract_and_store_entities method logs:
# - "No entity extractor available for {app_name}"
# - "No entities extracted from {app_name} result"
# - "Stored tool context in Mem0: {summary}"
```

### Mem0 connection issues

**Check:**
1. Is Mem0 service running? (`MEM0_API_URL` environment variable)
2. Check Mem0 logs for errors
3. Verify authentication (if required)

**Fallback:** System automatically falls back to Phase 1 suggestions

---

## Code Locations

### Backend

- **Entity Extractors:** `orchestrator/core/composio/entity_extractors.py`
- **Tool Executor Integration:** `orchestrator/core/composio/tool_executor.py` (lines 451-475, 610-700)
- **Context Manager:** `orchestrator/modules/memory/context_manager.py`
- **Suggestion Generator:** `orchestrator/api/tools.py` (lines 490-650)
- **Mem0 Client:** `orchestrator/modules/memory/integrations/mem0_client.py`

### Frontend

- **Chat Component:** `frontend/components/chatbot/chat.tsx` (lines 70-77, 351-394)
- **Suggestion Bar:** `frontend/components/suggestions/ToolSuggestionBar.tsx`
- **Types:** `frontend/components/suggestions/types.ts`

### Tests

- **Entity Extractors:** `orchestrator/tests/core/test_entity_extractors.py`
- **Context Manager:** `orchestrator/tests/modules/test_context_manager.py`
- **Suggestion Generator:** `orchestrator/tests/api/test_context_suggestions.py`

---

## Related Documentation

- **PRD-40:** Dynamic Tool Suggestions (Phase 1)
- **PRD-39:** Mem0 Integration
- **PRD-38.1:** Widget Architecture (uses similar context patterns)

---

**End of Documentation**
