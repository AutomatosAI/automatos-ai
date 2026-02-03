# PRD-40: Dynamic Tool Suggestions

**Version:** 1.0
**Status:** 🟡 Ready for Implementation
**Priority:** HIGH - Quick Win, High Impact
**Author:** Automatos AI Platform Team
**Last Updated:** 2026-01-29
**Dependencies:** PRD-38.1 (Widget Architecture), Composio Integration

---

## Executive Summary

Currently, users see 4 generic suggestions on the welcome screen ("Start a workflow", "Upload document", etc.) but **no suggestions on the chat screen** where actual work happens. Tool icons (Gmail, Slack, GitHub) are visible but not interactive.

This PRD introduces **dynamic, context-aware tool suggestions** that appear when users click tool icons OR when they're actively chatting. Each tool displays curated, action-specific prompts (e.g., clicking Gmail shows "Summarize unread emails from this morning", "Draft replies to urgent messages").

### The Problem

1. **Generic suggestions only on welcome screen** - Chat screen has no suggestions
2. **Tool icons are decorative** - Clicking Gmail/Slack does nothing
3. **Users don't know what's possible** - 863 Composio apps with 15,000+ actions, but no discovery
4. **No contextual help** - Users type "check my email" instead of seeing suggested email actions

### The Solution

**Phase 1**: Tool-specific suggestion chips that dynamically update based on:
- Clicked tool icon (Gmail → Gmail suggestions)
- Assigned tools for current agent
- User's recent tool usage (future: Mem0 integration)

**Vision**: "If I click the Gmail icon, show me what I can do with Gmail"

---

## Goals

### Primary Goals
1. ✅ **Increase tool discoverability** - Users learn what actions are available
2. ✅ **Reduce friction** - One-click prompts vs typing queries
3. ✅ **Add suggestions to chat screen** - Where users actually work
4. ✅ **Make tool icons interactive** - Click Gmail → see Gmail prompts

### Secondary Goals
1. Track suggestion click-through rates for analytics
2. Learn user preferences via Mem0 (Phase 2)
3. Generate suggestions from schemas (fallback for uncurated apps)

### Non-Goals (Future Phases)
- AI-generated suggestions based on conversation context
- Cross-tool suggestion combinations ("Check email AND Slack")
- Per-user personalized suggestions

---

## User Stories

### Story 1: Tool Icon Click
**As a** user
**I want to** click a tool icon (Gmail, Slack, etc.)
**So that** I can see relevant suggested actions for that tool

**Acceptance Criteria:**
- [ ] Clicking Gmail icon shows 4-6 Gmail-specific suggestion chips
- [ ] Clicking Slack icon shows Slack-specific suggestions
- [ ] Clicking same icon again closes the suggestions
- [ ] Clicking a different icon switches to that tool's suggestions
- [ ] Suggestions appear in both welcome screen AND chat screen

### Story 2: Chat Screen Suggestions
**As a** user chatting with an agent
**I want to** see suggested prompts above the input
**So that** I don't have to think of what to ask

**Acceptance Criteria:**
- [ ] Suggestion bar appears above chat input (not just on welcome screen)
- [ ] Shows 4 suggestions by default (generic or tool-specific)
- [ ] Updates when tool icon is clicked
- [ ] Clicking a suggestion sends that message to the agent
- [ ] Mobile-responsive (horizontal scroll on small screens)

### Story 3: Curated Suggestions
**As a** platform admin
**I want to** curate high-quality suggestions for popular tools
**So that** users get helpful, proven prompts

**Acceptance Criteria:**
- [ ] Database stores app_suggestions per Composio app
- [ ] Initial set: Gmail, Slack, GitHub, Google Calendar, Notion
- [ ] Suggestions use {{placeholders}} for user customization
- [ ] Admin can update suggestions via database (UI in future)

### Story 4: Schema-Generated Fallback
**As the** system
**I want to** generate suggestions from action schemas
**So that** uncurated tools still have basic suggestions

**Acceptance Criteria:**
- [ ] If app has no curated suggestions, generate from top actions
- [ ] Use action descriptions to create natural language prompts
- [ ] Return at least 3 suggestions for any tool
- [ ] Mark source as "generated" vs "curated" in API response

---

## Current State → Proposed State

| Aspect | Current | Phase 1 (This PRD) |
|--------|---------|-------------------|
| **Welcome Screen** | 4 static suggestions | 4 generic + tool-specific on click |
| **Chat Screen** | No suggestions | Suggestion bar above input |
| **Tool Icons** | Decorative only | Clickable → show suggestions |
| **Suggestion Source** | Hardcoded in frontend | Database-driven (app_suggestions column) |
| **Discovery** | Users guess | Curated prompts guide users |
| **Apps Covered** | N/A | Gmail, Slack, GitHub, Calendar, Notion (Phase 1) |

---

## Architecture

### Database Schema

**Add column to existing `composio_actions_cache` table:**

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

**Why this table?**
- Already stores Composio app metadata per app (one row per app)
- Response schemas and parameters already cached here
- Logical place for app-level suggestions (not action-level)

### API Endpoints

#### New: GET `/api/tools/{app_name}/suggestions`

**Purpose:** Fetch suggestions for a specific tool/app

**Request:**
```
GET /api/tools/GMAIL/suggestions
```

**Response:**
```json
{
  "app": "GMAIL",
  "suggestions": [
    "Summarize unread emails from this morning",
    "Draft replies to urgent messages",
    "Find emails with attachments from last week",
    "Show emails from {{contact}}"
  ],
  "source": "curated"  // or "generated"
}
```

**Implementation:** `orchestrator/api/tools.py`

```python
@router.get("/api/tools/{app_name}/suggestions")
async def get_tool_suggestions(
    app_name: str,
    db: Session = Depends(get_db)
) -> dict:
    """Get suggestions for a specific tool/app"""

    # Normalize app name (user might send "Gmail" or "gmail")
    app_name_upper = app_name.upper()

    # Try to get from cache first (curated suggestions)
    cached = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name_upper
    ).first()

    if cached and cached.app_suggestions:
        return {
            "app": app_name_upper,
            "suggestions": cached.app_suggestions,
            "source": "curated"
        }

    # Fallback: Generate from schema
    suggestions = generate_suggestions_from_schema(app_name_upper, db)
    return {
        "app": app_name_upper,
        "suggestions": suggestions,
        "source": "generated"
    }


def generate_suggestions_from_schema(app_name: str, db: Session) -> list:
    """Generate suggestions from top action descriptions"""
    actions = db.query(ComposioActionCache).filter(
        ComposioActionCache.app_name == app_name.upper()
    ).limit(10).all()

    suggestions = set()
    for action in actions:
        desc = action.description or ""
        name = action.display_name or action.action_name

        # Pattern matching to generate prompts
        if any(verb in desc.lower() for verb in ["list", "fetch", "get"]):
            suggestions.add(f"Show my {app_name.lower()} items")
        if any(verb in desc.lower() for verb in ["send", "create", "post"]):
            suggestions.add(f"Create a new {app_name.lower()} item")
        if "search" in desc.lower():
            suggestions.add(f"Search {app_name.lower()} for...")
        if any(verb in desc.lower() for verb in ["update", "edit", "modify"]):
            suggestions.add(f"Update a {app_name.lower()} item")

    return list(suggestions)[:4]
```

### Frontend Architecture

#### Components Structure

```
frontend/components/suggestions/
├── ToolSuggestionBar.tsx       # Main suggestion bar component
├── SuggestionChip.tsx          # Individual suggestion chip
└── types.ts                    # Suggestion type definitions
```

#### State Management

```typescript
// In chat.tsx or multimodal-input.tsx
const [activeTool, setActiveTool] = useState<string | null>(null)
const [toolSuggestions, setToolSuggestions] = useState<string[]>([])
const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false)

// Handler for tool icon click
const handleToolIconClick = async (appName: string) => {
  if (activeTool === appName) {
    // Toggle off
    setActiveTool(null)
    setToolSuggestions([])
    return
  }

  setActiveTool(appName)
  setIsLoadingSuggestions(true)

  try {
    const res = await fetch(`/api/tools/${appName}/suggestions`)
    const data = await res.json()
    setToolSuggestions(data.suggestions || [])
  } catch (error) {
    console.error('Failed to fetch suggestions:', error)
    setToolSuggestions([])
  } finally {
    setIsLoadingSuggestions(false)
  }
}

// Handler for suggestion click
const handleSuggestionClick = (suggestion: string) => {
  // Replace {{placeholders}} with user input (future enhancement)
  sendMessage(suggestion)
  setActiveTool(null) // Close suggestions after use
}
```

#### UI Components

**ToolSuggestionBar.tsx:**
```tsx
interface ToolSuggestionBarProps {
  suggestions: string[]
  activeTool: string | null
  onSuggestionClick: (suggestion: string) => void
  onClose: () => void
}

export function ToolSuggestionBar({
  suggestions,
  activeTool,
  onSuggestionClick,
  onClose
}: ToolSuggestionBarProps) {
  if (!activeTool || suggestions.length === 0) return null

  return (
    <div className="flex items-center gap-2 p-2 bg-muted/50 rounded-lg border border-border">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <AppIcon name={activeTool} />
        <span className="font-medium">{activeTool} suggestions:</span>
      </div>

      <div className="flex gap-2 flex-wrap flex-1">
        {suggestions.map((suggestion, i) => (
          <SuggestionChip
            key={i}
            text={suggestion}
            onClick={() => onSuggestionClick(suggestion)}
          />
        ))}
      </div>

      <Button variant="ghost" size="icon" onClick={onClose}>
        <X className="h-4 w-4" />
      </Button>
    </div>
  )
}
```

**Integration in chat.tsx:**
```tsx
{/* Add above chat input, below messages */}
<ToolSuggestionBar
  suggestions={toolSuggestions}
  activeTool={activeTool}
  onSuggestionClick={handleSuggestionClick}
  onClose={() => setActiveTool(null)}
/>

{/* Existing multimodal input */}
<MultimodalInput
  onToolIconClick={handleToolIconClick}
  // ... other props
/>
```

---

## Implementation Plan

### Phase 1.1: Database & API (Days 1-2)
**Goal:** Set up data layer

**Tasks:**
- [ ] Create migration script: `migrations/add_app_suggestions.py`
- [ ] Add `app_suggestions JSONB` column to `composio_actions_cache`
- [ ] Populate initial suggestions for 5 apps (Gmail, Slack, GitHub, Calendar, Notion)
- [ ] Implement `GET /api/tools/{app_name}/suggestions` endpoint
- [ ] Implement schema-based suggestion generator (fallback)
- [ ] Add unit tests for suggestion generation logic

**Files Modified:**
- `orchestrator/migrations/add_app_suggestions.py` (new)
- `orchestrator/api/tools.py` (new endpoint)
- `tests/api/test_tools_suggestions.py` (new)

**Acceptance:**
- Migration runs successfully
- API returns curated suggestions for Gmail
- API returns generated suggestions for uncurated apps
- Tests pass

---

### Phase 1.2: Frontend Components (Days 3-4)
**Goal:** Build reusable suggestion UI

**Tasks:**
- [ ] Create `ToolSuggestionBar` component
- [ ] Create `SuggestionChip` component
- [ ] Add suggestion state management to chat.tsx
- [ ] Wire up tool icon click handlers in multimodal-input.tsx
- [ ] Add loading states and error handling
- [ ] Mobile-responsive styling (horizontal scroll)

**Files Modified:**
- `frontend/components/suggestions/ToolSuggestionBar.tsx` (new)
- `frontend/components/suggestions/SuggestionChip.tsx` (new)
- `frontend/components/suggestions/types.ts` (new)
- `frontend/components/chatbot/chat.tsx` (modify)
- `frontend/components/chatbot/multimodal-input.tsx` (modify)

**Acceptance:**
- Suggestion bar renders correctly
- Tool icons are clickable
- Clicking tool icon fetches and displays suggestions
- Clicking suggestion sends message to agent
- Responsive on mobile

---

### Phase 1.3: Chat Screen Integration (Day 5)
**Goal:** Add suggestions to chat screen (not just welcome)

**Tasks:**
- [ ] Add suggestion bar above chat input
- [ ] Show default suggestions (agent's assigned tools) when no tool selected
- [ ] Persist suggestion visibility state
- [ ] Handle suggestion visibility on mobile (collapsible?)
- [ ] Add keyboard shortcuts (optional: Cmd+K to open suggestions)

**Files Modified:**
- `frontend/components/chatbot/chat.tsx` (add suggestion bar)

**Acceptance:**
- Suggestions visible in chat screen above input
- Default suggestions show agent's top tools
- Clicking tool icon updates suggestions
- Works on mobile

---

### Phase 1.4: Polish & Analytics (Day 6)
**Goal:** Production-ready UX and tracking

**Tasks:**
- [ ] Add analytics tracking for suggestion clicks
- [ ] Add analytics for tool icon clicks
- [ ] Track suggestion source (curated vs generated)
- [ ] Add tooltips for tool icons ("Click for Gmail suggestions")
- [ ] Add animations (smooth transitions)
- [ ] Accessibility: keyboard navigation, ARIA labels

**Files Modified:**
- `frontend/lib/analytics.ts` (add suggestion events)
- `frontend/components/suggestions/*.tsx` (add tracking)

**Acceptance:**
- Analytics events fire correctly
- Smooth UX with animations
- Accessible (keyboard + screen reader)

---

## Curated Suggestions (Initial Set)

### Gmail
```json
[
  "Summarize unread emails from this morning",
  "Draft replies to urgent messages",
  "Find emails with attachments from last week",
  "Show emails from {{contact}}",
  "Mark all unread emails as read",
  "Find emails about {{topic}}"
]
```

### Slack
```json
[
  "Send a message to #{{channel}}",
  "Summarize today's messages in #general",
  "Find messages mentioning {{keyword}}",
  "Check my unread DMs",
  "List all channels I'm in",
  "Search for {{query}} in Slack"
]
```

### GitHub
```json
[
  "Show my open pull requests",
  "List issues assigned to me",
  "Check CI status for {{repo}}",
  "Create a new issue in {{repo}}",
  "Show recent commits in {{repo}}",
  "Search code for {{query}}"
]
```

### Google Calendar
```json
[
  "What's on my calendar today?",
  "Schedule a meeting with {{person}}",
  "Find my next free slot this week",
  "Show meetings for tomorrow",
  "List all events this week",
  "Cancel meeting {{title}}"
]
```

### Notion
```json
[
  "Search my Notion for {{topic}}",
  "Create a new page in {{database}}",
  "Show recent updates to my workspace",
  "Find notes about {{subject}}",
  "List all pages in {{workspace}}",
  "Update page {{title}}"
]
```

---

## Migration Script

**File:** `orchestrator/migrations/add_app_suggestions.py`

```python
"""
Migration: Add app_suggestions column to composio_actions_cache
Date: 2026-01-29
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
import json

# Curated suggestions
INITIAL_SUGGESTIONS = {
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
    "GOOGLECALENDAR": [
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


def upgrade():
    # Add column
    op.add_column('composio_actions_cache',
        sa.Column('app_suggestions', JSONB, server_default='[]')
    )

    # Populate initial suggestions
    conn = op.get_bind()
    for app_name, suggestions in INITIAL_SUGGESTIONS.items():
        conn.execute(
            sa.text("""
                UPDATE composio_actions_cache
                SET app_suggestions = :suggestions
                WHERE app_name = :app_name
            """),
            {"suggestions": json.dumps(suggestions), "app_name": app_name}
        )


def downgrade():
    op.drop_column('composio_actions_cache', 'app_suggestions')
```

---

## Testing Strategy

### Unit Tests

**Test suggestion generation from schemas:**
```python
# tests/api/test_tools_suggestions.py

def test_get_suggestions_curated(db_session):
    """Test fetching curated suggestions"""
    response = client.get("/api/tools/GMAIL/suggestions")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "GMAIL"
    assert data["source"] == "curated"
    assert len(data["suggestions"]) > 0

def test_get_suggestions_generated(db_session):
    """Test fallback generation for uncurated apps"""
    response = client.get("/api/tools/UNKNOWN_APP/suggestions")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "generated"
    assert len(data["suggestions"]) >= 3
```

### Integration Tests

**Test full flow:**
```typescript
// tests/e2e/tool-suggestions.spec.ts

test('clicking tool icon shows suggestions', async ({ page }) => {
  await page.goto('/chat')

  // Click Gmail icon
  await page.click('[data-tool-icon="GMAIL"]')

  // Suggestions should appear
  await expect(page.locator('[data-testid="suggestion-bar"]')).toBeVisible()
  await expect(page.locator('[data-testid="suggestion-chip"]')).toHaveCount(4)

  // Click first suggestion
  await page.click('[data-testid="suggestion-chip"]:first-child')

  // Message should be sent
  await expect(page.locator('.chat-message').last()).toContainText('Summarize unread')
})
```

### Manual Test Scenarios

1. **Welcome Screen Suggestions**
   - [ ] Default 4 suggestions visible
   - [ ] Clicking tool icon replaces with tool-specific suggestions
   - [ ] Clicking suggestion sends message

2. **Chat Screen Suggestions**
   - [ ] Suggestion bar appears above input
   - [ ] Tool icon click updates suggestions
   - [ ] Clicking same icon closes suggestions

3. **Mobile Responsive**
   - [ ] Suggestions scroll horizontally on small screens
   - [ ] Tool icons remain accessible
   - [ ] Touch interactions work

4. **Edge Cases**
   - [ ] App with no curated suggestions → generates from schema
   - [ ] App with no actions → shows generic message
   - [ ] Network error → shows error toast

---

## Success Metrics

### Phase 1 Launch Metrics (Week 1)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Suggestion Click Rate** | 30% of users | Track suggestion clicks / total chat sessions |
| **Tool Icon Interaction** | 50% of users | Track tool icon clicks |
| **Curated vs Generated** | 80% curated | % of suggestion views from curated apps |
| **Mobile Usage** | Works on all devices | QA testing + analytics |

### Phase 2 Metrics (Month 1)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Popular Suggestions** | Top 10 per app | Track click frequency |
| **User Feedback** | 4+ stars | In-app feedback on suggestions |
| **Time to First Tool Use** | 30% faster | Compare before/after |
| **Tool Discovery** | 5+ apps/user | Track unique tools clicked |

### Analytics Events

```typescript
// Track suggestion interactions
analytics.track('suggestion_clicked', {
  app: 'GMAIL',
  suggestion: 'Summarize unread emails',
  source: 'curated', // or 'generated'
  location: 'chat' // or 'welcome'
})

analytics.track('tool_icon_clicked', {
  app: 'GMAIL',
  location: 'chat'
})

analytics.track('suggestions_loaded', {
  app: 'GMAIL',
  source: 'curated',
  count: 4
})
```

---

## Dependencies

### Backend
- Existing: `composio_actions_cache` table
- Existing: Composio integration
- New: Migration system (Alembic)

### Frontend
- Existing: Multimodal input component
- Existing: Tool icons rendering
- New: Suggestion components

### External
- None (all internal)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Low-quality generated suggestions** | High | Curate top 20 apps manually, iterate on generation logic |
| **Suggestion fatigue** | Medium | Limit to 4-6 suggestions, allow hiding |
| **Mobile UX cramped** | Medium | Horizontal scroll, collapsible bar |
| **Database migration fails** | High | Test migration on staging, rollback plan |
| **Performance (suggestion fetching)** | Low | Cache suggestions in frontend, 200ms timeout |

---

## Future Enhancements (Phase 2+)

### Phase 2: Context-Aware Suggestions
- Use conversation history to suggest relevant actions
- "You mentioned urgent emails" → show email filtering suggestions
- Mem0 integration: learn user's common tasks

### Phase 3: Suggestion Marketplace
- Users can create/share custom suggestion packs
- Community voting on best suggestions
- Revenue sharing for creators

### Phase 4: Multi-Tool Suggestions
- "Check email AND Slack for messages from John"
- Combine multiple tools in one suggestion
- Workflow templates as suggestions

---

## Open Questions

1. **Placeholder Handling**: How should {{contact}} placeholders work?
   - Option A: Replace with input field inline
   - Option B: Open modal to fill placeholders
   - Option C: Send as-is, LLM extracts from context
   - **Decision**: Option C for Phase 1 (simplest), Option A for Phase 2

2. **Suggestion Limit**: 4 or 6 suggestions?
   - **Decision**: 4 for mobile, 6 for desktop (responsive)

3. **Suggestion Persistence**: Remember last clicked tool?
   - **Decision**: No for Phase 1, Yes for Phase 2 (Mem0 integration)

4. **Admin UI**: How to edit suggestions without SQL?
   - **Decision**: Backlog, manual SQL for Phase 1

---

## Appendix: Files Modified/Created

### New Files
```
orchestrator/migrations/add_app_suggestions.py
orchestrator/api/tools.py (new endpoint)
frontend/components/suggestions/ToolSuggestionBar.tsx
frontend/components/suggestions/SuggestionChip.tsx
frontend/components/suggestions/types.ts
tests/api/test_tools_suggestions.py
tests/e2e/tool-suggestions.spec.ts
```

### Modified Files
```
orchestrator/api/__init__.py (register new endpoint)
frontend/components/chatbot/chat.tsx (add suggestion bar)
frontend/components/chatbot/multimodal-input.tsx (tool icon handlers)
frontend/lib/analytics.ts (tracking events)
```

---

## Sign-off

**Ready for Implementation**: ✅
**Estimated Duration**: 6 days (1 week sprint)
**Assigned To**: TBD
**Reviewer**: TBD

---

**End of PRD-40**
