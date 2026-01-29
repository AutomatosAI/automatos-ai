# Schema-Driven Widget Architecture Plan

## Problem Statement
- 863 Composio apps with 15,000+ features
- Current approach hardcodes action names (GMAIL, OUTLOOK, etc.)
- Not scalable - each new app requires code changes

## Solution: Schema-Driven Detection

### Key Insight
`composio_actions_cache` table already stores:
- `parameters` JSONB - input parameter schema
- `response_schema` JSONB - expected response structure

We should **USE** these schemas instead of hardcoding action names.

---

## Phase 1: ResponseTypeDetector (Backend)

### New File: `orchestrator/modules/tools/formatting/schema_detector.py`

```python
"""
Schema-Driven Response Type Detection

Detects widget type from response_schema, not action names.
Works for ALL 863+ Composio apps without hardcoding.
"""

from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResponseTypeDetector:
    """
    Detect widget type from Composio action response_schema.

    Instead of checking "GMAIL" in action_name, we inspect
    the response_schema to understand what data structure
    the action returns.
    """

    # Key patterns that indicate specific data types
    EMAIL_KEYS = {"emails", "messages", "threads", "inbox", "mail"}
    CODE_KEYS = {"code", "source", "snippet", "language", "syntax", "file_content"}
    DOCUMENT_KEYS = {"content", "text", "body", "document", "chunks", "pages", "markdown"}
    TABLE_KEYS = {"rows", "columns", "records", "items", "results", "data"}
    IMAGE_KEYS = {"image", "images", "url", "base64", "thumbnail", "attachments"}
    FILE_KEYS = {"files", "file", "path", "filename", "directory", "folders"}

    @classmethod
    def detect_from_schema(cls, response_schema: Optional[Dict[str, Any]]) -> str:
        """
        Detect widget type from response_schema JSONB.

        Args:
            response_schema: The response_schema from composio_actions_cache

        Returns:
            Widget type: 'email', 'code', 'document', 'data', 'image', 'file', 'generic'
        """
        if not response_schema:
            return "generic"

        properties = response_schema.get("properties", {})
        if not properties:
            # Try nested structures
            if "items" in response_schema:
                properties = response_schema["items"].get("properties", {})

        keys = set(k.lower() for k in properties.keys())

        # Priority order: most specific first

        # Email detection
        if keys & cls.EMAIL_KEYS:
            return "email"

        # Code detection
        if keys & cls.CODE_KEYS:
            # Check if it's actually code, not just "content"
            if "language" in keys or "syntax" in keys or "code" in keys:
                return "code"

        # Table/data detection (arrays of objects)
        if cls._has_array_of_objects(response_schema):
            return "data"

        # Document detection
        if len(keys & cls.DOCUMENT_KEYS) >= 2:
            return "document"

        # Image detection
        if keys & cls.IMAGE_KEYS:
            return "image"

        # File detection
        if keys & cls.FILE_KEYS:
            return "file"

        return "generic"

    @classmethod
    def detect_from_result(cls, result: Dict[str, Any]) -> str:
        """
        Fallback: detect type from actual result data shape.

        Used when response_schema is not available.
        """
        if not result or not isinstance(result, dict):
            return "generic"

        # Unwrap common nested structures
        data = result.get("data", result)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        keys = set(k.lower() for k in (data.keys() if isinstance(data, dict) else []))

        # Check for email-like data
        if keys & cls.EMAIL_KEYS:
            return "email"

        # Check for array of objects (table-like)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                return "data"

        # Check nested arrays
        for key in ["results", "items", "data", "records"]:
            nested = data.get(key) if isinstance(data, dict) else None
            if isinstance(nested, list) and len(nested) > 0 and isinstance(nested[0], dict):
                return "data"

        # Check for code
        if keys & cls.CODE_KEYS:
            return "code"

        # Check for document
        if keys & cls.DOCUMENT_KEYS:
            return "document"

        return "generic"

    @classmethod
    def _has_array_of_objects(cls, schema: Dict[str, Any]) -> bool:
        """Check if schema describes an array of objects (table-like)."""
        if schema.get("type") == "array":
            items = schema.get("items", {})
            return items.get("type") == "object"

        # Check properties for array types
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name.lower() in cls.TABLE_KEYS:
                if prop_schema.get("type") == "array":
                    items = prop_schema.get("items", {})
                    if items.get("type") == "object":
                        return True

        return False


class ParameterHintExtractor:
    """
    Extract LLM-friendly parameter hints from action parameter schemas.

    Instead of hardcoding "Gmail needs maxResults", we read the
    parameters JSONB and generate helpful hints dynamically.
    """

    @classmethod
    def extract_hints(cls, parameters: Optional[Dict[str, Any]]) -> str:
        """
        Convert JSON Schema parameters to human-readable hints.

        Args:
            parameters: The parameters JSONB from composio_actions_cache

        Returns:
            Formatted string of parameter hints for LLM context
        """
        if not parameters:
            return ""

        if parameters.get("type") != "object":
            return ""

        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))

        if not properties:
            return ""

        hints = []
        for param_name, param_def in properties.items():
            # Build hint line
            req_str = "REQUIRED" if param_name in required else "optional"
            param_type = param_def.get("type", "any")
            description = param_def.get("description", "")

            # Handle enums
            enum_vals = param_def.get("enum", [])
            if enum_vals:
                hint = f"  - {param_name} ({req_str}): {description} [options: {', '.join(str(v) for v in enum_vals[:5])}]"
            else:
                hint = f"  - {param_name} ({req_str}, {param_type}): {description}"

            # Handle defaults
            if "default" in param_def:
                hint += f" [default: {param_def['default']}]"

            hints.append(hint)

        return "\n".join(hints)

    @classmethod
    def get_required_params(cls, parameters: Optional[Dict[str, Any]]) -> list:
        """Get list of required parameter names."""
        if not parameters:
            return []
        return parameters.get("required", [])
```

---

## Phase 2: Update Result Formatter

### Changes to `orchestrator/modules/tools/formatting/result_formatter.py`

Replace hardcoded checks like:
```python
# BAD: Hardcoded
if "GMAIL" in action and any(x in action for x in ["LIST", "FETCH"]):
```

With schema-driven:
```python
# GOOD: Schema-driven
from modules.tools.formatting.schema_detector import ResponseTypeDetector

# Get action metadata from cache
action_metadata = db.query(ComposioActionCache).filter_by(action_name=action).first()

# Detect type from schema
if action_metadata and action_metadata.response_schema:
    widget_type = ResponseTypeDetector.detect_from_schema(action_metadata.response_schema)
else:
    widget_type = ResponseTypeDetector.detect_from_result(result)

# Route to appropriate formatter
if widget_type == "email":
    frontend_data["emails"] = cls._extract_emails_generic(result)
elif widget_type == "data":
    frontend_data["table"] = cls._extract_table_generic(result)
# ... etc
```

---

## Phase 3: Dynamic LLM Parameter Hints

### Changes to `orchestrator/consumers/chatbot/service.py`

When building tool context, inject parameter hints from schemas:

```python
from modules.tools.formatting.schema_detector import ParameterHintExtractor

# In get_chat_tools() or where Composio hints are injected:
if action_cache := db.query(ComposioActionCache).filter_by(action_name=action).first():
    param_hints = ParameterHintExtractor.extract_hints(action_cache.parameters)
    if param_hints:
        hint_lines.append(f"Parameters for {action}:")
        hint_lines.append(param_hints)
```

This way, for Gmail, the LLM automatically learns:
```
Parameters for GMAIL_LIST_EMAILS:
  - maxResults (optional, integer): Maximum number of messages to return [default: 100]
  - q (optional, string): Gmail search query (e.g., "is:unread after:2024/01/01")
  - labelIds (optional, array): Filter by label IDs
```

No hardcoding needed - it reads from the cached schema.

---

## Phase 4: Frontend Widget Router

### Changes to `frontend/components/widgets/router.ts`

Instead of hardcoded map:
```typescript
// BAD: Hardcoded
const TOOL_WIDGET_MAP = {
  GMAIL_LIST_EMAILS: 'email',
  GMAIL_SEND_EMAIL: 'email',
  // ... 15,000 more entries
}
```

Use backend API:
```typescript
// GOOD: Schema-driven
async function getWidgetTypeForAction(actionName: string): Promise<WidgetType> {
  const res = await fetch(`/api/tools/${actionName}/widget-type`)
  const data = await res.json()
  return data.widget_type || 'generic'
}
```

Or better - include widget_type in the tool-data response from backend.

---

## Implementation Priority

### Week 1: Create ResponseTypeDetector
1. Create `schema_detector.py` with ResponseTypeDetector
2. Add unit tests with sample schemas from real Composio actions
3. Test detection accuracy on 20+ different app types

### Week 2: Update Result Formatter
1. Refactor `format_for_frontend()` to use schema detection
2. Remove hardcoded action name checks
3. Add fallback to result-shape detection

### Week 3: Dynamic Parameter Hints
1. Add ParameterHintExtractor to LLM context building
2. Remove hardcoded parameter hints from prompts
3. Test with Gmail, Slack, GitHub, etc.

### Week 4: Frontend Updates
1. Add widget_type to tool-data SSE events
2. Update frontend router to use backend-provided type
3. Remove hardcoded TOOL_WIDGET_MAP

---

## Success Metrics

- **Zero hardcoded action names** in result_formatter.py
- **Auto-detection accuracy > 95%** for widget types
- **New app support** requires only cache sync, no code changes
- **LLM parameter usage** improves (correct params on first try)

---

## Files to Modify

| File | Change |
|------|--------|
| `modules/tools/formatting/schema_detector.py` | NEW - Create detector class |
| `modules/tools/formatting/result_formatter.py` | Remove hardcoded checks, use detector |
| `consumers/chatbot/service.py` | Add dynamic param hints from schemas |
| `consumers/chatbot/tool_router.py` | Include action metadata in tool context |
| `frontend/components/widgets/router.ts` | Use backend widget_type, remove map |
| `api/tools.py` | NEW endpoint: GET /api/tools/{action}/widget-type |
