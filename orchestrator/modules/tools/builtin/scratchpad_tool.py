"""
Scratchpad Tools
================

Built-in tools injected ONLY during recipe step execution.
- scratchpad_write: store named values for downstream steps
- scratchpad_read: retrieve values written by earlier steps
"""

import json

# ---------------------------------------------------------------------------
# scratchpad_write
# ---------------------------------------------------------------------------

SCRATCHPAD_WRITE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "scratchpad_write",
        "description": (
            "Store a named value in the shared recipe scratchpad for other "
            "steps to use. Use this to pass specific data (URLs, IDs, file "
            "paths, branch names, etc.) to subsequent recipe steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "A short snake_case name for this value, "
                        "e.g. 'pr_url', 'branch_name', 'analysis'"
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The value to store",
                },
            },
            "required": ["key", "value"],
        },
    },
}

SCRATCHPAD_WRITE_NAME = "scratchpad_write"

# Keep old alias for backward compat with existing imports
SCRATCHPAD_TOOL_NAME = SCRATCHPAD_WRITE_NAME


def handle_scratchpad_write(
    key: str,
    value: str,
    scratchpad,
    step_order: int,
) -> str:
    """Execute the scratchpad_write action."""
    scratchpad.write_export(step_order, key, value)
    return f"Stored '{key}' in scratchpad."


# ---------------------------------------------------------------------------
# scratchpad_read
# ---------------------------------------------------------------------------

SCRATCHPAD_READ_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "scratchpad_read",
        "description": (
            "Read values from the shared recipe scratchpad that were written "
            "by earlier steps. Use this to retrieve data like test reports, "
            "analysis results, URLs, or any values exported by previous steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "The key to read. Use '*' or omit to read all "
                        "available exports. Otherwise provide the exact "
                        "snake_case key used by the writing step."
                    ),
                },
            },
            "required": [],
        },
    },
}

SCRATCHPAD_READ_NAME = "scratchpad_read"


def handle_scratchpad_read(
    key: str,
    scratchpad,
) -> str:
    """
    Execute the scratchpad_read action.

    If key is '*' or empty, returns all exports as JSON.
    Otherwise returns the value for a specific key, or an error message.
    """
    exports = scratchpad.get_exports()

    if not key or key == "*":
        if not exports:
            return "Scratchpad is empty — no exports from previous steps."
        return json.dumps(exports, indent=2)

    value = exports.get(key)
    if value is not None:
        return value

    available = list(exports.keys())
    if available:
        return (
            f"Key '{key}' not found in scratchpad. "
            f"Available keys: {', '.join(available)}"
        )
    return f"Key '{key}' not found — scratchpad is empty."
