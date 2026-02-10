"""
Scratchpad Write Tool
=====================

Built-in tool injected ONLY during recipe step execution.
Allows agents to store named values in the shared recipe scratchpad
for downstream steps to use.
"""

# OpenAI function-call schema — injected into the tools list
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

SCRATCHPAD_TOOL_NAME = "scratchpad_write"


def handle_scratchpad_write(
    key: str,
    value: str,
    scratchpad,
    step_order: int,
) -> str:
    """
    Execute the scratchpad_write action.

    Returns a confirmation string for the LLM tool-call response.
    """
    scratchpad.write_export(step_order, key, value)
    return f"Stored '{key}' in scratchpad."
