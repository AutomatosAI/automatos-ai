"""Shared helpers — SSE parser, test data randomization, assertion utils."""

import random
import string
import time

# ---------------------------------------------------------------------------
# Run-level ID (unique per test session)
# ---------------------------------------------------------------------------
RUN_ID = hex(int(time.time()))[2:]


def uid(prefix: str = "test") -> str:
    """Generate a unique ID like 'test-65f3a1b2-a1b2c3'."""
    rand = "".join(random.choices(string.hexdigits[:16], k=6))
    return f"{prefix}-{RUN_ID}-{rand}"


def pick(items: list):
    """Pick a random element from a list."""
    return random.choice(items)


# ---------------------------------------------------------------------------
# Test data pools (ported from JS test-data.js)
# ---------------------------------------------------------------------------
CHAT_MESSAGES = [
    "Can you help me understand my workspace setup?",
    "Summarize what agents are available",
    "How do I connect a new tool?",
    "What routing rules are configured?",
    "Show me my recent workflow history",
    "Explain how skills work in the platform",
    "What integrations are active?",
    "Help me set up a new recipe",
    "Check the health of my system",
    "What are my current costs?",
    "How do channels work?",
    "Tell me about personas",
]

CHAT_FOLLOWUPS = [
    "Can you tell me more about that?",
    "Please elaborate on the details",
    "What are the limitations?",
    "Give me a specific example",
    "How does that compare to alternatives?",
    "Break that down further",
]

PERSONA_NAMES = [
    "QA Test Assistant",
    "API Validator Bot",
    "Integration Tester",
    "Smoke Test Helper",
    "Nightly QA Agent",
    "Pipeline Checker",
    "Endpoint Monitor",
    "Health Probe Bot",
]

PERSONA_PROMPTS = [
    "You are a concise QA assistant. Answer briefly and focus on test results.",
    "You are a technical API testing bot. Validate responses accurately.",
    "You are a friendly integration tester. Help verify system connections.",
    "You are a health monitoring assistant. Report system status clearly.",
]

CHANNEL_NAMES = [
    "test-webhook-channel",
    "qa-notification-hook",
    "nightly-alert-channel",
    "api-test-webhook",
    "integration-channel",
    "smoke-test-hook",
]

ROUTING_KEYWORDS = [
    ["test", "qa", "testing", "verify"],
    ["help", "support", "assist", "guide"],
    ["deploy", "release", "ship", "publish"],
    ["monitor", "health", "status", "check"],
    ["analyze", "report", "metrics", "stats"],
]

SEARCH_TERMS = ["github", "slack", "email", "calendar", "jira", "notion", "discord", "sheets"]

SKILL_TASKS = [
    {"description": "Send a notification to the team", "type": "communication"},
    {"description": "Analyze the latest metrics data", "type": "analysis"},
    {"description": "Generate a weekly summary report", "type": "reporting"},
    {"description": "Check deployment pipeline status", "type": "devops"},
    {"description": "Schedule a recurring task", "type": "scheduling"},
]

WEBHOOK_PAYLOADS = [
    {"source": "api-test", "event": "test.run", "data": {"run_id": "test-1"}},
    {"source": "qa-runner", "event": "test.complete", "data": {"passed": True}},
    {"source": "regression", "event": "suite.done", "data": {"failures": 0}},
    {"source": "monitor", "event": "health.check", "data": {"status": "ok"}},
]

RECIPE_SEARCH_TERMS = ["email", "slack", "deploy", "monitor", "report", "onboard"]

MODEL_TASK_TYPES = ["chat", "code", "analysis", "creative", "summarization"]


# ---------------------------------------------------------------------------
# SSE stream parser (for /api/chat POST)
# ---------------------------------------------------------------------------
def parse_sse_response(response) -> dict:
    """Parse an AI SDK SSE stream response into structured data.

    Works with httpx streaming or a raw text body.
    Returns: {text, chat_id, data_events, tool_events, status, raw}
    """
    text_parts = []
    data_events = []
    tool_events = []
    chat_id = None
    raw_lines = []

    # Accept either a string or an httpx response
    if hasattr(response, "text"):
        body = response.text
    else:
        body = str(response)

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_lines.append(line)

        # 0:"text" — text delta
        if line.startswith("0:"):
            chunk = line[2:].strip('"')
            text_parts.append(chunk)

        # d:{json} — data event
        elif line.startswith("d:"):
            try:
                import json
                data = json.loads(line[2:])
                data_events.append(data)
                if isinstance(data, dict):
                    if "chatId" in data:
                        chat_id = data["chatId"]
                    event_type = data.get("type", "")
                    if "tool" in event_type:
                        tool_events.append(data)
            except Exception:
                pass

        # 2:[json] — data array
        elif line.startswith("2:"):
            try:
                import json
                arr = json.loads(line[2:])
                if isinstance(arr, list):
                    data_events.extend(arr)
            except Exception:
                pass

        # e:{json} — error event
        elif line.startswith("e:"):
            try:
                import json
                data_events.append({"type": "error", "data": json.loads(line[2:])})
            except Exception:
                pass

    return {
        "text": "".join(text_parts),
        "chat_id": chat_id,
        "data_events": data_events,
        "tool_events": tool_events,
        "raw": raw_lines,
    }
