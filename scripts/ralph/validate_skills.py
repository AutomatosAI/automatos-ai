#!/usr/bin/env python3
"""Validate all rewritten SKILL.md files meet Automatos quality bar.
No external dependencies — stdlib only.
"""

import os
import re
import sys

SKILLS_DIR = "/Users/gkavanagh/Development/Automatos-AI-Platform/automatos-skills"

# Pre-existing skills (exempt from validation)
EXEMPT_SKILLS = {
    "sentinel", "scout", "atlas", "echo", "forge", "harper", "oracle",
    "rally", "bug-fixer", "qa-engineer", "web-research",
    "gmail-automation", "google-calendar-automation", "jira-admin",
    "jira-automation", "linkedin-automation", "slack-automation",
    "tiktok-automation", "twitter-automation",
}

VALID_TOOLS = {
    # Platform
    "platform_get_system_health", "platform_get_logs", "platform_get_llm_usage",
    "platform_get_cost_breakdown", "platform_workspace_stats", "platform_submit_report",
    "platform_get_latest_report", "platform_create_task", "platform_list_tasks",
    "platform_board_summary", "platform_search_memory", "platform_search_chat_history",
    "platform_query_loki_logs", "platform_query_prometheus", "platform_publish_blog_post",
    "platform_list_blog_posts", "platform_create_agent", "platform_list_agents",
    "platform_schedule_task", "platform_field_query", "platform_field_inject",
    # Workspace
    "workspace_read_file", "workspace_write_file", "workspace_list_dir",
    "workspace_grep", "workspace_exec", "workspace_git",
    # Composio
    "composio_execute",
}

BANNED_REFS = re.compile(r"\b(Cursor|OpenClaw|Qwen|Copilot|ChatGPT|OpenAI)\b", re.IGNORECASE)


def parse_frontmatter_simple(content: str) -> dict | None:
    """Parse YAML frontmatter using regex — handles the simple format we use."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    fm_text = match.group(1)
    result = {}

    # Extract simple key: value pairs
    for m in re.finditer(r"^(\w+):\s*(.+)$", fm_text, re.MULTILINE):
        key, val = m.group(1), m.group(2).strip()
        # Handle quoted strings
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        # Handle inline lists [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
        result[key] = val

    # Extract tools list (  - name: ... / description: ...)
    tools = []
    tool_blocks = re.findall(
        r"  - name:\s*(.+)\n    description:\s*(.+)",
        fm_text,
    )
    for name, desc in tool_blocks:
        tools.append({
            "name": name.strip().strip("'\""),
            "description": desc.strip().strip("'\""),
        })
    if tools:
        result["tools"] = tools

    return result


def validate_skill(slug: str, path: str) -> list[str]:
    """Validate a single SKILL.md. Returns list of errors."""
    errors = []
    with open(path) as f:
        content = f.read()

    lines = content.strip().split("\n")
    line_count = len(lines)

    # 1. Frontmatter
    fm = parse_frontmatter_simple(content)
    if fm is None:
        errors.append("Missing YAML frontmatter (---)")
        return errors

    required = {"name", "description", "version", "tags", "category", "tools"}
    missing = required - set(fm.keys())
    if missing:
        errors.append(f"Missing frontmatter fields: {missing}")

    # 2. Tools format: list of {name, description}
    tools = fm.get("tools", [])
    if not isinstance(tools, list) or len(tools) == 0:
        errors.append("tools: must be a non-empty list of {name, description}")
    else:
        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                errors.append(f"tools[{i}]: not a dict")
                continue
            if "name" not in tool:
                errors.append(f"tools[{i}]: missing 'name'")
            elif tool["name"] not in VALID_TOOLS:
                errors.append(f"tools[{i}]: unknown tool '{tool['name']}'")
            if "description" not in tool:
                errors.append(f"tools[{i}]: missing 'description'")

    # 3. Workflow section with JSON tool call
    has_workflow = bool(re.search(r"##\s+Workflow", content, re.IGNORECASE))
    if not has_workflow:
        errors.append("Missing ## Workflow section")

    has_json_block = bool(re.search(r"```json", content))
    if not has_json_block:
        errors.append("No ```json tool call block found")

    # 4. Output format section
    has_output = bool(re.search(r"##\s+Output\s+Format", content, re.IGNORECASE))
    if not has_output:
        errors.append("Missing ## Output Format section")

    # 5. Anti-pattern section
    has_anti = bool(re.search(r"##\s+What\s+NOT\s+To\s+Do", content, re.IGNORECASE))
    if not has_anti:
        errors.append("Missing ## What NOT To Do section")

    # 6. No banned external references
    banned = BANNED_REFS.findall(content)
    if banned:
        errors.append(f"Banned external references: {banned}")

    # 7. Line count (soft check)
    if line_count < 50:
        errors.append(f"Too short: {line_count} lines (min ~60)")
    elif line_count > 120:
        errors.append(f"Too long: {line_count} lines (max ~100)")

    return errors


def main():
    all_slugs = sorted(
        d for d in os.listdir(SKILLS_DIR)
        if os.path.isfile(os.path.join(SKILLS_DIR, d, "SKILL.md"))
    )

    rewritten = [s for s in all_slugs if s not in EXEMPT_SKILLS]
    print(f"Total skills found: {len(all_slugs)}")
    print(f"Rewritten skills to validate: {len(rewritten)}")
    print(f"Exempt (pre-existing): {len(all_slugs) - len(rewritten)}")
    print("=" * 60)

    pass_count = 0
    fail_count = 0
    warn_count = 0

    for slug in rewritten:
        path = os.path.join(SKILLS_DIR, slug, "SKILL.md")
        errors = validate_skill(slug, path)

        hard_errors = [e for e in errors if not e.startswith("Too short") and not e.startswith("Too long")]
        warnings = [e for e in errors if e.startswith("Too short") or e.startswith("Too long")]

        if hard_errors:
            print(f"FAIL: {slug}")
            for e in hard_errors:
                print(f"  - {e}")
            for w in warnings:
                print(f"  - WARN: {w}")
            fail_count += 1
        elif warnings:
            print(f"WARN: {slug}")
            for w in warnings:
                print(f"  - {w}")
            warn_count += 1
        else:
            print(f"OK:   {slug}")
            pass_count += 1

    print("=" * 60)
    print(f"PASS: {pass_count}  WARN: {warn_count}  FAIL: {fail_count}  TOTAL: {len(rewritten)}")

    if fail_count > 0:
        sys.exit(1)
    else:
        print("\nAll rewritten skills pass validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
