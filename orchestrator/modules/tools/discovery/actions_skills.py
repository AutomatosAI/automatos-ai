"""
Skill Editing Action Definitions
=================================

Platform tools that let agents read, create, edit, and delete workspace skills.
Wraps the workspace_skills.py REST API behind the standard platform-tool
interface so any agent (VECTOR, FORGE, Auto, etc.) can iterate on its own
skill content without going through the UI.

Fork-on-edit semantics carry through: editing a marketplace skill creates a
workspace-owned fork (workspace_id=ctx.workspace_id, lineage recorded in
skill_metadata.forked_from_skill_id) — the marketplace original stays
untouched, agent assignments migrate to the fork, marketplace junction is
dropped. Marketplace integrity is preserved.

Browse / list / install / assign tools already exist in actions_marketplace.py
and actions_assignments.py — this file only adds the missing edit surface.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_skills_actions(registry: ActionRegistry) -> None:
    """Register skill read/create/update/delete + runtime (load/run/enable) tools."""

    # PRD-202 S2: trigger-based L2 activation. Attached skills are listed at L1
    # (name + description) only; the model calls load_skill to pull a skill's
    # full body into context when its task matches the description.
    registry.register(ActionDefinition(
        name="load_skill",
        description=(
            "Load a skill's full instructions (its SKILL.md body) into your "
            "context for THIS turn. Your attached skills are shown with only "
            "their name and description until you load them — call this when "
            "your current task matches a skill's description and you need its "
            "detailed step-by-step instructions. Returns the skill's full body."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact skill name to load (from the 'Available Skills' list in your prompt).",
                },
            },
            "required": ["name"],
        },
        permission_level="read",
        tags=["skills", "load", "progressive-disclosure", "l2"],
        examples=[
            "load the growth-hacker skill",
            "load_skill data-analysis",
            "pull the full instructions for the sql skill",
        ],
    ))

    # PRD-202 S3: L3 script execution via the workspace worker (sandboxed,
    # per-workspace, token-gated). Only the script OUTPUT enters context.
    registry.register(ActionDefinition(
        name="run_skill_script",
        description=(
            "Run one of a skill's bundled scripts in the sandboxed workspace "
            "worker and return only its OUTPUT (stdout/stderr) — the script "
            "source never enters your context. Use for a skill's executable "
            "helpers (data crunching, format conversion, generation). The "
            "skill's L3 scripts must be enabled for this workspace by an admin "
            "first (import/read is always allowed; running is opt-in)."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "skill": {"type": "string", "description": "Skill name whose script to run."},
                "script": {"type": "string", "description": "Script filename within the skill's scripts/ bundle (e.g. 'convert.py')."},
                "args": {"type": "string", "description": "Optional command-line arguments passed to the script."},
                "interpreter": {"type": "string", "description": "Optional interpreter (default inferred from extension: .py->python, .sh->bash, .js->node)."},
            },
            "required": ["skill", "script"],
        },
        permission_level="write",
        tags=["skills", "script", "execute", "l3", "worker"],
        examples=[
            "run the convert.py script from the docx skill",
            "run_skill_script skill=analytics script=summarize.py args='--period 7d'",
        ],
    ))

    # PRD-202 S4: workspace-admin enablement gate for L3 execution. Importing /
    # reading a skill is always allowed once scanned; running its scripts needs
    # explicit per-workspace enablement (scanner-pass required, audited).
    registry.register(ActionDefinition(
        name="platform_set_skill_script_execution",
        description=(
            "Enable or disable L3 script execution for a skill in this "
            "workspace (workspace-admin action, audited). Importing and reading "
            "a skill is always allowed once scanned — but running its bundled "
            "scripts is inert until enabled here. Enabling re-runs the security "
            "scanner and refuses if the skill has critical findings."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {"type": "integer", "description": "Skill id to enable/disable script execution for. Either skill_id or skill_name."},
                "skill_name": {"type": "string", "description": "Skill name (used when skill_id is not known)."},
                "enabled": {"type": "boolean", "description": "true to enable L3 script execution, false to disable."},
            },
            "required": ["enabled"],
        },
        permission_level="write",
        admin_only=True,
        tags=["skills", "governance", "l3", "enablement", "admin"],
        examples=[
            "enable script execution for the analytics skill",
            "turn off L3 scripts for skill 42",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_skill_content",
        description=(
            "Read the full SKILL.md content (frontmatter + body) of a skill "
            "the workspace can access. Use this to inspect a skill before "
            "editing it, to compare versions, or to reuse content as a "
            "template. Works on marketplace skills enabled for the workspace "
            "and on workspace-owned skills (forked or user-created). "
            "Returns name, description, category, tags, version, raw content, "
            "origin (marketplace|workspace), and editable flag."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "integer",
                    "description": "Numeric skill id. Either skill_id or skill_name is required.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill name (case-insensitive partial match). Used when skill_id is not known.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["skills", "read", "content", "inspect"],
        examples=[
            "show me the VECTOR skill content",
            "read the platform-cost-watchdog skill",
            "what does the sentinel skill say",
            "open skill 42 for editing",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_create_workspace_skill",
        description=(
            "Create a new workspace-owned skill from SKILL.md content. The "
            "content must include valid YAML frontmatter (name, description, "
            "version, tags, tools) followed by the markdown body. The skill "
            "is auto-enabled for the workspace and runs through the security "
            "scanner before save: critical findings block, high-severity "
            "findings require acknowledge_warnings=true. Use when the agent "
            "needs to draft a brand-new skill, not when editing an existing "
            "one (use platform_update_skill for that)."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name. Must be unique within the workspace. Max 255 chars.",
                },
                "content": {
                    "type": "string",
                    "description": "Full SKILL.md content: YAML frontmatter (--- name/description/version/tags/tools ---) followed by markdown body.",
                },
                "description": {
                    "type": "string",
                    "description": "One-line description. Falls back to frontmatter description if omitted.",
                },
                "category": {
                    "type": "string",
                    "description": "Category slug (e.g. 'analytics', 'platform-operations'). Falls back to frontmatter category.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tag list. Falls back to frontmatter tags.",
                },
                "skill_type": {
                    "type": "string",
                    "description": "One of: cognitive | technical | communication. Defaults to cognitive.",
                },
                "acknowledge_warnings": {
                    "type": "boolean",
                    "description": "Set true to allow save when only high-severity (non-blocking) scanner findings exist. Critical findings always block.",
                },
            },
            "required": ["name", "content"],
        },
        permission_level="write",
        tags=["skills", "create", "workspace", "draft", "fork"],
        examples=[
            "create a new skill called daily-standup",
            "draft a workspace skill from this content",
            "save this SKILL.md as a workspace skill",
            "make a new skill for VECTOR",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_skill",
        description=(
            "Edit a skill's SKILL.md content. If the target is a marketplace "
            "skill (workspace_id IS NULL) enabled for this workspace, this "
            "performs fork-on-edit: clones the skill into the workspace, "
            "records lineage in skill_metadata.forked_from_skill_id, migrates "
            "agent_skills assignments to the fork, and drops the marketplace "
            "junction. The marketplace original stays untouched — your edits "
            "are local. If the target is already workspace-owned, the update "
            "happens in place. Runs the security scanner — critical findings "
            "block, high-severity findings need acknowledge_warnings=true. "
            "Returns the (possibly new) skill_id, forked flag, and any warnings."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "integer",
                    "description": "Skill id to edit. Can be a marketplace skill (will fork) or a workspace skill (in-place).",
                },
                "content": {
                    "type": "string",
                    "description": "New SKILL.md content (frontmatter + body). Replaces the existing content entirely.",
                },
                "description": {
                    "type": "string",
                    "description": "Override description. Falls back to frontmatter description.",
                },
                "category": {
                    "type": "string",
                    "description": "Override category. Falls back to frontmatter category.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Override tags. Falls back to frontmatter tags.",
                },
                "acknowledge_warnings": {
                    "type": "boolean",
                    "description": "Set true to allow save when only high-severity (non-blocking) findings exist.",
                },
            },
            "required": ["skill_id", "content"],
        },
        permission_level="write",
        tags=["skills", "update", "edit", "fork", "workspace"],
        examples=[
            "edit the VECTOR skill",
            "update the platform-cost-watchdog skill content",
            "fork-and-edit the growth-hacker skill",
            "improve my workspace skill",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_workspace_skill",
        description=(
            "Delete a workspace-owned skill (forked or user-created). Drops "
            "agent_skills assignments and the workspace_enabled_skills "
            "junction first, then deletes the skill row. Refuses to delete "
            "marketplace skills — those are global and cannot be removed by "
            "a workspace; use platform_install_skill (disable) instead."
        ),
        category="skills",
        parameters={
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "integer",
                    "description": "Workspace-owned skill id to delete.",
                },
            },
            "required": ["skill_id"],
        },
        permission_level="write",
        tags=["skills", "delete", "workspace", "cleanup"],
        examples=[
            "delete the workspace skill called test-skill",
            "remove skill 142",
            "drop my draft skill",
        ],
    ))
