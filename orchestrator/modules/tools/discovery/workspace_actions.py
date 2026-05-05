"""
Workspace Action Definitions
==============================

Agent tools for interacting with workspace files on the worker volume.
These let agents search code, read/write files, run commands, and use git
— all proxied through the workspace worker HTTP API.

Registered alongside platform_actions in the ActionRegistry.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_workspace_actions(registry: ActionRegistry) -> None:
    """Register all workspace tools with the action registry."""

    registry.register(ActionDefinition(
        name="workspace_read_file",
        description=(
            "Read the contents of a file from the workspace repository. "
            "Returns the file text, size, and language. Use when you need to "
            "examine source code, configuration files, or any text file in the repo."
        ),
        category="workspace_files",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to the file inside the workspace. "
                        "All paths are relative to the workspace root. "
                        "Repo files live under repos/ (e.g. 'repos/my-app/src/main.py'). "
                        "Artifacts and other workspace files use their direct path "
                        "(e.g. 'artifacts/results/test-summary.json'). "
                        "Do NOT prepend 'repos/' to paths that are already workspace-root-relative."
                    ),
                },
            },
            "required": ["path"],
        },
        permission_level="read",
        tags=["workspace", "files", "read", "code"],
        examples=[
            "read src/main.py",
            "show me the package.json",
            "what's in the config file?",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_write_file",
        description=(
            "Write or create a file in the workspace repository. Overwrites the "
            "file if it exists, creates it (including parent directories) if not. "
            "Use to fix bugs, add code, update configuration, or create new files."
        ),
        category="workspace_files",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path for the file inside the workspace. "
                        "All paths are relative to the workspace root. "
                        "Repo files live under repos/ (e.g. 'repos/my-app/src/utils.py'). "
                        "Other workspace paths use their direct path "
                        "(e.g. 'artifacts/data.json'). "
                        "Do NOT prepend 'repos/' to paths that are already workspace-root-relative."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The full text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
        permission_level="write",
        tags=["workspace", "files", "write", "code"],
        examples=[
            "fix the bug on line 42 of main.py",
            "create a new test file",
            "update the README",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_list_dir",
        description=(
            "List files and directories at a path in the workspace. Returns name, "
            "type (file/dir), and size for each entry. Use to explore the repo "
            "structure, find files, or understand project layout."
        ),
        category="workspace_files",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative directory path inside the workspace. "
                        "Defaults to '.' (workspace root)."
                    ),
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["workspace", "files", "list", "directory"],
        examples=[
            "what files are in the repo?",
            "list the src directory",
            "show me the project structure",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_grep",
        description=(
            "Search for a regex pattern across files in the workspace. Returns "
            "matching file paths, line numbers, and line content. Use to find "
            "function definitions, error messages, TODOs, or any text pattern."
        ),
        category="workspace_files",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g. 'def handle_login', 'TODO', 'import os').",
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory to search in. Defaults to '.' (entire workspace).",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.ts', '*.json').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default 50, max 200).",
                },
            },
            "required": ["pattern"],
        },
        permission_level="read",
        tags=["workspace", "search", "grep", "code"],
        examples=[
            "search for 'def handle_error' in the code",
            "find all TODO comments in Python files",
            "where is the database connection configured?",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_exec",
        description=(
            "Run a sandboxed shell command in the workspace (e.g. pytest, npm test, "
            "python script.py). Only whitelisted commands are allowed. Returns exit "
            "code, stdout, stderr, and duration. Use to run tests, linters, builds, "
            "or any development command."
        ),
        category="workspace_exec",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to execute (e.g. 'pytest tests/', 'npm test', "
                        "'python -m mypy src/')."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory relative to workspace root "
                        "(e.g. 'repos/my-app'). Defaults to workspace root."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait (default 120, max 300).",
                },
            },
            "required": ["command"],
        },
        permission_level="write",
        tags=["workspace", "exec", "shell", "test", "build"],
        examples=[
            "run the tests",
            "run pytest on the backend",
            "execute npm run build",
            "run the linter",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_html_to_png",
        description=(
            "Render an HTML page to a PNG image inside the workspace using "
            "headless Chromium. Use to turn templated HTML (e.g. social card "
            "templates from a cloned repo) into shareable PNGs. The output file "
            "is automatically registered as a deliverable and shows up in the "
            "Deliverables Gallery, Workspace Explorer, and Mission Outputs."
        ),
        category="workspace_render",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Absolute URL to render. Must be either: "
                        "(a) a 'file://' URL pointing INSIDE this workspace "
                        "(e.g. 'file:///workspaces/<id>/repos/automatos-social/render/index.html?template=definition&size=ig_post&...'), "
                        "or (b) an 'http(s)://' URL. Other schemes are rejected."
                    ),
                },
                "viewport": {
                    "type": "object",
                    "description": (
                        "Browser viewport in pixels. Set to the exact final "
                        "image dimensions — e.g. Instagram 4:5 = {w:1080,h:1350}, "
                        "LinkedIn = {w:1200,h:628}, IG Story = {w:1080,h:1920}, "
                        "Twitter/YouTube = {w:1600,h:900}."
                    ),
                    "properties": {
                        "w": {"type": "integer", "description": "Width in px (max 4096)."},
                        "h": {"type": "integer", "description": "Height in px (max 4096)."},
                    },
                    "required": ["w", "h"],
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative path for the PNG. Must end in '.png'. "
                        "Convention: 'deliverables/social/{YYYY-MM-DD}/{template}_{size}.png' "
                        "(e.g. 'deliverables/social/2026-04-29/definition_ig_post.png'). "
                        "Parent directories are created automatically."
                    ),
                },
                "wait_for": {
                    "type": "string",
                    "description": (
                        "CSS selector to await before screenshotting. Default "
                        "'[data-render-ready=\\'true\\']' matches the Automatos "
                        "render protocol — pages set this attribute on body once "
                        "fonts have loaded and layout has settled."
                    ),
                },
                "full_page": {
                    "type": "boolean",
                    "description": (
                        "When true, capture the full scrollable height. Default "
                        "false — social cards are sized to the viewport, so a "
                        "viewport screenshot is what you want."
                    ),
                },
            },
            "required": ["url", "viewport", "output_path"],
        },
        permission_level="write",
        promoted=True,
        tags=["workspace", "render", "html", "png", "screenshot", "social"],
        examples=[
            "render the definition template for instagram",
            "screenshot the html page to a png",
            "generate a social card png from the template",
            "produce a 1080x1350 instagram post from the html",
        ],
    ))

    registry.register(ActionDefinition(
        name="workspace_git",
        description=(
            "Execute a git operation in the workspace repository. Allowed operations: "
            "clone, status, diff, add, commit, push, pull, log, branch, checkout, stash, "
            "show, blame, fetch. For clone, pass the repo HTTPS URL as 'args' — the repo "
            "is cloned into repos/{repo-name}."
        ),
        category="workspace_git",
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "clone", "status", "diff", "add", "commit", "push", "pull",
                        "log", "branch", "checkout", "stash", "show", "blame", "fetch",
                    ],
                    "description": "The git operation to perform.",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory relative to workspace root, typically "
                        "the repo path (e.g. 'repos/my-app')."
                    ),
                },
                "args": {
                    "type": "string",
                    "description": (
                        "Additional arguments for the git command "
                        "(e.g. '-m \"fix login bug\"' for commit, '-A' for add, "
                        "'--oneline -10' for log)."
                    ),
                },
            },
            "required": ["operation"],
        },
        permission_level="write",
        tags=["workspace", "git", "vcs", "commit", "push"],
        examples=[
            "check git status",
            "commit the changes with message 'fix login bug'",
            "push to remote",
            "show the last 5 commits",
        ],
    ))
