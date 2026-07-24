"""CodeGraph ActionDefinitions (PRD-165 S4) — code navigation as agent tools.

Promotes the CodeGraph executors to first-class platform tools so any agent can
reason about code structure: list projects, find symbols, trace call graphs,
analyse change impact, and read an architecture overview.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_codegraph_actions(registry: ActionRegistry) -> None:
    """Register CodeGraph executors as platform actions."""

    registry.register(ActionDefinition(
        name="platform_codegraph_list_projects",
        description=(
            "List the code repositories indexed for this workspace, with their "
            "language, symbol counts, last-indexed time, and status. Use first to "
            "discover the project name other codegraph tools need."
        ),
        category="codegraph",
        parameters={"type": "object", "properties": {}},
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "projects", "repos"],
        examples=[
            "what code repos are indexed?",
            "list the codebases the agents can search",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_search",
        description=(
            "Search a codebase by meaning (semantic, default) or by name (fuzzy). "
            "Returns matching symbols with file path, line number, and signature. "
            "Use to locate where something lives before tracing calls or impact."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Indexed project name (from platform_codegraph_list_projects)."},
                "query": {"type": "string", "description": "What to find — a concept ('retry logic') or a name ('AgentFactory')."},
                "mode": {"type": "string", "description": "'semantic' (default, by meaning) or 'fuzzy' (by name)."},
                "limit": {"type": "integer", "description": "Max results (default from config)."},
            },
            "required": ["project", "query"],
        },
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "search", "semantic"],
        examples=[
            "find where retry logic lives in the orchestrator repo",
            "search the codebase for the tool router",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_get_symbol",
        description=(
            "Find a code symbol (function, class, method) by name in a project. "
            "Returns its file path, line number, signature, and docstring."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Indexed project name."},
                "symbol": {"type": "string", "description": "Symbol name to find (simple or qualified)."},
                "symbol_type": {"type": "string", "description": "Optional filter: function, class, method."},
                "limit": {"type": "integer", "description": "Max results (default from config)."},
            },
            "required": ["project", "symbol"],
        },
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "symbol", "definition"],
        examples=[
            "where is the function execute_task defined?",
            "find the class WorkflowOrchestrator",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_call_graph",
        description=(
            "Trace the call graph for a symbol: what it calls (outgoing) or what "
            "calls it (incoming). Answers 'what calls X?' and 'what does X call?'."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Indexed project name."},
                "symbol": {"type": "string", "description": "Symbol name (simple or qualified, e.g. 'module.py::Class::method')."},
                "direction": {"type": "string", "description": "'outgoing' (calls made, default), 'incoming' (callers), or 'both'."},
                "depth": {"type": "integer", "description": "Levels to traverse, 1-5 (default 1)."},
            },
            "required": ["project", "symbol"],
        },
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "call-graph", "callers"],
        examples=[
            "what calls AgentFactory.execute?",
            "what does the recipe executor call?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_dependencies",
        description=(
            "Change-impact analysis: which symbols depend on a given symbol (and "
            "which it depends on). Answers 'what breaks if I change Y?'."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Indexed project name."},
                "symbol": {"type": "string", "description": "Symbol name to analyse."},
                "direction": {"type": "string", "description": "'dependents' (what uses it), 'dependencies' (what it uses), or 'both' (default)."},
            },
            "required": ["project", "symbol"],
        },
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "impact", "dependencies"],
        examples=[
            "what breaks if I change the tool router?",
            "what depends on create_llm_manager?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_architecture",
        description=(
            "High-level architecture overview of a project: modules, key symbols, "
            "and dependency patterns. Use to orient before diving into specifics."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Indexed project name."},
                "focus_path": {"type": "string", "description": "Optional directory prefix to focus the overview on."},
            },
            "required": ["project"],
        },
        permission_level="read",
        promoted=True,
        tags=["codegraph", "code", "architecture", "overview"],
        examples=[
            "give me an architecture overview of the orchestrator repo",
            "what are the main modules in this codebase?",
        ],
    ))

    # ------------------------------------------------------------------
    # PRD-183 S4 — write tools (index / reindex / auto-reindex setter)
    # ------------------------------------------------------------------

    registry.register(ActionDefinition(
        name="platform_codegraph_index",
        description=(
            "Index (or refresh) a GitHub repository into the code graph so agents "
            "can search it and trace calls. Use to onboard a new codebase before "
            "the read tools can answer 'what calls X?' for it."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Name to give the indexed project."},
                "github_url": {"type": "string", "description": "Repository URL, e.g. https://github.com/org/repo."},
                "branch": {"type": "string", "description": "Branch to index (default 'main')."},
                "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional path fragments to skip (e.g. ['tests', 'docs'])."},
            },
            "required": ["project", "github_url"],
        },
        permission_level="write",
        promoted=True,
        tags=["codegraph", "code", "index", "onboard", "repo"],
        examples=[
            "index the automatos-ai repo",
            "add this github repo to the code graph",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_reindex",
        description=(
            "Re-index an already-onboarded project to refresh a stale code graph "
            "after code changes. Answers 'the graph is out of date, re-index it'."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Existing indexed project name."},
            },
            "required": ["project"],
        },
        permission_level="write",
        promoted=True,
        tags=["codegraph", "code", "reindex", "refresh"],
        examples=[
            "reindex the orchestrator repo",
            "refresh the code graph for automatos-ai",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_codegraph_set_auto_reindex",
        description=(
            "Turn push-driven auto-reindex on or off for a project. When on, a "
            "GitHub push webhook reindexes the repo automatically so the code "
            "graph stays fresh. Use to enable hands-off freshness for a repo."
        ),
        category="codegraph",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Existing indexed project name."},
                "enabled": {"type": "boolean", "description": "True to auto-reindex on push, False to disable."},
            },
            "required": ["project", "enabled"],
        },
        permission_level="write",
        promoted=True,
        tags=["codegraph", "code", "auto-reindex", "webhook", "freshness"],
        examples=[
            "enable auto-reindex for the orchestrator repo on every push",
            "stop auto-reindexing the automatos-ai repo",
        ],
    ))
