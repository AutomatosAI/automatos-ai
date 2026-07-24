"""
Tool Registry - Centralized Tool Management System
===================================================
PRD-17: Dynamic Tool Assignment & Centralized Tool Management
REBUILD: Enhanced with AgentRegistry integration (Phase 1.2)

Provides a single source of truth for ALL platform tools, accessible to:
- Orchestrator (task decomposition with tool recommendations)
- Agent Factory (dynamic tool injection)
- AgentRegistry (tool-based agent lookup)
- ChatBot (tool-augmented responses)
- User/API (tool discovery & execution)
- Future integrations (plugins, extensions)

Design Principles:
- Non-breaking: Wraps existing tool systems without modifying them
- Unified: Single interface for all tool types
- Discoverable: Tools are queryable by category, task type, or name
- Secure: Maintains existing security controls
- Extensible: Easy to add new tools or categories
- Integrated: Works seamlessly with AgentRegistry and GlobalFunctionRegistry
"""

import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlalchemy import or_, func
from sqlalchemy.orm import Session
from config import config

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and filtering"""
    RESEARCH = "research"              # RAG, semantic search, CodeGraph
    FILE_OPERATIONS = "file_ops"       # read, write, delete files
    SHELL_COMMANDS = "shell"           # execute shell commands
    DATABASE_TOOLS = "database"        # SQL operations (future)
    SSH_TOOLS = "ssh"                  # SSH remote command execution
    API_TOOLS = "api"                  # REST API calls & HTTP requests
    GIT_OPERATIONS = "git"             # Git operations (future)
    COMMUNICATION = "communication"    # Slack, Email, etc.
    DEVELOPER = "developer"            # GitHub, GitLab, etc.
    PRODUCTIVITY = "productivity"      # Jira, Linear, etc.
    WIDGET = "widget"                  # Embedded chat-widget UI affordances


class SecurityLevel(Enum):
    """Security levels for tool execution"""
    SAFE = "safe"                      # No security risk (read-only queries)
    CAUTIOUS = "cautious"              # Moderate risk (writes, non-destructive)
    DANGEROUS = "dangerous"            # High risk (deletes, shell commands)
    CRITICAL = "critical"              # Critical risk (system modifications)


@dataclass
class ToolParameter:
    """Specification for a tool parameter"""
    name: str
    type: str                          # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, str]] = None  # For array types: {"type": "string"}
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function parameter format"""
        prop = {
            "type": self.type,
            "description": self.description
        }
        
        if self.enum:
            prop["enum"] = self.enum
        if self.default is not None:
            prop["default"] = self.default
        # OpenAI requires 'items' for array types
        if self.type == "array":
            prop["items"] = self.items or {"type": "string"}
        
        return prop


@dataclass
class ToolSpec:
    """
    Complete specification for a tool.
    
    This is the atomic unit of the tool registry - represents a single
    executable tool with all its metadata, parameters, and execution info.
    """
    name: str
    category: ToolCategory
    description: str
    executor_class: str                # Class name that executes this tool
    executor_method: str               # Method to call on the executor
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: Optional[str] = None      # Description of return value
    security_level: SecurityLevel = SecurityLevel.SAFE
    permissions_required: Dict[str, bool] = field(default_factory=lambda: {"read": True})
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_openai_format()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def to_markdown_doc(self) -> str:
        """Generate markdown documentation for the tool"""
        lines = [
            f"### {self.name}",
            f"**Category**: {self.category.value}",
            f"**Security**: {self.security_level.value}",
            "",
            self.description,
            "",
            "**Parameters**:",
        ]
        
        for param in self.parameters:
            req_marker = "required" if param.required else "optional"
            lines.append(f"- `{param.name}` ({param.type}, {req_marker}): {param.description}")
        
        if self.examples:
            lines.append("")
            lines.append("**Examples**:")
            for example in self.examples:
                lines.append(f"```json")
                lines.append(f'{example}')
                lines.append(f"```")
        
        return "\n".join(lines)


class ToolRegistry:
    """
    Centralized registry for all platform tools.
    
    Responsibilities:
    - Register tools from all executors (platform, action, integrations)
    - Provide unified query interface
    - Map task types to tool requirements
    - Export tools in multiple formats (OpenAI, markdown, etc.)
    - Track tool usage and availability
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        self.tools: Dict[str, ToolSpec] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        self.logger = logger
        
        # Initialize with core platform tools
        self._register_core_tools()
        
        self.logger.info(f"ToolRegistry initialized with {len(self.tools)} tools")

    def _build_composio_tool_name(self, action_name: str) -> str:
        """
        Build a safe OpenAI function name for Composio actions.
        OpenAI enforces a max length of 64 chars on function names.
        """
        if not action_name:
            return "composio_action"
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", action_name.strip())
        base = f"composio_{normalized}"
        if len(base) <= 64:
            return base
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        max_slug_len = 64 - len("composio__") - len(digest)
        trimmed = normalized[:max_slug_len].rstrip("_")
        return f"composio_{trimmed}_{digest}"
    
    def register_tool(self, tool: ToolSpec) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: ToolSpec to register
        """
        if tool.name in self.tools:
            self.logger.warning(f"Tool {tool.name} already registered, updating...")
        
        self.tools[tool.name] = tool
        
        # Add to category index
        if tool.name not in self.categories[tool.category]:
            self.categories[tool.category].append(tool.name)
        
        self.logger.debug(f"Registered tool: {tool.name} (category: {tool.category.value}, security: {tool.security_level.value})")
    
    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self, active_only: bool = True) -> List[ToolSpec]:
        """Get all registered tools"""
        tools = list(self.tools.values())
        if active_only:
            tools = [t for t in tools if t.is_active]
        return tools
    
    def get_tools_by_category(self, category: ToolCategory, active_only: bool = True) -> List[ToolSpec]:
        """Get all tools in a specific category"""
        tool_names = self.categories.get(category, [])
        tools = [self.tools[name] for name in tool_names if name in self.tools]
        
        if active_only:
            tools = [t for t in tools if t.is_active]
        
        return tools
    
    def get_tools_for_task_type(self, task_type: str) -> List[str]:
        """
        Get recommended tool categories for a task type.
        
        Args:
            task_type: Type of task (code_review, bug_fix, server_restart, etc.)
        
        Returns:
            List of tool category names
        """
        # Import mapper to avoid circular dependency
        from modules.tools.services.tool_capability_mapper import ToolCapabilityMapper
        
        mapper = ToolCapabilityMapper()
        return mapper.get_tool_categories_for_task(task_type)
    
    def get_tools_for_categories(self, categories: List[str]) -> List[ToolSpec]:
        """
        Get all tools for specified categories.
        
        Args:
            categories: List of category names (strings)
        
        Returns:
            List of ToolSpec objects
        """
        tools = []
        for cat_name in categories:
            try:
                category = ToolCategory(cat_name)
                tools.extend(self.get_tools_by_category(category))
            except ValueError:
                self.logger.warning(f"Unknown tool category: {cat_name}")
        
        return tools
    
    def build_tool_prompt(
        self,
        tool_categories: List[str],
        include_examples: bool = True,
        format: str = "markdown"
    ) -> str:
        """
        Build a prompt section describing available tools.
        
        Args:
            tool_categories: List of category names to include
            include_examples: Whether to include usage examples
            format: Output format ("markdown", "json")
        
        Returns:
            Formatted tool documentation string
        """
        tools = self.get_tools_for_categories(tool_categories)
        
        if not tools:
            return ""
        
        if format == "markdown":
            return self._build_markdown_prompt(tools, include_examples)
        elif format == "json":
            return self._build_json_prompt(tools)
        else:
            return self._build_markdown_prompt(tools, include_examples)
    
    def _build_markdown_prompt(self, tools: List[ToolSpec], include_examples: bool) -> str:
        """Build markdown-formatted tool documentation"""
        lines = [
            "",
            "## 🔧 Available Tools",
            "",
            "You have access to the following tools to complete this task:",
            ""
        ]
        
        # Group by category
        tools_by_category: Dict[ToolCategory, List[ToolSpec]] = {}
        for tool in tools:
            if tool.category not in tools_by_category:
                tools_by_category[tool.category] = []
            tools_by_category[tool.category].append(tool)
        
        # Build documentation for each category
        for category, category_tools in tools_by_category.items():
            lines.append(f"### {category.value.replace('_', ' ').title()} Tools")
            lines.append("")
            
            for tool in category_tools:
                lines.append(f"**{tool.name}** - {tool.description}")
                lines.append(f'```json')
                lines.append(f'{{"action": "{tool.name}", "params": {{')
                
                param_lines = []
                for param in tool.parameters:
                    param_lines.append(f'  "{param.name}": "<{param.type}>"  // {param.description}')
                lines.append(",\n".join(param_lines))
                
                lines.append(f'}}}}')
                lines.append(f'```')
                
                if include_examples and tool.examples:
                    lines.append(f"Example: `{tool.examples[0]}`")
                
                lines.append("")
        
        lines.extend([
            "## Usage Instructions:",
            "",
            "1. Use tools by outputting JSON blocks in your response",
            "2. You can call multiple tools in sequence",
            "3. Wait for tool results before proceeding",
            "4. Use tool results to inform your final answer",
            ""
        ])
        
        return "\n".join(lines)
    
    def _build_json_prompt(self, tools: List[ToolSpec]) -> str:
        """Build JSON-formatted tool list"""
        import json
        return json.dumps([t.to_openai_format() for t in tools], indent=2)
    
    def export_openai_functions(self, tool_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Export tools in OpenAI function calling format.
        
        Args:
            tool_categories: Optional filter by categories
        
        Returns:
            List of function definitions for OpenAI API
        """
        if tool_categories:
            tools = self.get_tools_for_categories(tool_categories)
        else:
            tools = self.get_all_tools()
        
        return [t.to_openai_format() for t in tools]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        category_counts = {
            cat.value: len(tools)
            for cat, tools in self.categories.items()
        }
        
        security_counts = {}
        for tool in self.tools.values():
            level = tool.security_level.value
            security_counts[level] = security_counts.get(level, 0) + 1
        
        return {
            "total_tools": len(self.tools),
            "active_tools": len([t for t in self.tools.values() if t.is_active]),
            "categories": category_counts,
            "security_levels": security_counts,
            "last_updated": datetime.now().isoformat()
        }
    
    def _register_core_tools(self):
        """Register all core platform tools"""
        
        # ==========================================
        # RESEARCH TOOLS (from AgentPlatformTools)
        # ==========================================
        
        self.register_tool(ToolSpec(
            name="search_knowledge",
            category=ToolCategory.RESEARCH,
            description=(
                "Search INTERNAL workspace knowledge base — uploaded documents, guides, PDFs, and ingested files. "
                "Returns only content that was previously uploaded to this workspace. "
                "IMPORTANT: For live/current information, competitor research, market data, or anything NOT in your "
                "uploaded documents, use web search tools (TAVILY_SEARCH, COMPOSIO_SEARCH_WEB) instead. "
                "This tool searches local docs only. 2-attempt limit — vary your query if first attempt misses."
            ),
            executor_class="AgentPlatformTools",
            executor_method="execute_tool",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query - describe what you want to find",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of results (default: 5)",
                    required=False,
                    default=5
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_knowledge", "params": {"query": "How to create an agent?", "limit": 5}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="semantic_search",
            category=ToolCategory.RESEARCH,
            description=(
                "Find semantically similar content across all INTERNAL workspace documents using vector embeddings. "
                "Better than search_knowledge for concept-based queries where exact keywords may not match. "
                "Searches uploaded documents only — for external/live data, use web search tools instead."
            ),
            executor_class="AgentPlatformTools",
            executor_method="execute_tool",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Concept or topic to find similar content for",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum results (default: 5)",
                    required=False,
                    default=5
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "semantic_search", "params": {"query": "authentication patterns", "limit": 5}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="search_codebase",
            category=ToolCategory.RESEARCH,
            description=(
                "Search INTERNAL indexed codebases for functions, classes, and implementations. "
                "Only searches code previously ingested into the workspace via CodeGraph. "
                "For searching public repos or code on the internet, use web search tools instead."
            ),
            executor_class="AgentPlatformTools",
            executor_method="execute_tool",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Code pattern, function name, or concept to search for",
                    required=True
                ),
                ToolParameter(
                    name="file_type",
                    type="string",
                    description="Filter by file extension (e.g., 'py', 'ts', 'js')",
                    required=False
                ),
                ToolParameter(
                    name="project_name",
                    type="string",
                    description="Project name to search (default: 'Automatos-ai')",
                    required=False,
                    default="Automatos-ai"
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_codebase", "params": {"query": "authenticate_user", "project_name": "Automatos-ai"}}
            ]
        ))
        

        
        # ==========================================
        # MULTIMODAL RESEARCH TOOLS (PRD-19)
        # ==========================================
        
        self.register_tool(ToolSpec(
            name="search_tables",
            category=ToolCategory.RESEARCH,
            description=(
                "Search for tables and structured data extracted from UPLOADED documents. "
                "Returns tables with preserved structure in Markdown, CSV, and JSON formats. "
                "Only finds tables from documents already in the workspace knowledge base."
            ),
            executor_class="MultimodalKnowledgeTools",
            executor_method="search_tables",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="What kind of table or data to find (e.g., 'performance metrics', 'financial data')",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of tables to return (default: 5)",
                    required=False,
                    default=5
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_tables", "params": {"query": "API response times", "limit": 3}}
            ],
            metadata={"kb_type": "table", "added_in": "PRD-19"}
        ))
        
        self.register_tool(ToolSpec(
            name="search_images",
            category=ToolCategory.RESEARCH,
            description=(
                "Search for images, diagrams, and charts extracted from UPLOADED documents. "
                "Matches against AI-generated descriptions and OCR text. "
                "Useful for finding architecture diagrams, flowcharts, and screenshots from workspace docs."
            ),
            executor_class="MultimodalKnowledgeTools",
            executor_method="search_images",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="What kind of image or diagram to find (e.g., 'system architecture', 'workflow diagram')",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of images to return (default: 5)",
                    required=False,
                    default=5
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_images", "params": {"query": "database schema diagram", "limit": 3}}
            ],
            metadata={"kb_type": "image", "added_in": "PRD-19"}
        ))
        
        self.register_tool(ToolSpec(
            name="search_formulas",
            category=ToolCategory.RESEARCH,
            description=(
                "Search for mathematical formulas and equations extracted from UPLOADED documents. "
                "Returns LaTeX format with variable and operator extraction. "
                "Only finds formulas from documents already in the workspace knowledge base."
            ),
            executor_class="MultimodalKnowledgeTools",
            executor_method="search_formulas",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Mathematical concept or formula type (e.g., 'entropy formula', 'optimization algorithm')",
                    required=True
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum number of formulas to return (default: 5)",
                    required=False,
                    default=5
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_formulas", "params": {"query": "Shannon entropy", "limit": 3}}
            ],
            metadata={"kb_type": "formula", "added_in": "PRD-19"}
        ))
        
        self.register_tool(ToolSpec(
            name="search_multimodal",
            category=ToolCategory.RESEARCH,
            description=(
                "Unified search across ALL INTERNAL knowledge types: documents, code, tables, images, formulas. "
                "Use this for comprehensive research across the workspace when you need multiple content types at once. "
                "Searches uploaded/ingested content only — for external research, use web search tools."
            ),
            executor_class="MultimodalKnowledgeTools",
            executor_method="search_multimodal",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Research query that may span multiple content types",
                    required=True
                ),
                ToolParameter(
                    name="kb_types",
                    type="array",
                    description="Knowledge types to search (default: all types)",
                    required=False,
                    default=["document", "table", "image", "formula", "codegraph"]
                ),
                ToolParameter(
                    name="limit",
                    type="number",
                    description="Maximum total results across all types (default: 10)",
                    required=False,
                    default=10
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "search_multimodal", "params": {"query": "authentication system", "kb_types": ["document", "codegraph", "image"], "limit": 10}}
            ],
            metadata={"kb_types": ["document", "table", "image", "formula", "codegraph"], "added_in": "PRD-19"}
        ))
        
        # ==========================================
        # DATABASE TOOLS (NL-to-SQL)
        # ==========================================
        
        self.register_tool(ToolSpec(
            name="query_database",
            category=ToolCategory.DATABASE_TOOLS,
            description=(
                "Query CONNECTED databases using natural language. Converts your question to SQL and executes it "
                "against knowledge sources or the main Automatos database. Can generate charts via PandasAI. "
                "Use for structured data questions (counts, trends, metrics). For document content, use search_knowledge instead. "
                "For live internet data, use web search tools."
            ),
            executor_class="UnifiedToolExecutor",
            executor_method="_execute_database_tool",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Natural language query (e.g., 'Show failed workflows in the last 7 days', 'Count agents by type')",
                    required=True
                ),
                ToolParameter(
                    name="database_name",
                    type="string",
                    description="Specific database/knowledge source to query (optional - uses default if not specified)",
                    required=False
                ),
                ToolParameter(
                    name="analysis_prompt",
                    type="string",
                    description="Optional prompt for PandasAI to generate insights or charts from results",
                    required=False
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "query_database", "params": {"query": "Show failed workflows in the last 14 days"}},
                {"action": "query_database", "params": {"query": "How many agents are active?", "analysis_prompt": "Create a bar chart of agent types"}}
            ],
            metadata={"supports_pandas_ai": True, "added_in": "unified_tools"}
        ))
        
        # Smart Database Query (with intelligence features)
        self.register_tool(ToolSpec(
            name="smart_query_database",
            category=ToolCategory.DATABASE_TOOLS,
            description=(
                "Advanced database query with AI-powered clarification, rephrasing, explanation, and visualization. "
                "PREFER this over query_database for complex or ambiguous questions — it asks follow-ups and explains results. "
                "Use query_database for simple, direct queries. 2-attempt limit per turn — do NOT retry identical queries."
            ),
            executor_class="UnifiedToolExecutor",
            executor_method="_execute_smart_database_tool",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Natural language query (e.g., 'Show me sales trends', 'Compare revenue by region')",
                    required=True
                ),
                ToolParameter(
                    name="database_name",
                    type="string",
                    description="Specific database/knowledge source to query (optional)",
                    required=False
                ),
                ToolParameter(
                    name="skip_clarification",
                    type="boolean",
                    description="Skip clarification questions and attempt direct query (default: false)",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="clarification_answers",
                    type="object",
                    description="Answers to previous clarification questions (for multi-turn)",
                    required=False
                ),
                ToolParameter(
                    name="include_visualization",
                    type="boolean",
                    description="Include visualization suggestions (default: true)",
                    required=False,
                    default=True
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "smart_query_database", "params": {"query": "Show me sales trends"}},
                {"action": "smart_query_database", "params": {"query": "Compare performance", "skip_clarification": False}},
                {"action": "smart_query_database", "params": {"query": "Show me sales", "clarification_answers": {"time_period": "last 30 days", "group_by": "region"}}}
            ],
            metadata={
                "supports_pandas_ai": True,
                "supports_clarification": True,
                "supports_rephrasing": True,
                "supports_explanation": True,
                "supports_visualization": True,
                "added_in": "nl2sql_intelligence"
            }
        ))
        
        # ==========================================
        # FILE OPERATIONS — SUPERSEDED by workspace_* promoted actions
        # (workspace_read_file, workspace_write_file, workspace_list_dir, etc.)
        # Kept for backward compat with exec_file_ops.py proxy but
        # is_active=False so they don't appear in agent tool lists.
        # ==========================================

        self.register_tool(ToolSpec(
            name="read_file",
            category=ToolCategory.FILE_OPERATIONS,
            description=(
                "Read contents of a file from the agent workspace filesystem. "
                "Returns the raw file content. Use list_directory first if you need to find the file path. "
                "For searching across document CONTENT, use search_knowledge instead."
            ),
            executor_class="ActionExecutor",
            executor_method="read_file",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to read (relative to workspace)",
                    required=True
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8"
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            is_active=False,
            examples=[
                {"action": "read_file", "params": {"file_path": "config.json"}}
            ]
        ))

        self.register_tool(ToolSpec(
            name="write_file",
            category=ToolCategory.FILE_OPERATIONS,
            description=(
                "Write content to a file in the agent workspace (creates file if it doesn't exist, overwrites if it does). "
                "Use for saving reports, analysis outputs, code, or any deliverable. "
                "For polished documents (PDF/DOCX/XLSX), use generate_document instead."
            ),
            executor_class="ActionExecutor",
            executor_method="write_file",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to write (relative to workspace)",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8"
                )
            ],
            security_level=SecurityLevel.CAUTIOUS,
            permissions_required={"read": True, "write": True},
            is_active=False,
            examples=[
                {"action": "write_file", "params": {"file_path": "output.txt", "content": "Hello World"}}
            ]
        ))

        self.register_tool(ToolSpec(
            name="delete_file",
            category=ToolCategory.FILE_OPERATIONS,
            description=(
                "Permanently delete a file or directory from the agent workspace. Cannot be undone. "
                "Use only when explicitly asked to remove a file."
            ),
            executor_class="ActionExecutor",
            executor_method="delete_file",
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file or directory to delete",
                    required=True
                )
            ],
            security_level=SecurityLevel.DANGEROUS,
            permissions_required={"read": True, "write": True, "delete": True},
            is_active=False,
            examples=[
                {"action": "delete_file", "params": {"file_path": "temp.txt"}}
            ]
        ))

        self.register_tool(ToolSpec(
            name="list_directory",
            category=ToolCategory.FILE_OPERATIONS,
            description=(
                "List files and subdirectories in a workspace directory. "
                "Use to discover available files before reading them with read_file."
            ),
            executor_class="ActionExecutor",
            executor_method="list_directory",
            parameters=[
                ToolParameter(
                    name="dir_path",
                    type="string",
                    description="Path to the directory (default: current directory)",
                    required=False,
                    default="."
                )
            ],
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            is_active=False,
            examples=[
                {"action": "list_directory", "params": {"dir_path": "src"}}
            ]
        ))

        self.register_tool(ToolSpec(
            name="create_directory",
            category=ToolCategory.FILE_OPERATIONS,
            description="Create a new directory in the agent workspace. Creates parent directories if needed.",
            executor_class="ActionExecutor",
            executor_method="create_directory",
            parameters=[
                ToolParameter(
                    name="dir_path",
                    type="string",
                    description="Path to the directory to create",
                    required=True
                )
            ],
            security_level=SecurityLevel.CAUTIOUS,
            permissions_required={"read": True, "write": True},
            is_active=False,
            examples=[
                {"action": "create_directory", "params": {"dir_path": "new_folder"}}
            ]
        ))
        
        # ==========================================
        # SHELL COMMANDS (from ActionExecutor)
        # ==========================================
        
        self.register_tool(ToolSpec(
            name="execute_command",
            category=ToolCategory.SHELL_COMMANDS,
            description=(
                "Execute a shell command in the sandboxed agent workspace. "
                "RESTRICTED to whitelisted commands only: ls, cat, grep, find, git, python, npm, docker. "
                "Use for running scripts, git operations, or build commands. "
                "For remote server commands, use ssh_execute instead."
            ),
            executor_class="ActionExecutor",
            executor_method="execute_command",
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Shell command to execute (must be in whitelist)",
                    required=True
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Command timeout in seconds (default: 30)",
                    required=False,
                    default=30
                )
            ],
            security_level=SecurityLevel.DANGEROUS,
            permissions_required={"read": True, "write": True, "execute": True},
            examples=[
                {"action": "execute_command", "params": {"command": "ls -la", "timeout": 10}}
            ],
            metadata={
                "whitelisted_commands": [
                    "ls", "cat", "grep", "find", "git", "python", "npm", "docker"
                ],
                "warning": "Only whitelisted commands are allowed for security"
            }
        ))

        # ==========================================
        # HTTP REQUEST (Internal API Testing)
        # ==========================================

        self.register_tool(ToolSpec(
            name="http_request",
            category=ToolCategory.API_TOOLS,
            description=(
                f"Make HTTP requests to INTERNAL platform URLs only. "
                f"Use for testing API endpoints, checking health status, and verifying responses. "
                f"RESTRICTED to whitelisted domains: {config.INTERNAL_API_HOSTNAME}, {config.INTERNAL_FRONTEND_HOSTNAME}, "
                f"api.automatos.app, ui.automatos.app, localhost. "
                f"Will FAIL on external URLs. For external web requests, use web search tools or Composio actions."
            ),
            executor_class="UnifiedToolExecutor",
            executor_method="_execute_http_request",
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="Full URL to request (must be a whitelisted domain)",
                    required=True
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
                ),
                ToolParameter(
                    name="headers",
                    type="object",
                    description="Request headers as key-value pairs (e.g. {\"Content-Type\": \"application/json\"})",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="body",
                    type="object",
                    description="Request body (JSON object, sent as application/json for POST/PUT/PATCH)",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Request timeout in seconds (default: 30, max: 120)",
                    required=False,
                    default=30
                )
            ],
            returns="JSON object with status_code, headers, body, and duration_ms",
            security_level=SecurityLevel.CAUTIOUS,
            permissions_required={"read": True, "execute": True},
            examples=[
                {"action": "http_request", "params": {"url": f"http://{config.INTERNAL_API_HOSTNAME}/health", "method": "GET"}},
                {"action": "http_request", "params": {"url": f"http://{config.INTERNAL_API_HOSTNAME}/api/agents", "method": "GET", "headers": {"x-api-key": "your-key", "x-workspace-id": "your-ws-id"}}},
            ],
            metadata={
                "allowed_domains": [
                    config.INTERNAL_API_HOSTNAME,
                    config.INTERNAL_FRONTEND_HOSTNAME,
                    "api.automatos.app",
                    "ui.automatos.app",
                    "localhost",
                    "127.0.0.1",
                ],
                "max_response_bytes": 1_000_000,
                "added_in": "self-test-v1",
            }
        ))

        # ==========================================
        # SSH EXECUTION (Remote Commands)
        # ==========================================

        self.register_tool(ToolSpec(
            name="ssh_execute",
            category=ToolCategory.SSH_TOOLS,
            description=(
                "Execute commands on a REMOTE server via SSH. "
                "Use for cloning repositories, running tests, deploying, or checking remote server status. "
                "Requires SSH credentials (host, username, and either password or private key). "
                "For LOCAL workspace commands, use execute_command instead. "
                "Commands run in a non-interactive shell. Timeout default: 60s, max: 300s."
            ),
            executor_class="UnifiedToolExecutor",
            executor_method="_execute_ssh",
            parameters=[
                ToolParameter(
                    name="host",
                    type="string",
                    description="SSH server hostname or IP address",
                    required=True
                ),
                ToolParameter(
                    name="command",
                    type="string",
                    description="Shell command to execute on the remote server",
                    required=True
                ),
                ToolParameter(
                    name="username",
                    type="string",
                    description="SSH username (default: 'automatos')",
                    required=False,
                    default="automatos"
                ),
                ToolParameter(
                    name="port",
                    type="number",
                    description="SSH port (default: 22)",
                    required=False,
                    default=22
                ),
                ToolParameter(
                    name="password",
                    type="string",
                    description="SSH password (use credential_id instead when possible)",
                    required=False
                ),
                ToolParameter(
                    name="private_key",
                    type="string",
                    description="SSH private key content (PEM format)",
                    required=False
                ),
                ToolParameter(
                    name="credential_id",
                    type="string",
                    description="ID of stored SSH credential to use (preferred over inline password/key)",
                    required=False
                ),
                ToolParameter(
                    name="timeout",
                    type="number",
                    description="Command timeout in seconds (default: 60, max: 300)",
                    required=False,
                    default=60
                )
            ],
            returns="JSON object with exit_code, stdout, stderr, and duration_ms",
            security_level=SecurityLevel.DANGEROUS,
            permissions_required={"read": True, "write": True, "execute": True},
            examples=[
                {"action": "ssh_execute", "params": {"host": "build-server.railway.internal", "command": "cd /app && python -m pytest tests/ -v --tb=short", "username": "automatos", "timeout": 120}},
                {"action": "ssh_execute", "params": {"host": "build-server.railway.internal", "command": "git clone https://github.com/AutomatosAI/automatos-ai.git /tmp/repo", "credential_id": "ssh-prod-1"}},
            ],
            metadata={
                "requires_credentials": True,
                "max_output_bytes": 500_000,
                "added_in": "self-test-v1",
            }
        ))

        # ==========================================
        # COMPOSIO EXECUTION (External Apps)
        # ==========================================

        self.register_tool(
            ToolSpec(
                name="composio_execute",
                category=ToolCategory.API_TOOLS,
                description=(
                    "Execute any action across 1000+ external app integrations via Composio — "
                    "web search, email, messaging, GitHub, CRM, calendars, databases, and more. "
                    "If your tool list includes per-action tools (with typed parameters), prefer those "
                    "for accuracy. Use composio_execute for any action not already in your tool list. "
                    "Check platform_list_connected_apps to see what's available. "
                    "IMPORTANT: action-specific parameters MUST go inside the `params` object, NOT at the top level."
                ),
                executor_class="ComposioToolExecutor",
                executor_method="execute",
                parameters=[
                    ToolParameter(
                        name="app_name",
                        type="string",
                        description="App name (e.g., 'GMAIL', 'SLACK', 'GITHUB')",
                        required=False,
                    ),
                    ToolParameter(
                        name="action",
                        type="string",
                        description="Action name (e.g., 'GMAIL_LIST_EMAILS', 'SLACK_SEND_MESSAGE')",
                        required=True,
                    ),
                    ToolParameter(
                        name="params",
                        type="object",
                        description=(
                            "Action-specific parameters as a JSON object. "
                            "All fields required by the action (e.g. issue_key, channel, text) "
                            "MUST be placed inside this params object."
                        ),
                        required=False,
                        default={},
                    ),
                ],
                security_level=SecurityLevel.CAUTIOUS,
                permissions_required={"read": True, "execute": True},
                examples=[
                    {
                        "action": "composio_execute",
                        "params": {
                            "action": "SLACK_SEND_MESSAGE",
                            "params": {"channel": "#general", "text": "Hello"}
                        },
                    },
                ],
                metadata={"integration_type": "composio"},
            )
        )

        # ==========================================
        # DOCUMENT GENERATION (PRD-63)
        # ==========================================

        self.register_tool(ToolSpec(
            name="generate_document",
            category=ToolCategory.FILE_OPERATIONS,
            description=(
                "Generate a polished PDF, DOCX, or XLSX document for download. "
                "PREFER this over write_file when the user needs a formatted, downloadable document. "
                "WILL produce a BLANK document if you omit the 'data' parameter — you MUST include "
                "actual written content. For PDFs, include 'sections' with full paragraphs. "
                "Returns a download URL."
            ),
            executor_class="AgentPlatformTools",
            executor_method="execute_tool",
            parameters=[
                ToolParameter(
                    name="title",
                    type="string",
                    description="Document title (e.g., 'Monthly Sales Report', 'Invoice #1234')",
                    required=True
                ),
                ToolParameter(
                    name="format",
                    type="string",
                    description="Output format: pdf, docx, or xlsx",
                    required=True,
                    enum=["pdf", "docx", "xlsx"]
                ),
                ToolParameter(
                    name="data",
                    type="object",
                    description=(
                        "REQUIRED — the actual document content. Without this the document will be blank. "
                        "For PDF/DOCX reports: {\"sections\": [{\"title\": \"Section Name\", \"content\": \"Write full "
                        "paragraphs of text here — this is the body of the document.\"}], "
                        "\"author\": \"...\", \"date\": \"...\"}. "
                        "You MUST write out the full text content for each section — do not leave sections empty. "
                        "For tables/xlsx: {\"columns\": [\"col1\", \"col2\"], \"rows\": [[\"val1\", \"val2\"]]}."
                    ),
                    required=True
                ),
                ToolParameter(
                    name="template_name",
                    type="string",
                    description="Template to use (e.g., 'Basic Report', 'Invoice'). Omit for auto-selection.",
                    required=False
                ),
                # P2-09 S2 (F031/J2): the handler already parses/validates template_id,
                # but only the chatbot's inline schema declared it — so id-driven
                # template generation worked in chat and nowhere else. Declaring it
                # here gives the non-chat autonomy lane (missions/board/scheduled)
                # parity, and makes platform_get_template_schema's "use before
                # generate_document" discovery flow followable. Wording mirrors the
                # inline chat schema (agent_platform_tools.get_available_tools).
                ToolParameter(
                    name="template_id",
                    type="string",
                    description="UUID of a specific template to fill (from platform_list_templates). Takes precedence over template_name.",
                    required=False
                ),
            ],
            returns="JSON with filename, format, download_url, and size_kb",
            security_level=SecurityLevel.CAUTIOUS,
            permissions_required={"read": True, "write": True},
            examples=[
                {
                    "action": "generate_document",
                    "params": {
                        "title": "Weekly Status Report",
                        "format": "pdf",
                        "data": {"sections": [{"title": "Summary", "content": "All tasks completed on time. The team delivered 5 features and resolved 12 bugs."}]},
                    },
                },
                {
                    "action": "generate_document",
                    "params": {
                        "title": "User Export",
                        "format": "xlsx",
                        "data": {"rows": [{"name": "Alice", "email": "alice@example.com"}]},
                    },
                },
            ],
            metadata={"added_in": "PRD-63"}
        ))

        # ==========================================
        # WIDGET UI AFFORDANCES (PRD-008-A.2)
        # ==========================================
        # Replaces the deprecated server-side keyword matcher: the LLM
        # now decides when to open the inline callback form, calling this
        # tool with the active product context (if any). Access is gated
        # by the workspace's default-Site `settings.callback.enabled` flag
        # in ``validate_tool_access`` below so the tool is invisible to
        # the LLM on stores that haven't opted in.

        from modules.tools.widget_callback import WIDGET_OPEN_CALLBACK_FORM_NAME

        self.register_tool(ToolSpec(
            name=WIDGET_OPEN_CALLBACK_FORM_NAME,
            category=ToolCategory.WIDGET,
            description=(
                "Open the inline phone-callback form in the shopper's chat "
                "panel. Use when the shopper asks for a callback, a phone "
                "number, to speak with a human, or otherwise indicates they "
                "want voice / phone contact rather than chat. The form "
                "appears as a bubble in the chat; the shopper fills in name "
                "and number and the merchant receives the lead. After "
                "calling this tool, briefly confirm: 'I've opened a quick "
                "form — just need your name and number.' Do NOT suggest "
                "email or any other contact method."
            ),
            executor_class="UnifiedToolExecutor",
            executor_method="_execute_widget_callback",
            parameters=[
                ToolParameter(
                    name="product_context",
                    type="string",
                    description=(
                        "Optional product the shopper is asking about — used "
                        "to label the form (e.g. 'EN 12101-9 Control Panel'). "
                        "Use the product title when one is in the conversation "
                        "or page context; omit for general callback requests."
                    ),
                    required=False,
                ),
            ],
            returns="Confirmation that the form has been opened in the shopper's chat panel",
            security_level=SecurityLevel.SAFE,
            permissions_required={"read": True},
            examples=[
                {"action": "widget_open_callback_form", "params": {}},
                {"action": "widget_open_callback_form", "params": {"product_context": "EN 12101-9 Control Panel"}},
            ],
            metadata={"added_in": "PRD-008-A.2"},
        ))

    def _extract_methods(self, capabilities: Dict[str, Any]) -> List[str]:
            
        # Try different keys
        methods = (
            capabilities.get('methods') or 
            capabilities.get('tools') or 
            capabilities.get('actions') or 
            []
        )
        
        # Handle dict (use keys)
        if isinstance(methods, dict):
            return list(methods.keys())
            
        # Handle list
        if isinstance(methods, list):
            return [str(m) for m in methods]
            
        return []
    
    def validate_tool_access(
        self,
        agent_id: int,
        tool_name: str,
        db: Optional[Session] = None,
        workspace_id: Optional[Any] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if an agent has access to a tool.
        
        Args:
            agent_id: ID of the agent
            tool_name: Name of the tool
            db: Database session
            workspace_id: Optional workspace context for granular checks
        
        Returns:
            Tuple of (has_access, error_message)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found in registry"
        
        if not tool.is_active:
            return False, f"Tool '{tool_name}' is not active"

        # -----------------------------------------------------------------
        # PRD-123 Pattern #4: Tool Tier Policy Enforcement
        # system  → always available (skip all further checks)
        # platform → available by default (skip assignment checks)
        # marketplace/custom → require explicit assignment (existing logic)
        # -----------------------------------------------------------------
        tool_tier = getattr(tool, "tier", None) or "marketplace"
        # Resolve from DB model if the registry tool doesn't carry tier
        if tool_tier == "marketplace" and db:
            try:
                from core.models.tools import Tool as ToolModel
                db_tool = db.query(ToolModel).filter(ToolModel.name == tool_name).first()
                if db_tool and db_tool.tier:
                    tool_tier = db_tool.tier
            except Exception:
                pass  # fall through to default marketplace behavior

        if tool_tier == "system":
            return True, None  # system tools always pass

        if tool_tier == "platform":
            # Platform tools available by default — only blocked if workspace explicitly disables
            # (workspace-level disable not yet implemented — allow for now)
            return True, None

        # marketplace and custom tiers continue to existing assignment checks below

        # ---------------------------------------------------------------------
        # Widget UI tools: only exposed when the workspace's default Site has
        # the matching feature enabled. The Site config is the source of truth
        # (merchant toggles it on/off). When no DB session or no workspace
        # context is available we fail closed — better to hide the tool than
        # leak a UI affordance the merchant disabled.
        # ---------------------------------------------------------------------
        from modules.tools.widget_callback import WIDGET_OPEN_CALLBACK_FORM_NAME

        if tool_name == WIDGET_OPEN_CALLBACK_FORM_NAME:
            if not db or not workspace_id:
                return False, "Widget callback tool requires workspace context"
            try:
                from services.sites import get_default_site
                import uuid as _uuid

                ws_uuid = workspace_id if isinstance(workspace_id, _uuid.UUID) else _uuid.UUID(str(workspace_id))
                site = get_default_site(db, ws_uuid)
            except Exception as exc:
                return False, f"Failed to resolve site for widget callback: {exc}"

            if site is None:
                return False, "No Site configured for this workspace"

            callback_settings = (getattr(site, "settings", None) or {}).get("callback") or {}
            if not callback_settings.get("enabled"):
                return False, "Callback feature is disabled for this Site"
            # Enabled — fall through to the standard checks below.

        # ---------------------------------------------------------------------
        # Hard gating for Composio execution:
        # Only expose Composio tools when the agent has EXTERNAL app assignments
        # AND those apps are connected for the workspace.
        # ---------------------------------------------------------------------
        if tool_name == "composio_execute":
            if not db:
                return False, "Database session required to validate Composio tool access"
            if not workspace_id:
                return False, "Workspace context required for Composio tool access"

            try:
                from core.models.composio_cache import AgentAppAssignment
                from core.composio.entity_manager import EntityManager

                assigned_rows = (
                    db.query(AgentAppAssignment)
                    .filter(
                        AgentAppAssignment.agent_id == agent_id,
                        AgentAppAssignment.is_active == True,  # noqa: E712
                        AgentAppAssignment.app_type == "EXTERNAL",
                    )
                    .all()
                )
                assigned_apps = {str(r.app_name or "").upper().strip() for r in assigned_rows if r and r.app_name}
                assigned_apps.discard("")

                connected_apps = set()
                try:
                    manager = EntityManager(db)
                    connected_apps = {a.upper().strip() for a in manager.get_connected_apps(workspace_id)}
                except Exception:
                    connected_apps = set()

                # Auto-inherit: when agent has no explicit assignments, use all
                # workspace-connected apps. 850+ tools — agents shouldn't need
                # manual per-app assignment to use what's already connected.
                if not assigned_apps:
                    if connected_apps:
                        logger.info(
                            f"[ToolRegistry] Agent {agent_id} has no app assignments — "
                            f"inheriting {len(connected_apps)} workspace apps: {connected_apps}"
                        )
                        assigned_apps = connected_apps
                    else:
                        return False, "No external apps are connected for this workspace"

                eligible = assigned_apps.intersection(connected_apps) if connected_apps else assigned_apps
                if not eligible:
                    return False, "No assigned external apps are connected for this workspace"
            except Exception as exc:
                return False, f"Failed to validate Composio access: {exc}"

        # Context filtering to prevent tool bypass.
        if db:
            from core.models import Agent
            agent = db.query(Agent).get(agent_id)
            
            # Default to 'general' context
            active_context = "general"
            if agent and agent.configuration:
                 active_context = agent.configuration.get("active_context", "general")
            
            # Force Chatbot (Agent 1) to 'general'
            if agent_id == 1 and active_context == "all":
                active_context = "general"
            
            # Context Map (TODO: make dynamic when tool taxonomy stabilizes)
            CONTEXT_MAP = {
                "general": ["communication", "research", "productivity", "system", "collaboration", "developer", "api", "file_ops", "database"],
                "coding": ["developer", "github", "git", "code", "file_ops", "devtools", "research"],
                "ops": ["cloud", "k8s", "aws", "infrastructure", "monitoring", "database", "shell"],
                "communication": ["communication", "slack", "email", "chat", "collaboration"],
                "research": ["research", "data", "search", "rag", "database"],
            }
            
            allowed_categories = CONTEXT_MAP.get(active_context)
            
            if allowed_categories:
                # Resolve category
                tool_cat = tool.category.value if hasattr(tool.category, 'value') else str(tool.category)
                
                # Check cache for category if available (for adapters)
                if tool.metadata and tool.metadata.get("category"):
                    tool_cat = tool.metadata.get("category").lower()
                
                # Exception for System Tools
                is_system_tool = tool_name in [
                    "switch_context",
                    "search_knowledge",
                    WIDGET_OPEN_CALLBACK_FORM_NAME,
                ]
                
                if not is_system_tool and tool_cat not in allowed_categories:
                    self.logger.debug(f"ℹ️ ToolRegistry: Filtering out {tool_name} (category: {tool_cat}) - not in context '{active_context}'")
                    return False, f"Tool category '{tool_cat}' not allowed in current context '{active_context}'"

        # All other tools allowed by default (security handled by executors)
        return True, None


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None
_registry_init_logged: bool = False


def get_tool_registry(db_session: Optional[Session] = None) -> ToolRegistry:
    """
    Get or create the global tool registry instance.
    
    Args:
        db_session: Optional database session
    
    Returns:
        ToolRegistry instance
    """
    global _tool_registry, _registry_init_logged
    
    if _tool_registry is None:
        _tool_registry = ToolRegistry(db_session=db_session)
        if not _registry_init_logged:
            logger.info("Tool registry initialized (singleton)")
            _registry_init_logged = True
    
    return _tool_registry


def reset_tool_registry():
    """Reset the global registry (for testing)."""
    global _tool_registry, _registry_init_logged
    _tool_registry = None
    _registry_init_logged = False

