"""
Tool Registry - Centralized Tool Management System
===================================================
PRD-17: Dynamic Tool Assignment & Centralized Tool Management

Provides a single source of truth for ALL platform tools, accessible to:
- Orchestrator (task decomposition with tool recommendations)
- Agent Factory (dynamic tool injection)
- ChatBot (tool-augmented responses)
- User/API (tool discovery & execution)
- Future integrations (plugins, extensions)

Design Principles:
- Non-breaking: Wraps existing tool systems without modifying them
- Unified: Single interface for all tool types
- Discoverable: Tools are queryable by category, task type, or name
- Secure: Maintains existing security controls
- Extensible: Easy to add new tools or categories
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and filtering"""
    RESEARCH = "research"              # RAG, semantic search, CodeGraph
    FILE_OPERATIONS = "file_ops"       # read, write, delete files
    SHELL_COMMANDS = "shell"           # execute shell commands
    MCP_TOOLS = "mcp"                  # Third-party MCP integrations
    DATABASE_TOOLS = "database"        # SQL operations (future)
    SSH_TOOLS = "ssh"                  # SSH operations (future)
    API_TOOLS = "api"                  # REST API calls (future)
    GIT_OPERATIONS = "git"             # Git operations (future)


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
    - Register tools from all executors (platform, action, MCP)
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
        
        # Load MCP tools from database if session provided
        if self.db:
            self._register_mcp_tools()
        
        self.logger.info(f"ToolRegistry initialized with {len(self.tools)} tools")
    
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
        
        self.logger.info(f"Registered tool: {tool.name} (category: {tool.category.value}, security: {tool.security_level.value})")
    
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
        from services.tool_capability_mapper import ToolCapabilityMapper
        
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
            description="Search the Automatos knowledge base for documentation, guides, and information about the platform",
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
            description="Find semantically similar content across all platform documents using vector embeddings",
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
            description="Search indexed codebases for functions, classes, and implementations",
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
            description="Search for tables and structured data extracted from documents. Returns tables with preserved structure in Markdown, CSV, and JSON formats",
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
            description="Search for images, diagrams, and charts with AI-generated descriptions and OCR text. Useful for finding architecture diagrams, flowcharts, screenshots",
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
            description="Search for mathematical formulas and equations. Returns LaTeX format with variable and operator extraction",
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
            description="Unified search across ALL knowledge types: documents, code, tables, images, formulas. Use this for comprehensive research when you need multiple content types",
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
        # FILE OPERATIONS (from ActionExecutor)
        # ==========================================
        
        self.register_tool(ToolSpec(
            name="read_file",
            category=ToolCategory.FILE_OPERATIONS,
            description="Read contents of a file from the workspace",
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
            examples=[
                {"action": "read_file", "params": {"file_path": "config.json"}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="write_file",
            category=ToolCategory.FILE_OPERATIONS,
            description="Write content to a file in the workspace (creates file if it doesn't exist)",
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
            examples=[
                {"action": "write_file", "params": {"file_path": "output.txt", "content": "Hello World"}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="delete_file",
            category=ToolCategory.FILE_OPERATIONS,
            description="Delete a file or directory from the workspace",
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
            examples=[
                {"action": "delete_file", "params": {"file_path": "temp.txt"}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="list_directory",
            category=ToolCategory.FILE_OPERATIONS,
            description="List contents of a directory in the workspace",
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
            examples=[
                {"action": "list_directory", "params": {"dir_path": "src"}}
            ]
        ))
        
        self.register_tool(ToolSpec(
            name="create_directory",
            category=ToolCategory.FILE_OPERATIONS,
            description="Create a new directory in the workspace",
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
            description="Execute a shell command in the sandboxed workspace (whitelisted commands only)",
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
    
    def _register_mcp_tools(self):
        """Register MCP tools from database - only load configured tools"""
        try:
            from models import MCPTool
            
            # PRD-17: Only load tools that are active AND properly configured
            mcp_tools = self.db.query(MCPTool).filter(
                MCPTool.status == 'active',
                MCPTool.mcp_server_url.isnot(None),  # Must have server URL
                MCPTool.credentials_schema.isnot(None)  # Must have credentials schema
            ).all()
            
            for mcp_tool in mcp_tools:
                # Extract methods from capabilities
                capabilities = mcp_tool.capabilities or {}
                methods = capabilities.get('methods', [])
                
                # Register each method as a separate tool
                for method in methods:
                    tool_name = f"mcp_{mcp_tool.name.lower().replace(' ', '_')}_{method.replace('.', '_')}"
                    
                    self.register_tool(ToolSpec(
                        name=tool_name,
                        category=ToolCategory.MCP_TOOLS,
                        description=f"{mcp_tool.description} - Method: {method}",
                        executor_class="MCPToolExecutor",
                        executor_method="execute_tool",
                        parameters=[
                            ToolParameter(
                                name="params",
                                type="object",
                                description=f"Parameters for {method}",
                                required=False,
                                default={}
                            )
                        ],
                        security_level=SecurityLevel.CAUTIOUS,
                        permissions_required={"read": True, "execute": True},
                        metadata={
                            "tool_id": mcp_tool.id,
                            "provider": mcp_tool.provider,
                            "method": method,
                            "mcp_server_url": mcp_tool.mcp_server_url
                        }
                    ))
            
            self.logger.info(f"Registered {len(mcp_tools)} MCP tools from database")
            
        except Exception as e:
            self.logger.warning(f"Could not register MCP tools from database: {e}")
    
    def validate_tool_access(
        self,
        agent_id: int,
        tool_name: str,
        db: Optional[Session] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if an agent has access to a tool.
        
        Args:
            agent_id: ID of the agent
            tool_name: Name of the tool
            db: Database session
        
        Returns:
            Tuple of (has_access, error_message)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found in registry"
        
        if not tool.is_active:
            return False, f"Tool '{tool_name}' is not active"
        
        # For MCP tools, check database permissions
        if tool.category == ToolCategory.MCP_TOOLS and db:
            try:
                from models import AgentToolAssignment
                
                tool_id = tool.metadata.get("tool_id")
                if not tool_id:
                    return True, None  # No specific tool_id, allow access
                
                assignment = db.query(AgentToolAssignment).filter(
                    AgentToolAssignment.agent_id == agent_id,
                    AgentToolAssignment.tool_id == tool_id,
                    AgentToolAssignment.enabled == True
                ).first()
                
                if not assignment:
                    return False, f"Agent {agent_id} does not have access to MCP tool {tool_name}"
                
            except Exception as e:
                self.logger.error(f"Error checking MCP tool permissions: {e}")
                return False, str(e)
        
        # All other tools allowed by default (security handled by executors)
        return True, None


# Global registry instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry(db_session: Optional[Session] = None) -> ToolRegistry:
    """
    Get or create the global tool registry instance.
    
    Args:
        db_session: Optional database session for MCP tool loading
    
    Returns:
        ToolRegistry instance
    """
    global _tool_registry
    
    if _tool_registry is None:
        _tool_registry = ToolRegistry(db_session=db_session)
    elif db_session and not _tool_registry.db:
        # Update with database session if not already set
        _tool_registry.db = db_session
        _tool_registry._register_mcp_tools()
    
    return _tool_registry


def reset_tool_registry():
    """Reset the global registry (for testing)"""
    global _tool_registry
    _tool_registry = None

