"""
Database Tool Integration Service
==================================
PRD-21 + PRD-17: Auto-creates agent tools when database sources are added

This integrates with your Dynamic Tool Assignment system to:
1. Auto-create database query tools when sources are added
2. Register tools with ToolRegistry
3. Make tools available to agents based on task type
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from modules.tools import ToolRegistry, ToolCategory, SecurityLevel
from modules.tools.registry.tool_registry import ToolSpec, ToolParameter
from modules.tools.services.tool_capability_mapper import ToolCapabilityMapper
from core.models.database_knowledge import DatabaseKnowledgeSource

logger = logging.getLogger(__name__)


class DatabaseToolIntegration:
    """
    Bridges Database Knowledge Sources with Dynamic Tool Assignment.
    When a database is added, automatically creates tools that agents can use.
    """
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        capability_mapper: Optional[ToolCapabilityMapper] = None
    ):
        """Initialize with existing tool systems from PRD-17."""
        self.tool_registry = tool_registry or ToolRegistry()
        self.capability_mapper = capability_mapper or ToolCapabilityMapper()
        
        # Register database as a tool category if not exists
        if ToolCategory.DATABASE_TOOLS not in [cat for cat in ToolCategory]:
            logger.info("Database tools category already registered")
    
    def create_database_tools(
        self,
        source: DatabaseKnowledgeSource
    ) -> List[Dict[str, Any]]:
        """
        Create tools for a database source.
        Returns list of created tool definitions.
        """
        tools_created = []
        
        # Tool 1: Natural language query tool
        query_tool = self._create_query_tool(source)
        tools_created.append(query_tool)
        
        # Tool 2: Schema exploration tool  
        schema_tool = self._create_schema_tool(source)
        tools_created.append(schema_tool)
        
        # Tool 3: Template execution tool
        template_tool = self._create_template_tool(source)
        tools_created.append(template_tool)
        
        # Register all tools with ToolRegistry
        for tool in tools_created:
            self._register_tool(tool, source)
        
        # Update task mappings for database-related tasks
        self._update_task_mappings(source)
        
        logger.info(f"Created {len(tools_created)} tools for database '{source.name}'")
        return tools_created
    
    def _create_query_tool(self, source: DatabaseKnowledgeSource) -> Dict[str, Any]:
        """
        Create natural language query tool for the database.
        """
        tool_name = f"query_{self._sanitize_name(source.name)}_database"
        
        return {
            "name": tool_name,
            "display_name": f"Query {source.name} Database",
            "description": f"Query {source.name} database using natural language. "
                          f"Converts questions to SQL and returns results.",
            "category": ToolCategory.DATABASE_TOOLS,
            "security_level": SecurityLevel.CAUTIOUS,  # Read-only but still needs care
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'Show top 10 customers by revenue')",
                    "required": True
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return",
                    "default": 100,
                    "minimum": 1,
                    "maximum": source.max_rows_limit
                },
                "visualization_hint": {
                    "type": "string",
                    "description": "Preferred visualization type",
                    "enum": ["table", "bar", "line", "pie", "scatter"],
                    "default": "table"
                },
                "use_cache": {
                    "type": "boolean",
                    "description": "Use cached results if available",
                    "default": True
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "array"},
                    "columns": {"type": "array"},
                    "row_count": {"type": "integer"},
                    "sql": {"type": "string"},
                    "explanation": {"type": "string"},
                    "visualization": {"type": "object"},
                    "cached": {"type": "boolean"}
                }
            },
            "examples": [
                {
                    "input": {"query": "Show revenue trend for last 6 months"},
                    "output": "Returns monthly revenue data with line chart config"
                },
                {
                    "input": {"query": "Top 5 products by sales"},
                    "output": "Returns product sales data with bar chart config"
                }
            ],
            "metadata": {
                "source_id": source.id,
                "dialect": source.dialect,
                "requires_credential": source.credential_id,
                "workspace_id": source.workspace_id,
                "cache_ttl": source.query_cache_ttl
            }
        }
    
    def _create_schema_tool(self, source: DatabaseKnowledgeSource) -> Dict[str, Any]:
        """
        Create schema exploration tool.
        """
        tool_name = f"explore_{self._sanitize_name(source.name)}_schema"
        
        return {
            "name": tool_name,
            "display_name": f"Explore {source.name} Schema",
            "description": f"Explore schema of {source.name} database. "
                          f"Get tables, columns, relationships, and semantic definitions.",
            "category": ToolCategory.DATABASE_TOOLS,
            "security_level": SecurityLevel.SAFE,  # Just reading metadata
            "parameters": {
                "table_name": {
                    "type": "string",
                    "description": "Optional: specific table to explore",
                    "required": False
                },
                "include_relationships": {
                    "type": "boolean",
                    "description": "Include foreign key relationships",
                    "default": True
                },
                "include_metrics": {
                    "type": "boolean",
                    "description": "Include semantic layer metrics",
                    "default": True
                },
                "include_dimensions": {
                    "type": "boolean",
                    "description": "Include semantic layer dimensions",
                    "default": True
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "tables": {"type": "array"},
                    "relationships": {"type": "array"},
                    "metrics": {"type": "array"},
                    "dimensions": {"type": "array"},
                    "total_tables": {"type": "integer"},
                    "dialect": {"type": "string"}
                }
            },
            "metadata": {
                "source_id": source.id,
                "workspace_id": source.workspace_id,
                "cache_ttl": source.schema_cache_ttl
            }
        }
    
    def _create_template_tool(self, source: DatabaseKnowledgeSource) -> Dict[str, Any]:
        """
        Create template execution tool.
        """
        tool_name = f"run_{self._sanitize_name(source.name)}_template"
        
        return {
            "name": tool_name,
            "display_name": f"Run {source.name} Query Template",
            "description": f"Execute pre-built query templates for {source.name} database. "
                          f"Templates are tested and optimized queries for common use cases.",
            "category": ToolCategory.DATABASE_TOOLS,
            "security_level": SecurityLevel.SAFE,  # Templates are pre-validated
            "parameters": {
                "template_name": {
                    "type": "string",
                    "description": "Name of template to execute",
                    "required": True
                },
                "parameters": {
                    "type": "object",
                    "description": "Template parameters (varies by template)",
                    "default": {}
                }
            },
            "output": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "array"},
                    "template_used": {"type": "string"},
                    "visualization": {"type": "object"}
                }
            },
            "metadata": {
                "source_id": source.id,
                "workspace_id": source.workspace_id,
                "dialect": source.dialect
            }
        }
    
    def _register_tool(self, tool_def: Dict[str, Any], source: DatabaseKnowledgeSource) -> None:
        """
        Register tool with the ToolRegistry from PRD-17.
        """
        try:
            # Convert parameters dict to ToolParameter list
            params = []
            for param_name, param_def in tool_def.get("parameters", {}).items():
                params.append(ToolParameter(
                    name=param_name,
                    type=param_def.get("type", "string"),
                    description=param_def.get("description", ""),
                    required=param_def.get("required", False),
                    default=param_def.get("default"),
                    enum=param_def.get("enum")
                ))
            
            # Create ToolSpec
            tool_spec = ToolSpec(
                name=tool_def["name"],
                category=tool_def["category"],
                description=tool_def.get("description", ""),
                executor_class="DatabaseToolIntegration",
                executor_method=f"execute_{tool_def['name']}",
                parameters=params,
                security_level=tool_def.get("security_level", SecurityLevel.SAFE),
                metadata=tool_def.get("metadata", {})
            )
            
            # Add to ToolRegistry
            self.tool_registry.register_tool(tool_spec)
            
            logger.debug(f"Registered tool '{tool_def['name']}' with ToolRegistry")
            
        except Exception as e:
            logger.error(f"Failed to register tool '{tool_def['name']}': {e}")
    
    def _create_tool_executor(
        self,
        tool_def: Dict[str, Any],
        source: DatabaseKnowledgeSource
    ) -> callable:
        """
        Create the actual executor function for the tool.
        This will be called when agents use the tool.
        """
        async def executor(**kwargs):
            """Execute database tool with provided parameters."""
            from modules.nl2sql import DatabaseKnowledgeService
            
            # Get service instance
            service = DatabaseKnowledgeService()
            
            # Route to appropriate method based on tool name
            if "query_" in tool_def["name"]:
                # Natural language query
                return await service.query_database(
                    source_id=source.id,
                    natural_language_query=kwargs.get("query"),
                    user_id=kwargs.get("_user_id"),
                    agent_id=kwargs.get("_agent_id")
                )
                
            elif "explore_" in tool_def["name"]:
                # Schema exploration
                return await service.get_schema_info(
                    source_id=source.id,
                    table_name=kwargs.get("table_name"),
                    include_relationships=kwargs.get("include_relationships", True),
                    include_metrics=kwargs.get("include_metrics", True),
                    include_dimensions=kwargs.get("include_dimensions", True)
                )
                
            elif "run_" in tool_def["name"] and "template" in tool_def["name"]:
                # Template execution
                return await service.execute_template(
                    source_id=source.id,
                    template_name=kwargs.get("template_name"),
                    parameters=kwargs.get("parameters", {})
                )
            
            else:
                raise ValueError(f"Unknown tool type: {tool_def['name']}")
        
        return executor
    
    def _update_task_mappings(self, source: DatabaseKnowledgeSource) -> None:
        """
        Update task-to-tool mappings for database-related tasks.
        Ensures agents get database tools when needed.
        """
        database_tool_names = [
            f"query_{self._sanitize_name(source.name)}_database",
            f"explore_{self._sanitize_name(source.name)}_schema",
            f"run_{self._sanitize_name(source.name)}_template"
        ]
        
        # Task types that should get these database tools
        database_task_types = [
            "data_analysis",
            "reporting", 
            "database_query",
            "business_intelligence",
            "analytics",
            "data_exploration",
            "metrics_calculation",
            "performance_monitoring"
        ]
        
        # Update mappings in ToolCapabilityMapper
        # Uses the correct signature: task_type, required_categories, optional_categories, specific_tools, confidence
        for task_type in database_task_types:
            try:
                self.capability_mapper.add_custom_mapping(
                    task_type=task_type,
                    required_categories=["database"],
                    optional_categories=None,
                    specific_tools=database_tool_names,
                    confidence=0.9
                )
            except Exception as e:
                logger.warning(f"Could not add task mapping for {task_type}: {e}")
        
        logger.info(f"Updated task mappings for {len(database_task_types)} task types")
    
    def get_mcp_tool_definitions(self, source: DatabaseKnowledgeSource) -> List[Dict[str, Any]]:
        """
        PRD-61 Phase 7 (US-016): Get MCP-format tool definitions for NL2SQL.
        Exposes query, schema exploration, and training as MCP tools for
        external agents (Claude Desktop, Cursor, etc.).
        """
        source_name = source.name
        source_id = source.id

        return [
            {
                "name": "query_database",
                "description": (
                    f"Ask a natural language question about data in the "
                    f"'{source_name}' database. Returns SQL, results, and "
                    f"optional visualization."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the data"
                        },
                        "database_source": {
                            "type": "string",
                            "description": f"Name or ID of the database source (default: {source_name})",
                            "default": str(source_id)
                        },
                        "auto_execute": {
                            "type": "boolean",
                            "description": "Execute the query automatically",
                            "default": True
                        },
                        "include_chart": {
                            "type": "boolean",
                            "description": "Include visualization suggestion",
                            "default": False
                        }
                    },
                    "required": ["question"]
                },
                "metadata": {
                    "source_id": source_id,
                    "workspace_id": str(source.workspace_id),
                    "read_only": True,
                    "max_rows": source.max_rows_limit or 1000,
                }
            },
            {
                "name": "explore_database_schema",
                "description": (
                    f"Explore the schema of '{source_name}' database. "
                    f"Returns tables, columns, relationships, and sample data."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_source": {
                            "type": "string",
                            "default": str(source_id)
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Optional: specific table to explore"
                        }
                    }
                },
                "metadata": {
                    "source_id": source_id,
                    "workspace_id": str(source.workspace_id),
                }
            },
            {
                "name": "add_sql_training_example",
                "description": (
                    "Add a verified question/SQL pair to improve future "
                    "query generation accuracy."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The natural language question"
                        },
                        "sql": {
                            "type": "string",
                            "description": "The correct SQL query"
                        },
                        "database_source": {
                            "type": "string",
                            "default": str(source_id)
                        }
                    },
                    "required": ["question", "sql"]
                },
                "metadata": {
                    "source_id": source_id,
                    "workspace_id": str(source.workspace_id),
                }
            }
        ]

    def remove_database_tools(self, source_id: int, source_name: str) -> int:
        """
        Remove tools when a database source is deleted.
        Note: ToolRegistry doesn't support unregister yet - tools will be orphaned
        but won't cause issues since source is gone.
        """
        tool_names = [
            f"query_{self._sanitize_name(source_name)}_database",
            f"explore_{self._sanitize_name(source_name)}_schema",
            f"run_{self._sanitize_name(source_name)}_template"
        ]
        
        removed = 0
        for tool_name in tool_names:
            # Remove from registry if it exists
            if tool_name in self.tool_registry.tools:
                del self.tool_registry.tools[tool_name]
                removed += 1
                logger.debug(f"Removed tool '{tool_name}'")
        
        # Task mappings are ephemeral - will be rebuilt on next agent task
        logger.info(f"Removed {removed} tools for database source {source_id}")
        return removed
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize database name for use in tool names.
        Converts to lowercase, replaces spaces with underscores.
        """
        return name.lower().replace(" ", "_").replace("-", "_")


# Singleton instance
_integration_service = None

def get_database_tool_integration() -> DatabaseToolIntegration:
    """Get or create the database tool integration singleton."""
    global _integration_service
    if _integration_service is None:
        from modules.tools.registry import get_tool_registry
        from modules.tools.services.tool_capability_mapper import get_tool_capability_mapper
        
        _integration_service = DatabaseToolIntegration(
            tool_registry=get_tool_registry(),
            capability_mapper=get_tool_capability_mapper()
        )
    return _integration_service
