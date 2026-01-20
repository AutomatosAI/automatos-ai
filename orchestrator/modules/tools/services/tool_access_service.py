"""
Tool Access Service (PRD-35)
============================

Service for managing tool access, validation, and execution routing.

Responsibilities:
- Validate agent has access to a tool (assignment + tenant enablement)
- Get available tools for an agent
- Route tool execution through Context Forge
- Handle credential resolution for hosted mode
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.tool_assignments import (
    AgentToolAssignment,
    TenantToolConfig,
)
from modules.tools.services.adapter_tools_client import (
    AdapterToolsClient,
    get_adapter_tools_client,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentTool:
    """Tool available to an agent with all necessary metadata."""
    adapter_tool_id: str
    name: str
    description: Optional[str]
    category: Optional[str]
    provider: Optional[str]
    icon: Optional[str]
    credential_id: Optional[int]
    methods: List[str]
    enabled: bool = True


class ToolAccessService:
    """
    Service for managing tool access and execution.
    
    This service implements the PRD-35 architecture:
    - Adapter owns tool definitions (fetched via AdapterToolsClient)
    - Automatos owns enablement (TenantToolConfig) and assignments (AgentToolAssignment)
    - Tools must be enabled at tenant level AND assigned to agent
    """
    
    def __init__(
        self,
        db: Session,
        adapter_client: Optional[AdapterToolsClient] = None
    ):
        self.db = db
        self.adapter_client = adapter_client or get_adapter_tools_client()
    
    # =========================================================================
    # Tool Access Validation
    # =========================================================================
    
    async def validate_tool_access(
        self,
        agent_id: int,
        tenant_id: UUID,
        adapter_tool_id: str
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate if an agent can use a tool.
        
        Checks:
        1. Tool is assigned to agent (AgentToolAssignment.enabled = true)
        2. Tool is enabled for tenant (TenantToolConfig.enabled = true)
        3. Credential is configured for tool
        
        Args:
            agent_id: The agent's ID
            tenant_id: The tenant's UUID
            adapter_tool_id: The tool's ID from Adapter
            
        Returns:
            Tuple of (has_access, error_message, credential_id)
        """
        # Check agent assignment
        assignment = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == adapter_tool_id,
            AgentToolAssignment.enabled == True
        ).first()
        
        if not assignment:
            return False, f"Tool '{adapter_tool_id}' not assigned to agent {agent_id}", None
        
        # Check tenant enablement
        tenant_config = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.tool_id == adapter_tool_id,
            TenantToolConfig.enabled == True
        ).first()
        
        if not tenant_config:
            return False, f"Tool '{adapter_tool_id}' not enabled for tenant", None
        
        if not tenant_config.credential_id:
            return False, f"Tool '{adapter_tool_id}' has no configured credential", None
        
        return True, None, tenant_config.credential_id
    
    # =========================================================================
    # Get Tools for Agent
    # =========================================================================
    
    async def get_tools_for_agent(
        self,
        agent_id: int,
        tenant_id: UUID
    ) -> List[AgentTool]:
        """
        Get all tools available to an agent, filtering by permissions and active context.
        
        Args:
            agent_id: The agent's ID
            tenant_id: The tenant's UUID
            
        Returns:
            List of AgentTool objects with full metadata
        """
        # 1. Fetch agent to check active context
        # 1. Fetch agent to check active context
        from core.models import Agent
        agent = self.db.query(Agent).get(agent_id)
        
        # Default to 'general' context (Dynamic Loading) instead of 'all'
        active_context = "general"
        if agent and agent.configuration:
             active_context = agent.configuration.get("active_context", "general")
        
        # [HOTFIX] Force Chatbot (Agent 1) to start in 'general' context even if config says 'all'
        # This ensures the user sees the Dynamic Tooling experience immediately.
        if agent_id == 1 and active_context == "all":
            active_context = "general"
        
        # Context-to-Category Mapping
        # TODO: This hard-coded approach won't scale to 400+ tools - needs dynamic solution
        CONTEXT_MAP = {
            "general": ["communication", "research", "productivity", "system", "collaboration", "developer"], # shell removed
            "coding": ["developer", "github", "git", "code", "file_ops", "devtools"],
            "ops": ["cloud", "k8s", "aws", "infrastructure", "monitoring", "database", "shell"],
            "communication": ["communication", "slack", "email", "chat", "collaboration"],
            "research": ["research", "data", "search", "rag"],
            # 'all' implies no filtering
        }
        
        allowed_categories = CONTEXT_MAP.get(active_context)
        logger.info(f"🔍 [ToolAccess] Agent {agent_id} Context: '{active_context}'. Allowed categories: {allowed_categories}")

        # Get agent's tool assignments
        assignments = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.enabled == True
        ).all()
        
        if not assignments:
            return []
        
        assigned_tool_ids = [a.tool_id for a in assignments]
        
        # Get tenant's enabled tools that are assigned to this agent
        tenant_tools = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.enabled == True,
            TenantToolConfig.tool_id.in_(assigned_tool_ids)
        ).all()
        
        if not tenant_tools:
            return []
        
        # Fetch tool definitions from Adapter or Cache
        result: List[AgentTool] = []
        
        for tenant_tool in tenant_tools:
            try:
                # [NEW] Check Cache First (Just-in-Time Pattern)
                meta = tenant_tool.cached_metadata or {}
                
                # Apply Context Filtering
                if allowed_categories:
                    # Resolve category from cache or name
                    tool_cat = meta.get("category", "").lower() if meta else ""
                    # Exception: Always allow system/meta tools like 'switch_context'
                    is_system_tool = tenant_tool.adapter_tool_name in ["switch_context", "search_knowledge"]
                    
                    if not is_system_tool and tool_cat not in allowed_categories:
                        continue

                if tenant_tool.cached_metadata:
                    result.append(AgentTool(
                        adapter_tool_id=tenant_tool.tool_id,
                        name=tenant_tool.adapter_tool_name,
                        description=None, # Description might be in meta if we saved it
                        category=meta.get("category"),
                        provider=None, # Provider not always in meta, but less critical
                        icon=None,
                        credential_id=tenant_tool.credential_id,
                        methods=meta.get("methods", []),
                        enabled=True
                    ))
                    continue

                # Fallback: Live Fetch (Legacy / Migration support)
                logger.warning(f"Cache miss for tool {tenant_tool.tool_id}, fetching live")
                tool_def = await self.adapter_client.get_tool(tenant_tool.tool_id)
                
                if tool_def:
                    # Apply filtering context check again for live fetched tools
                    if allowed_categories:
                        tool_cat = tool_def.get("category", "").lower()
                        is_system_tool = tool_def.get("name") in ["switch_context", "search_knowledge"]
                        if not is_system_tool and tool_cat not in allowed_categories:
                            continue

                    methods = self.adapter_client.get_tool_methods(tool_def)
                    
                    result.append(AgentTool(
                        adapter_tool_id=tenant_tool.tool_id,
                        name=tool_def.get("name") or tenant_tool.adapter_tool_name,
                        description=tool_def.get("description"),
                        category=tool_def.get("category"),
                        provider=tool_def.get("provider"),
                        icon=tool_def.get("icon"),
                        credential_id=tenant_tool.credential_id,
                        methods=methods,
                        enabled=True
                    ))
                else:
                    # Tool not found in Adapter, use cached name
                    logger.warning(f"Tool {tenant_tool.tool_id} not found in Adapter")
                    result.append(AgentTool(
                        adapter_tool_id=tenant_tool.tool_id,
                        name=tenant_tool.adapter_tool_name,
                        description=None,
                        category=None,
                        provider=None,
                        icon=None,
                        credential_id=tenant_tool.credential_id,
                        methods=[],
                        enabled=True
                    ))
                    
            except Exception as e:
                logger.error(f"Error fetching tool {tenant_tool.tool_id}: {e}")
                continue
        
        return result
    
    # =========================================================================
    # Tool Enablement (Tenant Level)
    # =========================================================================
    
    async def get_available_tools_for_tenant(
        self,
        tenant_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all tools available for a tenant to enable.
        
        Returns tools from Adapter with tenant-specific enablement status.
        
        Args:
            tenant_id: The tenant's UUID
            
        Returns:
            List of tools with enablement status
        """
        # Get all tools from Adapter
        adapter_tools = await self.adapter_client.list_tools(enabled_only=True)
        
        # Get tenant's tool configs
        tenant_configs = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id
        ).all()
        
        # Build lookup
        config_by_tool = {tc.tool_id: tc for tc in tenant_configs}
        
        # Merge data
        result = []
        for tool in adapter_tools:
            tool_id = tool.get("name") or tool.get("adapter_tool_id")
            tenant_config = config_by_tool.get(tool_id)
            
            result.append({
                "adapter_tool_id": tool_id,
                "name": tool.get("name"),
                "description": tool.get("description"),
                "provider": tool.get("provider"),
                "category": tool.get("category"),
                "credential_type": tool.get("credential_type"),
                "auth_config": tool.get("auth_config"),
                "icon": tool.get("icon"),
                "tags": tool.get("tags"),
                "enabled_for_tenant": tenant_config.enabled if tenant_config else False,
                "credential_configured": tenant_config.credential_id is not None if tenant_config else False,
                "tenant_config_id": tenant_config.id if tenant_config else None
            })
        
        return result
    
    async def enable_tool_for_tenant(
        self,
        tenant_id: UUID,
        adapter_tool_id: str,
        adapter_tool_name: str,
        credential_id: int,
        configuration: Optional[Dict[str, Any]] = None
    ) -> TenantToolConfig:
        """
        Enable a tool for a tenant with credential mapping.
        
        Args:
            tenant_id: The tenant's UUID
            adapter_tool_id: The tool's ID from Adapter
            adapter_tool_name: Display name for the tool
            credential_id: The credential to use for this tool (1:1 mapping)
            configuration: Optional tool configuration overrides
            
        Returns:
            The created or updated TenantToolConfig
        """
        # Check if config exists
        existing = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.tool_id == adapter_tool_id
        ).first()
        
        if existing:
            existing.enabled = True
            existing.credential_id = credential_id
            existing.adapter_tool_name = adapter_tool_name
            if configuration:
                existing.configuration = configuration
            
            # [NEW] Refresh cached metadata from Adapter
            try:
                # We need to fetch the full definition to get current methods/schema
                # Ideally this should be passed in or fetched here. 
                # Assuming adapter_client is available via self.adapter_client
                logger.info(f"Refreshing metadata for tool {adapter_tool_id}")
                tool_def = await self.adapter_client.get_tool(adapter_tool_id)
                if tool_def:
                    methods = self.adapter_client.get_tool_methods(tool_def)
                    existing.cached_metadata = {
                        "methods": methods,
                        "category": tool_def.get("category"),
                        "input_schema": tool_def.get("input_schema"), # If available from Adapter
                        "capabilities": tool_def.get("capabilities")
                    }
            except Exception as e:
                logger.error(f"Failed to refresh metadata for tool {adapter_tool_id}: {e}")

            self.db.commit()
            self.db.refresh(existing)
            logger.info(f"Updated tool config: tenant={tenant_id}, tool={adapter_tool_id}")
            return existing
        else:
            # [NEW] Fetch metadata for initial cache
            cached_meta = {}
            try:
                logger.info(f"Fetching metadata for new tool {adapter_tool_id}")
                tool_def = await self.adapter_client.get_tool(adapter_tool_id)
                if tool_def:
                    methods = self.adapter_client.get_tool_methods(tool_def)
                    cached_meta = {
                        "methods": methods,
                        "category": tool_def.get("category"),
                        "input_schema": tool_def.get("input_schema"),
                        "capabilities": tool_def.get("capabilities")
                    }
            except Exception as e:
                logger.error(f"Failed to fetch metadata for tool {adapter_tool_id}: {e}")

            config = TenantToolConfig(
                tenant_id=tenant_id,
                adapter_tool_id=adapter_tool_id,
                adapter_tool_name=adapter_tool_name,
                enabled=True,
                credential_id=credential_id,
                configuration=configuration or {},
                cached_metadata=cached_meta
            )
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
            logger.info(f"Created tool config: tenant={tenant_id}, tool={adapter_tool_id}")
            return config
    
    def disable_tool_for_tenant(
        self,
        tenant_id: UUID,
        adapter_tool_id: str
    ) -> bool:
        """
        Disable a tool for a tenant.
        
        Args:
            tenant_id: The tenant's UUID
            adapter_tool_id: The tool's ID from Adapter
            
        Returns:
            True if disabled, False if not found
        """
        config = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.tool_id == adapter_tool_id
        ).first()
        
        if not config:
            return False
        
        config.enabled = False
        self.db.commit()
        logger.info(f"Disabled tool: tenant={tenant_id}, tool={adapter_tool_id}")
        return True
    
    # =========================================================================
    # Agent Tool Assignment
    # =========================================================================
    
    async def get_agent_assignments(
        self,
        agent_id: int,
        tenant_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all tool assignments for an agent with full details.
        
        Args:
            agent_id: The agent's ID
            tenant_id: The tenant's UUID (for validation and metadata)
            
        Returns:
            List of assignments with tool details
        """
        # Get all assignments for agent
        assignments = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id
        ).all()
        
        # Get tenant's enabled tools for context
        tenant_configs = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.enabled == True
        ).all()
        
        enabled_tools = {tc.tool_id: tc for tc in tenant_configs}
        
        result = []
        for assignment in assignments:
            tool_def = await self.adapter_client.get_tool(assignment.tool_id)
            tenant_config = enabled_tools.get(assignment.tool_id)
            
            result.append({
                "id": assignment.id,
                "adapter_tool_id": assignment.tool_id,
                "name": tool_def.get("name") if tool_def else assignment.tool_id,
                "description": tool_def.get("description") if tool_def else None,
                "category": tool_def.get("category") if tool_def else None,
                "icon": tool_def.get("icon") if tool_def else None,
                "enabled": assignment.enabled,
                "tool_enabled_for_tenant": tenant_config is not None,
                "assigned_at": assignment.created_at
            })
        
        return result
    
    def assign_tool_to_agent(
        self,
        agent_id: int,
        tenant_id: UUID,
        adapter_tool_id: str,
        enabled: bool = True
    ) -> Tuple[AgentToolAssignment, Optional[str]]:
        """
        Assign a tool to an agent.
        
        Args:
            agent_id: The agent's ID
            tenant_id: The tenant's UUID
            adapter_tool_id: The tool's ID from Adapter
            enabled: Whether the assignment is enabled
            
        Returns:
            Tuple of (assignment, error_message)
        """
        # Validate tool is enabled for tenant
        tenant_config = self.db.query(TenantToolConfig).filter(
            TenantToolConfig.tenant_id == tenant_id,
            TenantToolConfig.tool_id == adapter_tool_id,
            TenantToolConfig.enabled == True
        ).first()
        
        if not tenant_config:
            return None, f"Tool '{adapter_tool_id}' must be enabled for your organization first"
        
        # Check if assignment exists
        existing = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == adapter_tool_id
        ).first()
        
        if existing:
            existing.enabled = enabled
            self.db.commit()
            self.db.refresh(existing)
            logger.info(f"Updated assignment: agent={agent_id}, tool={adapter_tool_id}, enabled={enabled}")
            return existing, None
        else:
            assignment = AgentToolAssignment(
                agent_id=agent_id,
                tool_id=adapter_tool_id,
                enabled=enabled
            )
            self.db.add(assignment)
            self.db.commit()
            self.db.refresh(assignment)
            logger.info(f"Created assignment: agent={agent_id}, tool={adapter_tool_id}")
            return assignment, None
    
    def remove_tool_from_agent(
        self,
        agent_id: int,
        adapter_tool_id: str
    ) -> bool:
        """
        Remove a tool assignment from an agent.
        
        Args:
            agent_id: The agent's ID
            adapter_tool_id: The tool's ID from Adapter
            
        Returns:
            True if removed, False if not found
        """
        assignment = self.db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == agent_id,
            AgentToolAssignment.tool_id == adapter_tool_id
        ).first()
        
        if not assignment:
            return False
        
        self.db.delete(assignment)
        self.db.commit()
        logger.info(f"Removed assignment: agent={agent_id}, tool={adapter_tool_id}")
        return True
    
    def batch_update_agent_assignments(
        self,
        agent_id: int,
        tenant_id: UUID,
        assignments: List[Dict[str, Any]]
    ) -> Tuple[int, List[str]]:
        """
        Batch update agent tool assignments.
        
        Args:
            agent_id: The agent's ID
            tenant_id: The tenant's UUID
            assignments: List of {"adapter_tool_id": str, "enabled": bool}
            
        Returns:
            Tuple of (count updated, list of errors)
        """
        updated = 0
        errors = []
        
        for item in assignments:
            adapter_tool_id = item.get("adapter_tool_id")
            enabled = item.get("enabled", True)
            
            if not adapter_tool_id:
                continue
            
            if enabled:
                result, error = self.assign_tool_to_agent(
                    agent_id, tenant_id, adapter_tool_id, enabled=True
                )
                if error:
                    errors.append(error)
                else:
                    updated += 1
            else:
                # Just update enabled status, don't remove
                existing = self.db.query(AgentToolAssignment).filter(
                    AgentToolAssignment.agent_id == agent_id,
                    AgentToolAssignment.tool_id == adapter_tool_id
                ).first()
                
                if existing:
                    existing.enabled = False
                    updated += 1
                else:
                    # Create disabled assignment for tracking
                    assignment = AgentToolAssignment(
                        agent_id=agent_id,
                        tool_id=adapter_tool_id,
                        enabled=False
                    )
                    self.db.add(assignment)
                    updated += 1
        
        self.db.commit()
        return updated, errors


# Factory function
def get_tool_access_service(db: Session) -> ToolAccessService:
    """Create a ToolAccessService instance."""
    return ToolAccessService(db)
