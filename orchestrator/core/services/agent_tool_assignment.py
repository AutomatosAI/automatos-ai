"""
Agent Tool Assignment Service (Phase 2)
========================================

Service for managing which tools are assigned to which agents.

Features:
- Assign apps to agents (external Composio apps + internal tools)
- Get available apps for assignment (only connected apps)
- Track assignment history and priority
- Workspace-scoped permissions
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import and_

from core.metadata_cache.models import (
    ComposioAppsCache,
    AgentAppAssignment
)
from core.models.composio import ComposioEntity, ComposioConnection
from core.models.core import Agent

logger = logging.getLogger(__name__)


class AgentToolAssignmentService:
    """
    Service for managing agent-tool assignments.
    
    Separates concerns:
    - agent_app_assignments: Which apps an agent CAN use
    - agent_app_features: Which specific actions within those apps are enabled
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def assign_app_to_agent(
        self,
        agent_id: int,
        app_name: str,
        app_type: str = 'EXTERNAL',
        assigned_by: Optional[int] = None,
        priority: int = 0,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentAppAssignment:
        """
        Assign an app to an agent.
        
        Args:
            agent_id: Agent ID
            app_name: App name (e.g., 'GMAIL')
            app_type: 'EXTERNAL' or 'INTERNAL'
            assigned_by: User ID who made the assignment
            priority: Priority for tool routing (higher = preferred)
            config: App-specific configuration
            
        Returns:
            AgentAppAssignment object
        """
        logger.info(f"[assignment] Assigning {app_name} ({app_type}) to agent {agent_id}")
        
        # Validate agent exists
        agent = self.db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Validate app exists (for external apps)
        if app_type == 'EXTERNAL':
            app = self.db.query(ComposioAppsCache).filter(
                ComposioAppsCache.app_name == app_name.upper()
            ).first()
            if not app:
                raise ValueError(f"App {app_name} not found in cache")
        
        # Check if already assigned
        existing = self.db.query(AgentAppAssignment).filter(
            and_(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.app_name == app_name.upper()
            )
        ).first()
        
        if existing:
            # Update existing assignment
            existing.is_active = True
            existing.priority = priority
            existing.assigned_at = datetime.utcnow()
            if assigned_by:
                existing.assigned_by = assigned_by
            if config:
                existing.config = config
            self.db.commit()
            logger.info(f"[assignment] Updated existing assignment {existing.id}")
            return existing
        
        # Create new assignment
        assignment = AgentAppAssignment(
            agent_id=agent_id,
            app_name=app_name.upper(),
            app_type=app_type,
            assigned_by=assigned_by,
            is_active=True,
            priority=priority,
            config=config or {}
        )
        self.db.add(assignment)
        self.db.commit()
        self.db.refresh(assignment)
        
        logger.info(f"[assignment] Created new assignment {assignment.id}")
        return assignment
    
    def unassign_app_from_agent(self, agent_id: int, app_name: str) -> bool:
        """
        Unassign an app from an agent (soft delete).
        
        Args:
            agent_id: Agent ID
            app_name: App name
            
        Returns:
            True if successful, False if not found
        """
        logger.info(f"[assignment] Unassigning {app_name} from agent {agent_id}")
        
        assignment = self.db.query(AgentAppAssignment).filter(
            and_(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.app_name == app_name.upper()
            )
        ).first()
        
        if not assignment:
            logger.warning(f"[assignment] Assignment not found")
            return False
        
        assignment.is_active = False
        self.db.commit()
        logger.info(f"[assignment] Unassigned successfully")
        return True
    
    def get_agent_assignments(
        self,
        agent_id: int,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all app assignments for an agent.
        
        Args:
            agent_id: Agent ID
            active_only: Only return active assignments
            
        Returns:
            List of assignment dictionaries with app details
        """
        query = self.db.query(AgentAppAssignment).filter(
            AgentAppAssignment.agent_id == agent_id
        )
        
        if active_only:
            query = query.filter(AgentAppAssignment.is_active == True)
        
        assignments = query.order_by(AgentAppAssignment.priority.desc()).all()
        
        result = []
        for assignment in assignments:
            # Get app details from cache
            app = self.db.query(ComposioAppsCache).filter(
                ComposioAppsCache.app_name == assignment.app_name
            ).first()
            
            result.append({
                "id": assignment.id,
                "agent_id": assignment.agent_id,
                "app_name": assignment.app_name,
                "app_type": assignment.app_type,
                "is_active": assignment.is_active,
                "priority": assignment.priority,
                "assigned_at": assignment.assigned_at,
                "assigned_by": assignment.assigned_by,
                "config": assignment.config,
                # App details (if found)
                "display_name": app.display_name if app else assignment.app_name,
                "description": app.description if app else None,
                "logo_url": app.logo_url if app else None,
                "categories": app.categories if app else [],
                "action_count": app.action_count if app else 0
            })
        
        return result
    
    def get_available_apps_for_assignment(
        self,
        workspace_id: UUID,
        agent_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get apps available for assignment to an agent.
        
        Only returns:
        1. Connected external apps (from Composio)
        2. Internal tools (RAG, CodeGraph, Memory, NL2SQL)
        
        Does NOT return:
        - Unconnected Composio apps (user must connect first)
        
        Args:
            workspace_id: Workspace ID
            agent_id: Optional agent ID to mark already-assigned apps
            
        Returns:
            List of available apps with assignment status
        """
        logger.info(f"[assignment] Getting available apps for workspace={workspace_id}, agent={agent_id}")
        
        # Get workspace's connected external apps
        entity = self.db.query(ComposioEntity).filter(
            ComposioEntity.workspace_id == workspace_id
        ).first()
        
        connected_apps = []
        if entity:
            connections = self.db.query(ComposioConnection).filter(
                and_(
                    ComposioConnection.entity_id == entity.id,
                    ComposioConnection.status == 'active'
                )
            ).all()
            
            for conn in connections:
                app = self.db.query(ComposioAppsCache).filter(
                    ComposioAppsCache.app_name == conn.app_name
                ).first()
                if app:
                    connected_apps.append({
                        "app_name": app.app_name,
                        "display_name": app.display_name,
                        "description": app.description,
                        "logo_url": app.logo_url,
                        "categories": app.categories,
                        "action_count": app.action_count,
                        "app_type": "EXTERNAL",
                        "connected_at": conn.connected_at
                    })
        
        # Get internal tools
        internal_tools = self.db.query(ComposioAppsCache).filter(
            ComposioAppsCache.categories.contains(["internal"])
        ).all()
        
        internal_apps = [
            {
                "app_name": app.app_name,
                "display_name": app.display_name,
                "description": app.description,
                "logo_url": app.logo_url,
                "categories": app.categories,
                "action_count": app.action_count,
                "app_type": "INTERNAL",
                "connected_at": None
            }
            for app in internal_tools
        ]
        
        # Combine all available apps
        all_apps = connected_apps + internal_apps
        
        # If agent_id provided, mark which apps are already assigned
        if agent_id:
            assigned_app_names = set()
            assignments = self.db.query(AgentAppAssignment).filter(
                and_(
                    AgentAppAssignment.agent_id == agent_id,
                    AgentAppAssignment.is_active == True
                )
            ).all()
            assigned_app_names = {a.app_name for a in assignments}
            
            for app in all_apps:
                app["is_assigned"] = app["app_name"] in assigned_app_names
        
        logger.info(f"[assignment] Found {len(all_apps)} available apps ({len(connected_apps)} external, {len(internal_apps)} internal)")
        return all_apps
    
    def batch_assign(
        self,
        agent_id: int,
        app_names: List[str],
        assigned_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Batch assign multiple apps to an agent.
        
        Args:
            agent_id: Agent ID
            app_names: List of app names to assign
            assigned_by: User ID who made the assignment
            
        Returns:
            Summary of assignments
        """
        logger.info(f"[assignment] Batch assigning {len(app_names)} apps to agent {agent_id}")
        
        success_count = 0
        failed_apps = []
        
        for app_name in app_names:
            try:
                self.assign_app_to_agent(
                    agent_id=agent_id,
                    app_name=app_name,
                    assigned_by=assigned_by
                )
                success_count += 1
            except Exception as e:
                logger.error(f"[assignment] Failed to assign {app_name}: {e}")
                failed_apps.append({"app_name": app_name, "error": str(e)})
        
        return {
            "success_count": success_count,
            "failed_count": len(failed_apps),
            "failed_apps": failed_apps
        }
    
    def get_agent_app_names(self, agent_id: int, active_only: bool = True) -> List[str]:
        """
        Get just the app names assigned to an agent (for quick lookup).
        
        Args:
            agent_id: Agent ID
            active_only: Only return active assignments
            
        Returns:
            List of app names
        """
        query = self.db.query(AgentAppAssignment.app_name).filter(
            AgentAppAssignment.agent_id == agent_id
        )
        
        if active_only:
            query = query.filter(AgentAppAssignment.is_active == True)
        
        return [app_name for (app_name,) in query.all()]
