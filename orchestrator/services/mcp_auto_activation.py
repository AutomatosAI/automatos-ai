"""
MCP Auto-Activation Service
============================

Automatically enables MCP servers when matching credentials are added.

How it works:
1. User adds a credential (e.g., AWS)
2. System finds MCP servers with metadata.auto_enable_on_credential = "aws_credentials"
3. Enables those MCP servers (status: inactive → active)
4. Links credential to the MCP server
5. Logs the activation

This creates a seamless UX: Add credential → Tools instantly available!
"""

from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime
from typing import List, Dict, Any
import logging

from models import MCPTool, Credential
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)


class MCPAutoActivationService:
    """
    Service for automatically activating MCP servers based on credentials
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def activate_mcp_servers_for_credential(
        self,
        credential_id: int,
        credential_type_name: str
    ) -> Dict[str, Any]:
        """
        Auto-activate MCP servers that match the credential type
        
        Args:
            credential_id: ID of the newly created credential
            credential_type_name: Name of the credential type (e.g., "aws_credentials")
        
        Returns:
            Dictionary with activation results:
            {
                'activated_count': int,
                'total_matching': int,
                'servers': [server_names],
                'errors': [error_messages]
            }
        """
        
        logger.info(f"🔌 MCP Auto-Activation triggered for credential type: {credential_type_name}")
        
        try:
            # Find matching MCP servers
            # We check both:
            # 1. Exact match in credentials_schema.credential_type
            # 2. Match in metadata.auto_enable_on_credential
            matching_servers = self.db.query(MCPTool).filter(
                or_(
                    # Check credentials_schema.credential_type
                    MCPTool.credentials_schema['credential_type'].astext == credential_type_name,
                    # Check metadata.auto_enable_on_credential
                    MCPTool.tool_metadata['auto_enable_on_credential'].astext == credential_type_name
                ),
                # Only activate inactive servers
                MCPTool.status == 'inactive'
            ).all()
            
            logger.info(f"📋 Found {len(matching_servers)} inactive MCP servers matching '{credential_type_name}'")
            
            if len(matching_servers) == 0:
                logger.info("  ℹ️  No inactive servers to activate")
                return {
                    'activated_count': 0,
                    'total_matching': 0,
                    'servers': [],
                    'errors': []
                }
            
            # Activate each server
            activated_servers = []
            errors = []
            
            for server in matching_servers:
                try:
                    logger.info(f"  🔧 Activating: {server.name}")
                    
                    # Update status to active
                    server.status = 'active'
                    
                    # Update metadata to link credential
                    if server.tool_metadata is None:
                        server.tool_metadata = {}
                    
                    server.tool_metadata['linked_credential_id'] = credential_id
                    server.tool_metadata['activated_at'] = datetime.now().isoformat()
                    server.tool_metadata['activation_method'] = 'auto'
                    
                    # Mark metadata as modified (SQLAlchemy needs this for JSONB)
                    flag_modified(server, 'tool_metadata')
                    
                    server.updated_at = datetime.now()
                    
                    activated_servers.append(server.name)
                    logger.info(f"    ✅ Activated successfully")
                    
                except Exception as e:
                    error_msg = f"Failed to activate {server.name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"    ❌ {error_msg}")
            
            # Commit changes
            self.db.commit()
            
            logger.info(f"\n🎉 Auto-activation complete!")
            logger.info(f"  ✅ Activated: {len(activated_servers)}/{len(matching_servers)} servers")
            if errors:
                logger.warning(f"  ⚠️  Errors: {len(errors)}")
            
            # Log activated servers
            if activated_servers:
                logger.info(f"\n📦 Activated MCP Servers:")
                for server_name in activated_servers:
                    logger.info(f"  • {server_name}")
            
            return {
                'activated_count': len(activated_servers),
                'total_matching': len(matching_servers),
                'servers': activated_servers,
                'errors': errors
            }
        
        except Exception as e:
            logger.error(f"💥 Auto-activation failed: {e}", exc_info=True)
            self.db.rollback()
            return {
                'activated_count': 0,
                'total_matching': 0,
                'servers': [],
                'errors': [str(e)]
            }
    
    def get_activatable_servers_for_credential_type(
        self,
        credential_type_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get list of MCP servers that would be activated for a credential type
        (without actually activating them - useful for UI preview)
        
        Args:
            credential_type_name: Name of the credential type
        
        Returns:
            List of server info dictionaries
        """
        
        matching_servers = self.db.query(MCPTool).filter(
            or_(
                MCPTool.credentials_schema['credential_type'].astext == credential_type_name,
                MCPTool.tool_metadata['auto_enable_on_credential'].astext == credential_type_name
            ),
            MCPTool.status == 'inactive'
        ).all()
        
        return [
            {
                'id': server.id,
                'name': server.name,
                'description': server.description,
                'provider': server.provider,
                'category': server.category,
                'icon': server.icon,
                'tags': server.tags
            }
            for server in matching_servers
        ]
    
    async def deactivate_mcp_servers_for_credential(
        self,
        credential_id: int
    ) -> Dict[str, Any]:
        """
        Deactivate MCP servers when a credential is deleted
        
        Args:
            credential_id: ID of the deleted credential
        
        Returns:
            Deactivation results
        """
        
        logger.info(f"🔌 MCP Auto-Deactivation triggered for credential_id: {credential_id}")
        
        try:
            # Find servers linked to this credential
            linked_servers = self.db.query(MCPTool).filter(
                MCPTool.tool_metadata['linked_credential_id'].astext == str(credential_id),
                MCPTool.status == 'active'
            ).all()
            
            logger.info(f"📋 Found {len(linked_servers)} active MCP servers linked to credential {credential_id}")
            
            if len(linked_servers) == 0:
                return {
                    'deactivated_count': 0,
                    'servers': []
                }
            
            deactivated_servers = []
            
            for server in linked_servers:
                try:
                    logger.info(f"  🔌 Deactivating: {server.name}")
                    
                    # Update status to inactive
                    server.status = 'inactive'
                    
                    # Update metadata
                    if server.tool_metadata is None:
                        server.tool_metadata = {}
                    
                    server.tool_metadata['deactivated_at'] = datetime.now().isoformat()
                    server.tool_metadata['deactivation_reason'] = 'credential_deleted'
                    # Keep linked_credential_id for audit trail
                    
                    flag_modified(server, 'tool_metadata')
                    server.updated_at = datetime.now()
                    
                    deactivated_servers.append(server.name)
                    logger.info(f"    ✅ Deactivated successfully")
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed to deactivate {server.name}: {e}")
            
            self.db.commit()
            
            logger.info(f"\n🎉 Auto-deactivation complete: {len(deactivated_servers)} servers")
            
            return {
                'deactivated_count': len(deactivated_servers),
                'servers': deactivated_servers
            }
        
        except Exception as e:
            logger.error(f"💥 Auto-deactivation failed: {e}", exc_info=True)
            self.db.rollback()
            return {
                'deactivated_count': 0,
                'servers': [],
                'error': str(e)
            }


# Convenience functions for use in other modules

def get_auto_activation_service(db: Session) -> MCPAutoActivationService:
    """Get instance of auto-activation service"""
    return MCPAutoActivationService(db)

