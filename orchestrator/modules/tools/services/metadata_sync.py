"""
Tool Metadata Sync Service

Fetches tool metadata from Context Forge and caches it in MCPTool.tool_metadata.
This enables fast runtime tool loading without querying CF on every chat request.

Usage:
    from modules.tools.services.metadata_sync import sync_tool_metadata
    
    # Sync Slack tool metadata
    success = await sync_tool_metadata(db, "slack")
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import httpx
from sqlalchemy.orm import Session

from core.models.core import MCPTool

logger = logging.getLogger(__name__)


class MetadataSyncError(Exception):
    """Raised when metadata sync fails"""
    pass


async def get_cf_tools(tool_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Query Context Forge /tools endpoint to get tool catalog.
    
    Args:
        tool_id: Optional filter for specific tool (e.g., "slack")
        
    Returns:
        List of tool definitions from Context Forge
        
    Raises:
        MetadataSyncError: If CF is unavailable or returns an error
    """
    cf_url = os.getenv("CONTEXT_FORGE_URL", "https://mcp.automatos.app")
    cf_token = os.getenv("CONTEXT_FORGE_TOKEN", "")
    
    if not cf_token:
        raise MetadataSyncError("CONTEXT_FORGE_TOKEN not configured")
    
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Fetching tools from Context Forge: {cf_url}/tools")
            response = await client.get(f"{cf_url}/tools", headers=headers)
            response.raise_for_status()
            
            tools = response.json()
            
            # Filter for specific tool if requested
            if tool_id:
                # Match tools like: automatos-unified-adapter-mcp-slack-*
                prefix = f"automatos-unified-adapter-mcp-{tool_id}-"
                tools = [t for t in tools if t.get("name", "").startswith(prefix)]
                logger.info(f"Filtered to {len(tools)} tools for tool_id='{tool_id}'")
            
            return tools
            
    except httpx.HTTPError as e:
        raise MetadataSyncError(f"Failed to fetch tools from Context Forge: {str(e)}")


def parse_adapter_tool_metadata(tools: List[Dict[str, Any]], tool_id: str) -> Dict[str, Any]:
    """
    Parse Context Forge tool list into our metadata format.
    
    Args:
        tools: List of tools from CF (filtered to one adapter tool)
        tool_id: The tool identifier (e.g., "slack")
        
    Returns:
        Metadata dict ready for MCPTool.tool_metadata
        
    Example CF tool name: automatos-unified-adapter-mcp-slack-chat-postmessage
    Parsed to: { "methods": { "chat.postMessage": {...} } }
    """
    methods = {}
    prefix = f"automatos-unified-adapter-mcp-{tool_id}-"
    
    for tool in tools:
        cf_name = tool.get("name", "")
        if not cf_name.startswith(prefix):
            continue
        
        # Extract method name: slack-chat-postmessage → chat.postMessage
        method_part = cf_name[len(prefix):]  # "chat-postmessage"
        
        # Convert dashes to dots for namespace
        # Rules: 
        # - First dash becomes dot (chat-postmessage → chat.postmessage)
        # - CamelCase the method (postmessage → postMessage)
        parts = method_part.split("-", 1)
        if len(parts) == 2:
            namespace, method = parts
            # Restore camelCase for common patterns
            method = restore_camel_case(method)
            method_name = f"{namespace}.{method}"
        else:
            method_name = method_part
        
        # Store method metadata
        methods[method_name] = {
            "cf_tool_name": cf_name,
            "description": tool.get("description", ""),
            "input_schema": tool.get("inputSchema", {})
        }
    
    logger.info(f"Parsed {len(methods)} methods for tool '{tool_id}'")
    
    return {
        "methods": methods,
        "metadata_version": "1.0",
        "last_synced_at": datetime.utcnow().isoformat()
    }


def restore_camel_case(method: str) -> str:
    """
    Restore camelCase for common Slack API patterns.
    
    postmessage → postMessage
    scheduledmessages-list → scheduledMessages.list (handled by caller)
    """
    # Common Slack API patterns
    replacements = {
        "postmessage": "postMessage",
        "postephemeral": "postEphemeral",
        "schedulemessage": "scheduleMessage",
        "deletescheduledmessage": "deleteScheduledMessage",
        "getpermalink": "getPermalink",
        "userlookupbyemail": "userLookupByEmail"
    }
    
    return replacements.get(method.lower(), method)


async def sync_tool_metadata(db: Session, tool_id: str) -> bool:
    """
    Sync tool metadata from Context Forge and store in database.
    
    Args:
        db: Database session
        tool_id: Tool identifier (e.g., "slack", "github")
        
    Returns:
        True if sync successful, False otherwise
        
    Example:
        success = await sync_tool_metadata(db, "slack")
        if success:
            print("Slack metadata synced!")
    """
    try:
        logger.info(f"Starting metadata sync for tool: {tool_id}")
        
        # Get tool from database
        tool = db.query(MCPTool).filter(MCPTool.tool_id == tool_id).first()
        if not tool:
            raise MetadataSyncError(f"Tool '{tool_id}' not found in database")
        
        # Fetch from Context Forge
        cf_tools = await get_cf_tools(tool_id=tool_id)
        
        if not cf_tools:
            logger.warning(f"No tools found in Context Forge for tool_id='{tool_id}'")
            return False
        
        # Parse into our format
        metadata = parse_adapter_tool_metadata(cf_tools, tool_id)
        
        # Merge with existing metadata (preserve non-method fields)
        current_metadata = tool.tool_metadata or {}
        current_metadata.update(metadata)
        
        # Update database
        tool.tool_metadata = current_metadata
        db.commit()
        
        logger.info(f"✅ Synced {len(metadata['methods'])} methods for '{tool_id}'")
        return True
        
    except MetadataSyncError as e:
        logger.error(f"Metadata sync failed for '{tool_id}': {str(e)}")
        db.rollback()
        return False
    except Exception as e:
        logger.exception(f"Unexpected error syncing metadata for '{tool_id}'")
        db.rollback()
        return False
