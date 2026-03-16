"""
File operation executors -- read, write, list, create, delete files.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def execute_file_op(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
) -> Dict[str, Any]:
    """Execute file operations via AgentActionExecutor."""
    requested_path = parameters.get('path', '') or parameters.get('file_path', '')

    # Log the requested path vs workspace for debugging
    logger.info(f"File operation '{tool_name}' requested for path: {requested_path}")
    logger.info(f"   Workspace root: {executor.workspace_dir}")

    if tool_name == 'read_file':
        success, content = executor.action_executor.read_file(requested_path)
        result = {
            "success": success,
            "action": "read_file",
            "params": parameters,
            "requested_path": requested_path,
            "workspace": str(executor.workspace_dir),
        }
        if success:
            result["result"] = content
        else:
            result["error"] = content
            logger.warning(f"read_file failed: {content}")
        return result

    elif tool_name == 'write_file':
        success, msg = executor.action_executor.write_file(
            requested_path,
            parameters.get('content', '')
        )
        result = {
            "success": success,
            "action": "write_file",
            "params": parameters,
            "requested_path": requested_path,
            "workspace": str(executor.workspace_dir),
        }
        if success:
            result["result"] = msg
        else:
            result["error"] = msg
            logger.warning(f"write_file failed: {msg}")
        return result

    elif tool_name == 'list_directory':
        success, items = executor.action_executor.list_directory(requested_path or '.')
        result = {
            "success": success,
            "action": "list_directory",
            "params": parameters,
            "requested_path": requested_path or '.',
            "workspace": str(executor.workspace_dir),
        }
        if success:
            result["result"] = items
        else:
            result["error"] = items
            logger.warning(f"list_directory failed: {items}")
        return result

    elif tool_name == 'create_directory':
        success, msg = executor.action_executor.create_directory(requested_path)
        result = {
            "success": success,
            "action": "create_directory",
            "params": parameters,
            "requested_path": requested_path,
            "workspace": str(executor.workspace_dir),
        }
        if success:
            result["result"] = msg
        else:
            result["error"] = msg
            logger.warning(f"create_directory failed: {msg}")
        return result

    elif tool_name == 'delete_file':
        success, msg = executor.action_executor.delete_file(requested_path)
        result = {
            "success": success,
            "action": "delete_file",
            "params": parameters,
            "requested_path": requested_path,
            "workspace": str(executor.workspace_dir),
        }
        if success:
            result["result"] = msg
        else:
            result["error"] = msg
            logger.warning(f"delete_file failed: {msg}")
        return result

    else:
        return {
            "success": False,
            "error": f"Unknown file operation: {tool_name}"
        }
