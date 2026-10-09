"""
Composio tool executors -- direct action, meta-tool, and tool router.
Extracted from unified_executor.py.
"""

import base64
import logging
import pathlib
from typing import Any, Dict, Optional
from uuid import UUID

from modules.tools.execution.composio_params import sent_params, stray_params
from modules.tools.registry.tool_registry import ToolSpec

logger = logging.getLogger(__name__)

# Image file extensions to detect in Composio results
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}
_EXT_TO_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif", ".svg": "image/svg+xml",
}


async def _upload_local_images(result: Dict[str, Any], workspace_id: Optional[UUID]) -> None:
    """Detect local image file paths in Composio results and upload to image store.

    Replaces local paths like /home/automatos/.composio/outputs/.../*.jpg with
    public URLs served by /api/generated-images/{image_id}.
    """
    if not result or not result.get("successful", result.get("success")):
        return

    data = result.get("data", {})
    if not isinstance(data, dict):
        return

    replaced = False
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        p = pathlib.Path(value)
        if p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if not p.is_absolute():
            continue
        # It's a local image path — try to read and upload
        if not p.exists():
            logger.warning("[Composio] Image path does not exist: %s", value)
            continue
        try:
            image_bytes = p.read_bytes()
            b64 = base64.b64encode(image_bytes).decode("ascii")
            mime = _EXT_TO_MIME.get(p.suffix.lower(), "image/png")
            ws_str = str(workspace_id) if workspace_id else "default"

            from core.services.image_store import get_image_store
            store = get_image_store()
            image_id = await store.save_image(b64, mime_type=mime, workspace_id=ws_str)

            from config import Config as config
            backend_url = (config.BACKEND_URL or "").rstrip("/")
            public_url = f"{backend_url}/api/generated-images/{image_id}"

            data[key] = public_url
            replaced = True
            logger.info(
                "[Composio] Uploaded local image %s -> %s (%d bytes)",
                value, public_url, len(image_bytes),
            )
            # Clean up the local file
            try:
                p.unlink()
            except OSError:
                pass
        except Exception as e:
            logger.warning("[Composio] Failed to upload image %s: %s", value, e)

    if replaced:
        result["data"] = data


async def execute_composio_tool(
    executor,
    tool_spec: ToolSpec,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID],
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a Composio action via ComposioToolExecutor."""
    if not executor.composio_executor:
        return {
            "success": False,
            "error": "Composio executor not available",
            "tool": tool_spec.name
        }

    action = tool_spec.metadata.get("action") if tool_spec.metadata else None
    if not action and tool_spec.name.startswith("composio_"):
        action = tool_spec.name.replace("composio_", "", 1)

    if not action:
        return {
            "success": False,
            "error": "Missing Composio action name",
            "tool": tool_spec.name
        }

    if not workspace_id:
        return {
            "success": False,
            "error": "Workspace ID required for Composio tool execution",
            "tool": tool_spec.name
        }

    params = parameters.get("params") if isinstance(parameters, dict) else None
    if params is None:
        params = parameters or {}
    trace = trace_id or "no-trace"
    logger.info(
        f"[tool-trace {trace}] Composio execute action={action} "
        f"agent={agent_id} workspace={workspace_id} params_keys={list(params.keys()) if isinstance(params, dict) else type(params).__name__}"
    )

    return await executor.composio_executor.execute(
        action=action,
        params=params,
        agent_id=agent_id,
        workspace_id=workspace_id
    )


def _attach_schemas(executor, result: Dict[str, Any], action: str) -> None:
    """Attach the cached response/parameters schema for the action (or the one it was remapped
    to): generic widget detection without hardcoding provider names. A failed lookup is logged."""
    try:
        from core.models.composio_cache import ComposioActionCache
        from sqlalchemy import or_

        executed_action = result.get("action", action)  # May have been remapped
        action_cache = executor.db.query(ComposioActionCache).filter(
            or_(
                ComposioActionCache.action_name == action,
                ComposioActionCache.action_name == executed_action,
                ComposioActionCache.action_slug == action.lower().replace("_", "-"),
            )
        ).first()

        if action_cache:
            if action_cache.response_schema:
                result["response_schema"] = action_cache.response_schema
                logger.info(f"[Composio] Attached response_schema for {action_cache.action_name}")
            if action_cache.parameters:
                result["parameters_schema"] = action_cache.parameters
        else:
            logger.warning(f"[Composio] No cache entry found for action={action} or executed={executed_action}")
    except Exception:  # noqa: BLE001 — a failed schema lookup never fails the action that ran
        logger.warning(f"[Composio] Could not look up the schema for action={action}", exc_info=True)


async def execute_composio_execute(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute an arbitrary Composio action for an assigned/connected app.

    Expected input:
    - app_name: "GMAIL" / "SLACK" / ...
    - action: action name as stored in composio_actions_cache (e.g., "GMAIL_LIST_EMAILS")
    - params: object passed to the Composio action
    """
    if not executor.composio_executor:
        return {"success": False, "error": "Composio executor not available", "tool": tool_name}
    if not workspace_id:
        return {"success": False, "error": "workspace_id required for Composio execution", "tool": tool_name}

    if not isinstance(parameters, dict):
        return {"success": False, "error": "Invalid parameters (expected object)", "tool": tool_name}

    raw_action = parameters.get("action") or parameters.get("action_name")
    # `params` or `parameters`, with the top-level keys a model left beside them: the
    # params the owner's-click card shows are these (P256-FIX-RVW-31).
    params = sent_params(parameters)
    app_name = parameters.get("app_name") or parameters.get("app")
    strays = stray_params(parameters)
    if strays:
        logger.info(f"[composio_execute] Remapped top-level keys into params: {list(strays)}")

    if not raw_action:
        return {"success": False, "error": "Missing required field: action", "tool": tool_name}

    # Normalize action name to uppercase (Composio actions are typically uppercase)
    # This handles LLM inconsistency (e.g., "slack_send_message" vs "SLACK_SEND_MESSAGE")
    action = str(raw_action).upper().strip()

    trace = trace_id or "no-trace"
    logger.info(
        f"[tool-trace {trace}] Composio execute app={app_name} action={action} "
        f"(raw: {raw_action}) agent={agent_id} workspace={workspace_id} params_keys={list(params.keys())}"
    )

    result = await executor.composio_executor.execute(
        action=action,
        params=params,
        agent_id=agent_id,
        workspace_id=workspace_id,
        app_name=str(app_name).upper().strip() if app_name else None,
    )

    # Post-process: upload local image files to image store and replace paths with public URLs
    await _upload_local_images(result, workspace_id)

    # PRD-139: Telemetry write removed — unified hook in execute_tool handles this.
    # See modules/tools/execution/telemetry.py

    _attach_schemas(executor, result, action)
    return result


async def execute_composio_tool_router(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id: Optional[UUID] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Composio Tool Router meta-tools (search_tools, execute_tool).

    Tool Router is scoped to agent's assigned apps for better action selection.
    """
    from modules.tools.execution.composio_router_executor import ComposioToolRouterExecutor

    try:
        # Initialize Tool Router executor for this agent
        router_executor = ComposioToolRouterExecutor(
            db_session=executor.db_session,
            workspace_id=workspace_id,
            agent_id=agent_id
        )

        # Route to appropriate method
        if tool_name == "composio_search_tools":
            query = parameters.get("query")
            max_results = parameters.get("max_results", 5)

            if not query:
                return {
                    "success": False,
                    "error": "Missing required parameter: query",
                    "tool": tool_name
                }

            result = router_executor.search_tools(query, max_results)

        elif tool_name == "composio_execute_tool":
            action = parameters.get("action")
            params = parameters.get("params", {})

            if not action:
                return {
                    "success": False,
                    "error": "Missing required parameter: action",
                    "tool": tool_name
                }

            result = router_executor.execute_tool(action, params)

        else:
            return {
                "success": False,
                "error": f"Unknown Tool Router tool: {tool_name}",
                "tool": tool_name
            }

        return result

    except Exception as e:
        logger.error(f"[Tool Router] Execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name
        }
