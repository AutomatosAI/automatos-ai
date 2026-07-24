"""
Multimodal search executor -- tables, images, formulas, combined search.
Extracted from unified_executor.py.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def execute_multimodal_tool(
    executor,
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int,
    workspace_id=None,
) -> Dict[str, Any]:
    """Execute multimodal search (tables, images, formulas, combined).

    PRD-156 S1: ``workspace_id`` is threaded from the dispatcher and forced into
    every call (overriding any LLM-supplied value), so the four tools always
    scope to the caller's tenant. ``executor.db`` is the request-scoped session.
    """
    from modules.rag.services.multimodal_knowledge_tools import MultimodalKnowledgeTools

    logger.info(f"Agent {agent_id} executing multimodal tool: {tool_name} (workspace={workspace_id})")

    try:
        tools = MultimodalKnowledgeTools(db_session=getattr(executor, "db", None))

        # Tenant scope is authoritative from the dispatcher — never trust an
        # LLM-supplied workspace_id in the tool parameters.
        params = {**parameters, "workspace_id": workspace_id}

        if tool_name == 'search_multimodal':
            return await tools.search_multimodal(**params)
        elif tool_name == 'search_tables':
            return await tools.search_tables(**params)
        elif tool_name == 'search_images':
            return await tools.search_images(**params)
        elif tool_name == 'search_formulas':
            return await tools.search_formulas(**params)
        else:
            return {"success": False, "error": f"Unknown multimodal tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Multimodal tool error: {e}")
        return {"success": False, "error": str(e)}
