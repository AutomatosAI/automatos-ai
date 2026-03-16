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
) -> Dict[str, Any]:
    """Execute multimodal search (tables, images, formulas, combined)."""
    from modules.rag.services.multimodal_knowledge_tools import MultimodalKnowledgeTools

    logger.info(f"Agent {agent_id} executing multimodal tool: {tool_name}")

    try:
        tools = MultimodalKnowledgeTools()

        if tool_name == 'search_multimodal':
            return await tools.search_multimodal(**parameters)
        elif tool_name == 'search_tables':
            return await tools.search_tables(**parameters)
        elif tool_name == 'search_images':
            return await tools.search_images(**parameters)
        elif tool_name == 'search_formulas':
            return await tools.search_formulas(**parameters)
        else:
            return {"success": False, "error": f"Unknown multimodal tool: {tool_name}"}
    except Exception as e:
        logger.error(f"Multimodal tool error: {e}")
        return {"success": False, "error": str(e)}
