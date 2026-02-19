"""
Chatbot Tool Router — Thin wrapper over shared tool_router
===========================================================

Re-exports shared functions for backward compatibility.
Adds chatbot-specific result formatting (build_tool_context_message).

All generic tool execution logic now lives in modules.tools.tool_router.
This module re-exports with original names so existing imports keep working.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from modules.tools.tool_router import (
    # Public API — re-export with ORIGINAL names for backward compat
    get_tools_for_agent as get_chatbot_tools,
    get_agent_tools as get_chat_tools,
    execute_tool,
    execute_tool_with_validation,
    validate_action_for_intent,
    get_filtered_composio_actions,
    get_capability_filter_stats,
    ToolRouter,
    # NOTE: We do NOT re-export get_tool_router from shared — we override it
    # below to return a ChatToolRouter with build_tool_context_message support.
)
from modules.tools.formatting.result_formatter import ToolResultFormatter

logger = logging.getLogger(__name__)

# Re-export everything at module level for backward compatibility
__all__ = [
    "get_chatbot_tools",
    "get_chat_tools",
    "execute_tool",
    "execute_tool_with_validation",
    "validate_action_for_intent",
    "get_filtered_composio_actions",
    "get_capability_filter_stats",
    "ToolRouter",
    "get_tool_router",
    "ChatToolRouter",
    "build_tool_context_message",
]


# =============================================================================
# CHATBOT-SPECIFIC: Tool context formatting for LLM injection
# =============================================================================

def build_tool_context_message(
    tool_name: str,
    result: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """
    Build a system message with tool results for LLM context.

    This is chatbot-specific presentation logic (emojis, download links,
    "CRITICAL INSTRUCTIONS" prompts). Other consumers format differently.

    Returns None if results are empty or tool failed.
    """
    formatter = ToolResultFormatter

    if not result.get('success'):
        return None

    raw = result.get('raw_result', {})
    standardized = formatter.standardize_result(raw, tool_name)

    if not standardized.get('results'):
        return None

    # Build context based on tool type
    if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
        docs = standardized['results']
        if not docs:
            return None

        # Group chunks by document
        docs_by_source = {}
        for doc in docs:
            source = doc.get('filename', doc.get('source', 'Unknown'))
            similarity = doc.get('similarity', doc.get('score', 0.0))
            content = doc.get('excerpt', doc.get('content', ''))

            # Extract document_id and metadata for API-based access
            metadata = doc.get('metadata', {})
            document_id = doc.get('document_id', metadata.get('document_id'))
            chunk_id = doc.get('chunk_id', doc.get('id'))

            if source not in docs_by_source:
                docs_by_source[source] = {
                    'source': source,
                    'document_id': document_id,
                    'chunks': [],
                    'chunk_ids': [],
                    'max_similarity': 0.0
                }

            docs_by_source[source]['chunks'].append(content)
            if chunk_id:
                docs_by_source[source]['chunk_ids'].append(chunk_id)
            docs_by_source[source]['max_similarity'] = max(
                docs_by_source[source]['max_similarity'],
                similarity
            )

        # Sort by relevance
        sorted_docs = sorted(
            docs_by_source.values(),
            key=lambda d: d['max_similarity'],
            reverse=True
        )[:5]  # Top 5 documents

        # Build user-friendly context
        doc_parts = [f"I found relevant information in {len(sorted_docs)} document(s):", ""]

        for i, doc_info in enumerate(sorted_docs, 1):
            source = doc_info['source']
            document_id = doc_info.get('document_id')
            relevance = int(doc_info['max_similarity'] * 100)
            chunk_count = len(doc_info['chunks'])

            # Extract title from filename
            title = source.replace('.md', '').replace('.pdf', '')
            title = title.replace('-', ' ').replace('_', ' ').title()

            # Remove numbering prefix (e.g., "02-" or "30-")
            title = re.sub(r'^\d+\s*', '', title)

            doc_parts.append(f"\U0001f4c4 **[{i}] {title}** ({source})")
            doc_parts.append(f"   \u2022 {chunk_count} relevant section(s) found")
            doc_parts.append(f"   \u2022 Relevance: {relevance}%")

            # Add document link using API endpoint (not hardcoded local path)
            if document_id:
                doc_parts.append(f"   \u2022 [View Document](/api/documents/{document_id}/content)")
            doc_parts.append("")

        # Add full content from all chunks for LLM to use
        doc_parts.append("\n\U0001f4da **FULL CONTENT FROM DOCUMENTS** (use this to answer the question):\n")
        for i, doc_info in enumerate(sorted_docs, 1):
            source = doc_info['source']
            title = source.replace('.md', '').replace('.pdf', '')
            title = title.replace('-', ' ').replace('_', ' ').title()
            title = re.sub(r'^\d+\s*', '', title)

            doc_parts.append(f"\n--- {title} ---")
            # Include ALL relevant chunks, not just preview
            for chunk in doc_info['chunks']:
                doc_parts.append(chunk)
                doc_parts.append("")

        doc_context = "\n".join(doc_parts)

        return {
            "role": "system",
            "content": f"{doc_context}\n\n**CRITICAL INSTRUCTIONS**:\n1. READ ALL the full content provided above\n2. SYNTHESIZE information from multiple sources\n3. Write a COMPREHENSIVE answer using the actual content\n4. Do NOT just list documents or say 'search didn't yield results'\n5. The content IS THERE - use it to write detailed responses\n6. Cite document names when referencing specific information\n7. If asked for code examples, include the actual code shown above\n8. If asked for statistics, use the actual numbers provided"
        }

    elif tool_name in ['search_codebase', 'search_code']:
        code_results = standardized['results'][:5]
        if not code_results:
            return None

        code_context = "\n\n".join([
            f"\U0001f4c1 {r.get('symbol_name', 'Code')} ({r.get('file_path', 'unknown')})\n```python\n{r.get('code', '')[:800]}\n```"
            for r in code_results
        ])

        return {
            "role": "system",
            "content": f"\U0001f4bb CODE FROM USER'S CODEBASE:\n\n{code_context}\n\nYou MUST show and explain this code when answering."
        }

    elif tool_name in ['query_database', 'smart_query_database']:
        # Handle clarification needed
        if raw.get('status') == 'needs_clarification':
            clarifications = raw.get('clarifications', [])
            return {
                "role": "system",
                "content": f"\U0001f914 CLARIFICATION NEEDED:\n\nThe query is ambiguous. Ask the user:\n" +
                          "\n".join(f"\u2022 {q}" for q in clarifications) +
                          "\n\nOnce they answer, you can query again with more specifics."
            }

        # Include ALL data — already limited at query level
        all_data = raw.get('data', [])
        data_str = json.dumps(all_data, default=str, indent=2)[:3000]

        db_context = f"Database query result: {raw.get('row_count', 0)} rows returned\n"
        db_context += f"SQL: {raw.get('sql', 'N/A')[:300]}\n"

        # Include rephrased query if different
        if raw.get('rephrased_query'):
            db_context += f"Interpreted as: {raw.get('rephrased_query')}\n"

        db_context += f"COMPLETE DATA:\n{data_str}"

        # Include explanation
        if raw.get('explanation'):
            db_context += f"\n\n\U0001f4dd EXPLANATION: {raw.get('explanation')}"

        # Include PandasAI insight if available
        pandas_ai = raw.get('pandas_ai', {})
        if pandas_ai:
            db_context += f"\n\n\U0001f4ca AI ANALYSIS: {pandas_ai.get('summary', '')}"

        # Include visualization suggestion
        if raw.get('visualization'):
            viz = raw.get('visualization', {})
            db_context += f"\n\n\U0001f4c8 VISUALIZATION: Recommended {viz.get('type', 'chart')} chart"

        # Include follow-up suggestions
        follow_ups = raw.get('follow_up_questions', [])
        if follow_ups:
            db_context += f"\n\n\U0001f4a1 FOLLOW-UP IDEAS:\n" + "\n".join(f"\u2022 {q}" for q in follow_ups[:3])

        return {
            "role": "system",
            "content": f"\U0001f5c4\ufe0f DATABASE QUERY RESULTS:\n\n{db_context}\n\nYou MUST present ALL this data to the user, not just a summary. Show complete results. If there's a chart in the artifacts panel, tell the user to check it."
        }

    return None


class ChatToolRouter(ToolRouter):
    """Chatbot-specific extensions to the generic ToolRouter."""

    def build_tool_context_message(
        self,
        tool_name: str,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Delegate to the module-level function for chatbot-specific formatting."""
        return build_tool_context_message(tool_name, result)


# =============================================================================
# CHATBOT get_tool_router — overrides the shared version to return
# a ChatToolRouter (which has build_tool_context_message).
# =============================================================================

_chat_tool_router: Optional[ChatToolRouter] = None


def get_tool_router() -> ChatToolRouter:
    """Get or create the chatbot-specific ToolRouter singleton."""
    global _chat_tool_router
    if _chat_tool_router is None:
        _chat_tool_router = ChatToolRouter()
    return _chat_tool_router
