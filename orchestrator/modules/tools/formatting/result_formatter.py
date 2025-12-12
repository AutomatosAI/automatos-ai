"""
Unified Tool Result Formatter
=============================

SINGLE SOURCE OF TRUTH for formatting tool results.
All services use this - NO MORE DUPLICATION.
"""

import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ToolResultFormatter:
    """
    Unified formatter for all tool results.
    Ensures consistent structure across chatbot, agents, and workflows.
    """
    
    @staticmethod
    def _clean_document_filename(filename: str) -> str:
        """Clean document filename by removing hash prefixes."""
        if not filename:
            return 'Document'
        
        # Remove path components
        filename = filename.split('/')[-1]
        
        # Check for hash prefix pattern (32-64 char hex string followed by underscore)
        if '_' in filename:
            parts = filename.split('_', 1)
            if len(parts) > 1:
                first_part = parts[0]
                # Check if it looks like a hash (hexadecimal, 32-64 chars)
                if 32 <= len(first_part) <= 64 and all(c in '0123456789abcdef' for c in first_part.lower()):
                    return parts[1]
        
        return filename
    
    @staticmethod
    def _extract_useful_content(content: str, max_chars: int = 800) -> str:
        """Extract useful content from document chunk."""
        if not content:
            return ''
        
        excerpt = content.strip()
        
        # Smart truncation at sentence/paragraph boundary
        if len(excerpt) > max_chars:
            truncated = excerpt[:max_chars]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n\n')
            cut_point = max(last_period, last_newline)
            
            if cut_point > max_chars // 2:
                excerpt = truncated[:cut_point + 1].strip()
                if not excerpt.endswith('.'):
                    excerpt += '.'
            else:
                excerpt = truncated.strip()
            excerpt += '...'
        
        return excerpt
    
    @staticmethod
    def format_documents(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format document results consistently.
        
        Input: List of raw document dicts from RAG/agent_platform_tools
        Output: Standardized format for frontend/LLM
        """
        formatted = []
        
        for r in raw_results:
            # Handle different input formats
            raw_filename = (
                r.get('title') or 
                r.get('filename') or 
                r.get('source') or 
                r.get('source_file') or 
                'Document'
            )
            
            clean_name = ToolResultFormatter._clean_document_filename(raw_filename)
            
            # Extract content from various possible keys
            content = r.get('content') or r.get('text') or r.get('excerpt') or ''
            useful_excerpt = ToolResultFormatter._extract_useful_content(content, max_chars=500)
            
            # Extract similarity/relevance score
            similarity = (
                r.get('similarity') or 
                r.get('relevance') or 
                r.get('score') or 
                0.0
            )
            if isinstance(similarity, str):
                try:
                    similarity = float(similarity)
                except:
                    similarity = 0.0
            
            formatted.append({
                'filename': clean_name,
                'similarity': float(similarity),
                'excerpt': useful_excerpt,
                'source': clean_name,
                'content': content,  # Full content for LLM
                'title': clean_name,  # Alias for consistency
            })
        
        return formatted
    
    @staticmethod
    def format_code(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format code search results consistently.
        
        Input: List of raw code symbol dicts from CodeGraph
        Output: Standardized format for frontend/LLM
        """
        formatted = []
        
        for r in raw_results:
            formatted.append({
                'symbol_name': (
                    r.get('symbol') or 
                    r.get('symbol_name') or 
                    r.get('name') or 
                    'Unknown'
                ),
                'symbol_type': (
                    r.get('type') or 
                    r.get('symbol_type') or 
                    'code'
                ),
                'file_path': (
                    r.get('file') or 
                    r.get('file_path') or 
                    r.get('path') or 
                    ''
                ),
                'line_number': (
                    r.get('line_number') or 
                    r.get('line') or 
                    0
                ),
                'code': (
                    r.get('code') or 
                    r.get('code_snippet') or 
                    r.get('content') or 
                    ''
                ),
                'language': r.get('language', 'python'),
                'docstring': (
                    r.get('docstring') or 
                    r.get('description') or 
                    ''
                ),
            })
        
        return formatted
    
    @staticmethod
    def format_database(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format database query results consistently.
        
        Input: Raw database query result
        Output: Standardized format
        """
        return {
            'success': result.get('success', False) or result.get('status') == 'success',
            'sql': result.get('sql', ''),
            'row_count': result.get('row_count', 0),
            'data': result.get('data', []),
            'columns': result.get('columns', []),
            'execution_time_ms': result.get('execution_time_ms', 0),
            'pandas_ai': result.get('pandas_ai'),  # Optional insight
        }
    
    @staticmethod
    def standardize_result(
        raw_result: Dict[str, Any],
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Standardize any tool result to common structure.
        
        Always returns:
        {
            "success": bool,
            "status": "success" | "error",
            "results": List[Dict],  # Always plural, always a list
            "metadata": Dict,
            "error": Optional[str]
        }
        """
        # Extract success status (handle multiple formats)
        success = (
            raw_result.get('success') or 
            raw_result.get('status') == 'success' or
            (raw_result.get('status') and 'error' not in raw_result.get('status', '').lower())
        )
        
        # Extract results (handle both singular and plural)
        raw_results = raw_result.get('results') or raw_result.get('result') or []
        if not isinstance(raw_results, list):
            raw_results = [raw_results] if raw_results else []
        
        # Format based on tool type
        formatted_results = raw_results
        if raw_results:
            if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
                formatted_results = ToolResultFormatter.format_documents(raw_results)
            elif tool_name in ['search_codebase', 'search_code']:
                formatted_results = ToolResultFormatter.format_code(raw_results)
            elif tool_name in ['query_database', 'smart_query_database']:
                return ToolResultFormatter.format_database(raw_result)
        
        # Return standardized structure
        return {
            "success": success,
            "status": "success" if success else "error",
            "results": formatted_results,
            "metadata": {
                "tool": tool_name,
                "count": len(formatted_results),
                "original_keys": list(raw_result.keys()),
            },
            "error": raw_result.get('error') if not success else None,
            # Preserve original data for backward compatibility
            **{k: v for k, v in raw_result.items() if k not in ['success', 'status', 'results', 'result', 'error']}
        }
    
    @staticmethod
    def format_for_frontend(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Format tool result specifically for frontend artifact viewer.
        
        Returns structure expected by frontend components.
        """
        standardized = ToolResultFormatter.standardize_result(result, tool_name)
        
        frontend_data = {}
        
        if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
            # Group documents by source file
            docs_by_source = {}
            for doc in standardized['results']:
                source = doc.get('filename', doc.get('source', 'Unknown'))
                content = doc.get('content', '')
                excerpt = doc.get('excerpt', '')
                similarity = doc.get('similarity', 0.0)
                
                # Use the source (filename) directly - this is the actual file on disk
                # Build correct path
                file_path = f"/var/automatos/documents/{source}"
                
                if source not in docs_by_source:
                    docs_by_source[source] = {
                        'source': source,
                        'file_path': file_path,
                        'chunks': [],
                        'max_similarity': 0.0,
                        'title': source.replace('.md', '').replace('.pdf', '').replace('-', ' ').replace('_', ' ').title()
                    }
                
                docs_by_source[source]['chunks'].append({
                    'content': content,
                    'excerpt': excerpt
                })
                docs_by_source[source]['max_similarity'] = max(
                    docs_by_source[source]['max_similarity'],
                    similarity
                )
            
            # Sort by relevance and convert to list
            grouped_docs = sorted(
                docs_by_source.values(),
                key=lambda d: d['max_similarity'],
                reverse=True
            )
            
            # Format for frontend display
            frontend_data['documents'] = [
                {
                    'filename': doc['source'],
                    'title': doc['title'],
                    'file_path': doc['file_path'],
                    'relevance': int(doc['max_similarity'] * 100),
                    'chunk_count': len(doc['chunks']),
                    'preview': doc['chunks'][0]['excerpt'] if doc['chunks'] else '',
                    'download_url': f"/api/documents/download?path={doc['file_path']}",
                    # NEW: Add full content for artifact viewer
                    'full_content': '\n\n'.join([chunk['content'] for chunk in doc['chunks']]),
                    # NEW: Provide chunk list for RAG chunk inspector UI
                    'chunks': doc['chunks'],
                }
                for doc in grouped_docs
            ]
        
        elif tool_name in ['search_codebase', 'search_code']:
            frontend_data['code_snippets'] = standardized['results']
        
        elif tool_name in ['query_database', 'smart_query_database']:
            # Frontend expects database_results as an array with pandas_ai inside
            status = result.get('status')
            db_result = {
                'database': result.get('database', 'Database'),
                'status': status,
                'sql': result.get('sql', ''),
                'row_count': result.get('row_count', 0),
                'execution_time_ms': result.get('execution_time_ms', 0),
                'data': result.get('data', []),
                'columns': result.get('columns', []),
                # Smart NL2SQL extras (when present)
                'explanation': result.get('explanation'),
                'rephrased_query': result.get('rephrased_query'),
                'visualization': result.get('visualization'),
                'follow_up_questions': result.get('follow_up_questions'),
                # Clarification flow (when present)
                'clarifications': result.get('clarifications'),
                'clarification_answers': result.get('clarification_answers'),
                'original_query': result.get('original_query'),
                'message': result.get('message'),
            }
            
            # Include PandasAI visualization inside the result
            pandas_ai = result.get('pandas_ai', {})
            if pandas_ai:
                db_result['pandas_ai'] = pandas_ai
            
            # Frontend expects an array of results
            frontend_data['database_results'] = [db_result]
        
        return frontend_data
    
    @staticmethod
    def format_for_llm(result: Dict[str, Any], tool_name: str, max_chars: int = 3000) -> str:
        """
        Format tool result for LLM context (truncated summary).
        
        Full data goes to frontend, summary goes to LLM.
        """
        standardized = ToolResultFormatter.standardize_result(result, tool_name)
        
        if not standardized['success']:
            return f"Tool {tool_name} failed: {standardized.get('error', 'Unknown error')}"
        
        summary_parts = [f"Tool: {tool_name}"]
        summary_parts.append(f"Status: {standardized['status']}")
        summary_parts.append(f"Results: {standardized['metadata']['count']} items")
        
        # Add preview of results
        results = standardized['results'][:3]  # Top 3 only
        
        if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
            summary_parts.append(
                "NOTE: The UI will render clickable document cards and chunk inspectors. "
                "Do NOT list filenames/links in your final answer; use the excerpts below only to explain the topic."
            )
            for i, doc in enumerate(results, start=1):
                excerpt = (doc.get('excerpt', '') or '')[:450]
                score = float(doc.get('similarity', 0) or 0) * 100.0
                # Avoid leaking filenames to the LLM (it tends to echo them back as a list)
                summary_parts.append(f"\n[Source {i}] ({score:.1f}%)")
                summary_parts.append(excerpt)
        
        elif tool_name in ['search_codebase', 'search_code']:
            for code in results:
                summary_parts.append(f"\n💻 {code.get('symbol_name', 'Code')} ({code.get('file_path', 'unknown')})")
                summary_parts.append(f"```{code.get('language', 'python')}\n{code.get('code', '')[:400]}\n```")
        
        elif tool_name in ['query_database', 'smart_query_database']:
            summary_parts.append(f"\n🗄️ SQL: {standardized.get('sql', '')[:300]}")
            summary_parts.append(f"Total Rows: {standardized.get('row_count', 0)}")
            
            # Include ALL data (already limited at query level) - don't truncate here
            all_data = standardized.get('data', [])
            if all_data:
                summary_parts.append(f"Complete data ({len(all_data)} rows):")
                summary_parts.append(json.dumps(all_data, default=str, indent=2)[:2000])
            
            # Include PandasAI insight if available
            pandas_insight = standardized.get('pandas_ai', {})
            if pandas_insight:
                summary_parts.append(f"\n📊 AI Analysis: {pandas_insight.get('summary', '')}")
                if pandas_insight.get('charts'):
                    summary_parts.append("(Chart generated - see visualization)")
        
        full_summary = "\n".join(summary_parts)
        
        # Truncate if too long
        if len(full_summary) > max_chars:
            full_summary = full_summary[:max_chars] + "..."
        
        return full_summary

