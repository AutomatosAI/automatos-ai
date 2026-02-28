"""
Unified Tool Result Formatter
=============================

SINGLE SOURCE OF TRUTH for formatting tool results.
All services use this - NO MORE DUPLICATION.
"""

import json
import logging
import os
from pathlib import Path
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
    def _download_text_from_s3(s3_uri: str) -> str:
        """Download a text file from S3 given an s3:// URI."""
        try:
            import boto3
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            if not key:
                return ""

            s3_client = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            )
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            logger.info(f"[FullContent] ✅ Downloaded original from S3 ({len(content)} chars): {key}")
            return content
        except Exception as e:
            logger.warning(f"[FullContent] S3 download failed for {s3_uri}: {e}")
            return ""

    # File types that can be read as plain text from S3
    _TEXT_EXTENSIONS = frozenset({
        '.md', '.txt', '.json', '.csv', '.yaml', '.yml',
        '.xml', '.html', '.htm', '.py', '.js', '.ts', '.tsx',
        '.jsx', '.go', '.rs', '.java', '.rb', '.sh', '.toml',
    })

    _db_engine = None  # cached SQLAlchemy engine

    @staticmethod
    def _get_db_engine():
        """Return a cached SQLAlchemy engine (created once, reused)."""
        if ToolResultFormatter._db_engine is None:
            from sqlalchemy import create_engine
            from config import config as app_config
            db_url = getattr(app_config, "DATABASE_URL", None)
            if not db_url:
                db_url = os.getenv("DATABASE_URL")
            if not db_url:
                logger.warning("[FullContent] No DATABASE_URL — cannot fetch document content")
                return None
            ToolResultFormatter._db_engine = create_engine(db_url)
        return ToolResultFormatter._db_engine

    @staticmethod
    def _fetch_document_content_from_db(document_id: int) -> str:
        """
        Fetch full document content:
        1. Try downloading the original file from S3 (perfect formatting)
        2. Fall back to reassembling chunks from document_chunks table
        """
        logger.info(f"[FullContent] Fetching full document for document_id={document_id}")
        try:
            from sqlalchemy import text
            engine = ToolResultFormatter._get_db_engine()
            if engine is None:
                return ""

            # Step 1: Try to get original file from S3
            with engine.connect() as conn:
                doc_row = conn.execute(
                    text("SELECT file_path, filename FROM documents WHERE id = :doc_id"),
                    {"doc_id": document_id},
                ).fetchone()

            if doc_row and doc_row[0] and str(doc_row[0]).startswith("s3://"):
                file_path = str(doc_row[0])
                filename = str(doc_row[1] or "").lower()

                # Only download text-based files (not PDFs, DOCX, etc.)
                ext = Path(filename).suffix.lower() if filename else ""
                if ext in ToolResultFormatter._TEXT_EXTENSIONS:
                    s3_content = ToolResultFormatter._download_text_from_s3(file_path)
                    if s3_content:
                        return s3_content
                    logger.info(f"[FullContent] S3 download failed, falling back to chunk reassembly")
                else:
                    logger.info(f"[FullContent] Non-text file ({ext}), using chunk reassembly")

            # Step 2: Fall back to chunk reassembly
            with engine.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT content FROM document_chunks
                        WHERE document_id = :doc_id
                        ORDER BY (metadata->>'chunk_index')::int NULLS LAST, id
                    """),
                    {"doc_id": document_id},
                ).fetchall()

            if rows:
                full_content = "\n\n".join(row[0] for row in rows if row[0])
                logger.info(f"[FullContent] ✅ Assembled {len(rows)} chunks ({len(full_content)} chars) for doc {document_id}")
                return full_content
            logger.warning(f"[FullContent] No content found for document_id={document_id}")
            return ""
        except Exception as e:
            logger.warning(f"[FullContent] Failed for doc {document_id}: {e}", exc_info=True)
            return ""

    @staticmethod
    def _fetch_document_content(file_path: str, max_size_kb: int = 500) -> str:
        """
        Fetch full document content from file system (legacy fallback).
        
        Args:
            file_path: Full path to document file
            max_size_kb: Maximum file size to read (default 500KB)
            
        Returns:
            Full document content or empty string if error
        """
        try:
            requested_path = ToolResultFormatter._resolve_document_path(file_path)
            if not requested_path:
                return ""
            
            # Check file exists and size
            if not os.path.exists(requested_path):
                logger.warning(f"Document not found: {requested_path}")
                return ""
            
            file_size_kb = os.path.getsize(requested_path) / 1024
            if file_size_kb > max_size_kb:
                logger.warning(f"Document too large ({file_size_kb:.1f}KB): {requested_path}")
                return f"[Document too large to display: {file_size_kb:.1f}KB. Use download link instead.]"
            
            # Read file content
            with open(requested_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error reading document {file_path}: {e}")
            return ""

    @staticmethod
    def _document_roots() -> List[Path]:
        """
        Allowed document roots for artifact rendering.

        We keep the original `/var/automatos/documents` default, but also support
        local-repo docs for dev (e.g. `automatos-ai/docs/**`) without hardcoding an
        absolute machine-specific path.
        """
        roots: List[Path] = []

        env_roots = (
            os.getenv("AUTOMATOS_DOCUMENTS_DIRS")
            or os.getenv("AUTOMATOS_DOCUMENTS_DIR")
            or os.getenv("DOCUMENTS_DIR")
        )
        if env_roots:
            for raw in env_roots.split(","):
                p = raw.strip()
                if not p:
                    continue
                try:
                    roots.append(Path(p).expanduser().resolve())
                except Exception:
                    continue

        # Default prod path
        try:
            roots.append(Path("/var/automatos/documents").resolve())
        except Exception:
            pass

        # Dev-friendly repo-relative fallbacks
        # result_formatter.py -> formatting -> tools -> modules -> orchestrator -> automatos-ai
        try:
            automatos_ai_root = Path(__file__).resolve().parents[4]
            docs_root = (automatos_ai_root / "docs").resolve()
            roots.append(docs_root)
            roots.append((docs_root / "PRDS").resolve())
        except Exception:
            pass

        # De-dupe while preserving order
        seen = set()
        out: List[Path] = []
        for r in roots:
            if str(r) in seen:
                continue
            seen.add(str(r))
            out.append(r)
        return out

    @staticmethod
    def _is_under_allowed_root(path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        for root in ToolResultFormatter._document_roots():
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _resolve_document_path(file_path_or_name: str) -> Optional[Path]:
        """
        Resolve a document path safely.

        Accepts:
        - absolute paths (must be under allowed roots)
        - bare filenames (resolved under allowed roots)
        """
        if not file_path_or_name:
            return None

        raw = str(file_path_or_name).strip()
        if not raw:
            return None

        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            if ToolResultFormatter._is_under_allowed_root(candidate):
                return candidate.resolve()
            logger.warning(f"Access denied: {raw} not under allowed document roots")
            return None

        # Treat as filename under allowed roots
        filename = candidate.name
        for root in ToolResultFormatter._document_roots():
            p = (root / filename)
            if p.exists() and p.is_file():
                return p.resolve()

        # Also allow one-level relative paths under docs roots (e.g. PRDS/foo.md)
        for root in ToolResultFormatter._document_roots():
            p = (root / candidate)
            if p.exists() and p.is_file():
                return p.resolve()

        # Not found
        return None
    
    @staticmethod
    def _detect_content_type(filename: str) -> str:
        """Detect content type from filename for proper rendering."""
        if not filename:
            return 'text'
        
        filename_lower = filename.lower()
        if filename_lower.endswith('.md'):
            return 'markdown'
        elif filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith('.json'):
            return 'json'
        elif filename_lower.endswith(('.py', '.js', '.ts', '.tsx', '.jsx')):
            return 'code'
        elif filename_lower.endswith('.txt'):
            return 'text'
        else:
            return 'text'
    
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
                'document_id': r.get('document_id'),
                'metadata': r.get('metadata', {}),
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
            file_path = (
                r.get('file') or
                r.get('file_path') or
                r.get('path') or
                ''
            )
            # Detect language from file extension
            lang = r.get('language', '')
            if not lang and file_path:
                ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
                lang_map = {'py': 'python', 'ts': 'typescript', 'tsx': 'typescript',
                           'js': 'javascript', 'jsx': 'javascript', 'go': 'go',
                           'rs': 'rust', 'java': 'java', 'rb': 'ruby'}
                lang = lang_map.get(ext, ext or 'python')

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
                'file_path': file_path,
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
                'language': lang,
                'docstring': (
                    r.get('docstring') or
                    r.get('description') or
                    ''
                ),
                'signature': r.get('signature', ''),
                'explanation': r.get('explanation', ''),
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

        # Composio returns payload under `data` (not `results`). If we don't normalize this,
        # the LLM sees "0 items" and assumes nothing happened.
        if (not raw_results) and (tool_name == "composio_execute" or tool_name.startswith("composio_")):
            data = raw_result.get("data")
            if isinstance(data, list):
                raw_results = data
            elif isinstance(data, dict):
                # Common API shapes: {"items":[...]}, {"messages":[...]}, {"threads":[...]} etc.
                for k in ("results", "items", "messages", "emails", "threads", "data"):
                    v = data.get(k)
                    if isinstance(v, list):
                        raw_results = v
                        break
                if not raw_results:
                    # Preserve structured payload as a single result item
                    raw_results = [data]
            elif data is not None:
                raw_results = [data]

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
            # Group chunks by document_id (not filename — filenames from S3 Vectors can be temp names)
            docs_by_id = {}
            for doc in standardized['results']:
                source = doc.get('filename', doc.get('source', 'Unknown'))
                content = doc.get('content', '')
                excerpt = doc.get('excerpt', '')
                similarity = doc.get('similarity', 0.0)
                document_id = doc.get('document_id')
                logger.info(f"[FrontendData] Chunk: source={source}, document_id={document_id}")

                # Group by document_id when available, fall back to filename
                group_key = document_id if document_id else source

                if group_key not in docs_by_id:
                    docs_by_id[group_key] = {
                        'source': source,
                        'document_id': document_id,
                        'chunks': [],
                        'max_similarity': 0.0,
                        'title': source.replace('.md', '').replace('.pdf', '').replace('-', ' ').replace('_', ' ').title()
                    }

                docs_by_id[group_key]['chunks'].append({
                    'content': content,
                    'excerpt': excerpt,
                    'similarity': similarity,
                })
                docs_by_id[group_key]['max_similarity'] = max(
                    docs_by_id[group_key]['max_similarity'],
                    similarity
                )

            # Sort by relevance and convert to list
            grouped_docs = sorted(
                docs_by_id.values(),
                key=lambda d: d['max_similarity'],
                reverse=True
            )

            # Fetch full document content from document_chunks table (not local filesystem)
            frontend_data['documents'] = []
            for doc in grouped_docs:
                document_id = doc.get('document_id')
                full_content = ""

                # Reconstruct full document from all chunks in DB
                if document_id:
                    full_content = ToolResultFormatter._fetch_document_content_from_db(document_id)

                # Fall back to concatenating matched chunks if DB fetch fails
                if not full_content:
                    full_content = "\n\n".join(c['content'] for c in doc['chunks'])

                frontend_data['documents'].append({
                    'filename': doc['source'],
                    'title': doc['title'],
                    'document_id': document_id,
                    'relevance': int(doc['max_similarity'] * 100),
                    'chunk_count': len(doc['chunks']),
                    'preview': doc['chunks'][0].get('excerpt') or doc['chunks'][0].get('content', '')[:200] if doc['chunks'] else '',
                    'download_url': f"/api/documents/{document_id}/download" if document_id else None,
                    'full_content': full_content,
                    'content': full_content,
                    'has_full_content': bool(document_id and full_content),
                    'content_type': ToolResultFormatter._detect_content_type(doc['source']),
                    'chunks': doc['chunks'],
                })
        
        elif tool_name in ['search_codebase', 'search_code']:
            frontend_data['code_snippets'] = standardized['results']
            logger.info(f"[FrontendData] search_codebase: {len(standardized['results'])} code snippets")
        
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

        elif tool_name == "execute_command":
            # Create terminal widget data for command execution
            logger.info(f"[TerminalWidget] Creating terminal widget for command: {result.get('command', 'N/A')}")
            frontend_data['terminal_output'] = {
                'command': result.get('command', ''),
                'output': result.get('output', result.get('stdout', '')),
                'stderr': result.get('stderr', ''),
                'exitCode': result.get('exit_code', result.get('returncode', 0)),
                'executionTime': result.get('execution_time_ms', 0),
                'workingDirectory': result.get('cwd', result.get('working_directory', '')),
            }
            logger.info(f"[TerminalWidget] Added terminal_output to frontend_data")

        elif tool_name == "composio_execute" or tool_name.startswith("composio_"):
            # Schema-driven widget type detection (PRD-38 Enhancement)
            # Instead of hardcoding "GMAIL" or "OUTLOOK", we detect from response structure
            from modules.tools.formatting.schema_detector import ResponseTypeDetector, GenericDataExtractor

            action = result.get("action", "")
            raw_data = result.get("data", {})
            response_schema = result.get("response_schema")  # May be passed from executor

            logger.info(f"[SchemaDetector] Composio action: {action}, has_schema: {bool(response_schema)}")

            # Detect widget type using schema-driven approach
            # This works for Gmail, Outlook, Yahoo, or ANY email provider
            widget_type = ResponseTypeDetector.detect(
                response_schema=response_schema,
                result=raw_data,
                action_name=action
            )
            logger.info(f"[SchemaDetector] Detected widget type: {widget_type}")

            # Extract data based on detected type (generic, not provider-specific)
            if widget_type == "email":
                emails = GenericDataExtractor.extract_emails(raw_data)
                if emails:
                    frontend_data["emails"] = emails
                    logger.info(f"[SchemaDetector] Extracted {len(emails)} emails for widget")
                    for i, e in enumerate(emails[:2]):
                        logger.info(f"[SchemaDetector] Email {i+1}: subject='{str(e.get('subject', 'N/A'))[:50]}', from='{e.get('from', 'N/A')}'")
                else:
                    logger.warning(f"[SchemaDetector] Email type detected but no valid emails extracted")

            elif widget_type == "data":
                columns, rows = GenericDataExtractor.extract_table_data(raw_data)
                if rows:
                    frontend_data["table_data"] = {
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows)
                    }
                    logger.info(f"[SchemaDetector] Extracted table with {len(columns)} columns, {len(rows)} rows")

            elif widget_type == "file":
                # File data - pass through for FileWidget
                frontend_data["files"] = raw_data.get("files") or raw_data.get("data") or [raw_data]
                logger.info(f"[SchemaDetector] Extracted file data for widget")

            elif widget_type == "calendar":
                # Calendar events - pass through for future CalendarWidget
                frontend_data["events"] = raw_data.get("events") or raw_data.get("items") or raw_data.get("data")
                logger.info(f"[SchemaDetector] Extracted calendar data for widget")

            # Always include generic API results for fallback display
            frontend_data["api_results"] = standardized["results"]
            frontend_data["api_metadata"] = standardized.get("metadata", {})
            frontend_data["detected_type"] = widget_type  # Let frontend know what was detected

        elif tool_name == "generate_document":
            # PRD-63: Document Generation — provide download widget data
            # Raw result is {"success": True, "results": [{"filename": ..., "download_url": ..., "size_kb": ...}]}
            doc_result = (result.get('results') or [{}])[0] if isinstance(result.get('results'), list) else result
            frontend_data['generated_document'] = {
                'filename': doc_result.get('filename', 'document'),
                'format': doc_result.get('format', 'pdf'),
                'download_url': doc_result.get('download_url', ''),
                'preview_url': doc_result.get('preview_url'),
                'size_kb': doc_result.get('size_kb', 0),
                'title': doc_result.get('title', doc_result.get('filename', 'Document')),
                'content': doc_result.get('content', ''),
            }
            logger.info(f"[FrontendData] generate_document: {frontend_data['generated_document']['filename']}")

        return frontend_data
    
    @staticmethod
    def format_for_llm(result: Dict[str, Any], tool_name: str, max_chars: int = 4500) -> str:
        """
        Format tool result for LLM context (truncated summary).

        Full data goes to frontend, summary goes to LLM.
        """
        # Workspace tools return structured data directly from the worker.
        # Bypass standardizer (which expects "results" key) and pass data as-is.
        if tool_name.startswith("workspace_"):
            if not result.get("success"):
                return f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}"
            ws_data = {k: v for k, v in result.items() if k != "success"}
            try:
                ws_json = json.dumps(ws_data, default=str, indent=2)
            except Exception:
                ws_json = str(ws_data)
            llm_text = f"Tool: {tool_name}\nStatus: success\n\n{ws_json}"
            return llm_text[:max_chars]

        # Platform tools return data under custom keys (agents, recipes, etc.)
        # Bypass standardizer which only looks for "results"/"result" keys.
        if tool_name.startswith("platform_"):
            if not result.get("success"):
                return f"Tool {tool_name} failed: {result.get('error', 'Unknown error')}"
            platform_data = {k: v for k, v in result.items() if k != "success"}
            try:
                platform_json = json.dumps(platform_data, default=str, indent=2)
            except Exception:
                platform_json = str(platform_data)
            llm_text = f"Tool: {tool_name}\nStatus: success\n\n{platform_json}"
            return llm_text[:max_chars]

        standardized = ToolResultFormatter.standardize_result(result, tool_name)

        if not standardized['success']:
            return f"Tool {tool_name} failed: {standardized.get('error', 'Unknown error')}"

        summary_parts = [f"Tool: {tool_name}"]
        summary_parts.append(f"Status: {standardized['status']}")
        summary_parts.append(f"Results: {standardized['metadata']['count']} items")
        
        # Add preview of results
        results = standardized['results'][:4]  # Top 4 to fit token limits
        
        if tool_name in ['search_knowledge', 'search_documents', 'semantic_search']:
            summary_parts.append(
                "IMPORTANT: You have been provided with FULL CONTENT excerpts below (up to 800 chars each). "
                "Use this content to write a comprehensive, synthesized response. Do NOT just say 'I found documents' - "
                "actually use the content to answer the question thoroughly. The UI will show document cards separately."
            )
            for i, doc in enumerate(results, start=1):
                excerpt = (doc.get('excerpt', '') or '')[:600]
                score = float(doc.get('similarity', 0) or 0) * 100.0
                # Avoid leaking filenames to the LLM (it tends to echo them back as a list)
                summary_parts.append(f"\n[Source {i}] ({score:.1f}%)")
                summary_parts.append(excerpt)
        
        elif tool_name in ['search_codebase', 'search_code']:
            for code in results:
                summary_parts.append(f"\n💻 {code.get('symbol_name', 'Code')} ({code.get('file_path', 'unknown')})")
                summary_parts.append(f"```{code.get('language', 'python')}\n{code.get('code', '')[:500]}\n```")
        
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

        elif tool_name == "composio_execute" or tool_name.startswith("composio_"):
            # Provide a compact, structured preview of external API results
            # so the LLM can actually answer using the returned data.
            preview = standardized["results"][:3]
            logger.info(f"[LLM-Context] Composio results count: {len(standardized['results'])}, preview items: {len(preview)}")
            if preview:
                try:
                    preview_json = json.dumps(preview, default=str, indent=2)[:1800]
                    logger.info(f"[LLM-Context] Composio preview (first 500 chars): {preview_json[:500]}")
                except Exception as e:
                    preview_json = str(preview)[:1800]
                    logger.warning(f"[LLM-Context] JSON dump failed: {e}")
                summary_parts.append("\nAPI result preview (use this to answer):")
                summary_parts.append(preview_json)
            else:
                logger.warning("[LLM-Context] Composio returned 0 items - LLM will hallucinate!")
                summary_parts.append("\nAPI returned 0 items for this query.")

        elif tool_name == "generate_document":
            doc_result = (result.get('results') or [{}])[0] if isinstance(result.get('results'), list) else result
            filename = doc_result.get('filename', 'document')
            fmt = doc_result.get('format', 'pdf')
            size_kb = doc_result.get('size_kb', 0)
            download_url = doc_result.get('download_url', '')
            summary_parts.append(f"\nGenerated {fmt.upper()} document: {filename} ({size_kb} KB)")
            summary_parts.append(f"Download URL: {download_url}")
            summary_parts.append("IMPORTANT: Show the download link to the user using the exact URL above. Do NOT invent document:// links.")

        full_summary = "\n".join(summary_parts)
        
        # Truncate if too long
        if len(full_summary) > max_chars:
            full_summary = full_summary[:max_chars] + "..."
        
        return full_summary

