"""
CodeGraph Service - Code Indexing and Search
===========================================

Indexes code repositories (GitHub) and provides intelligent code search.
Extracts symbols (functions, classes) and builds relationship graphs.
"""

import os
import re
import ast
import json
import hashlib
import logging
import tempfile
import shutil
import builtins
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import time

from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, etc.)"""
    symbol_type: str  # 'function', 'class', 'interface', 'variable'
    name: str
    qualified_name: str
    file_path: str
    line_number: int
    signature: str
    docstring: Optional[str]
    code_snippet: str
    metadata: Dict[str, Any]


@dataclass
class CodeRelationship:
    """Represents a relationship between code symbols"""
    from_symbol: str  # qualified name
    to_symbol: str    # qualified name
    relationship_type: str  # 'calls', 'imports', 'extends', 'implements', 'references'
    metadata: Dict[str, Any]


@dataclass
class ParseResult:
    """Result of parsing a single file"""
    symbols: List[CodeSymbol]
    relationships: List[CodeRelationship]  # NEW
    file_hash: str
    lines_of_code: int
    language: str


class CodeGraphService:
    """
    CodeGraph service for indexing and searching code repositories
    """
    
    def __init__(self, db: Session):
        self.db = db
        
        # Use centralized embedding manager (reads from General Settings)
        from core.llm import create_embedding_manager
        self.embedding_manager = create_embedding_manager()
        logger.info(f"CodeGraphService using {self.embedding_manager.get_provider_info()['provider']} embeddings")
        
        # NEW: Initialize EnhancedVectorStore for centralized vector search
        self._vector_store = None
        try:
            from modules.search import EnhancedVectorStore
            import os
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/automatos")
            self._vector_store = EnhancedVectorStore(
                database_url=db_url,
                table_name="codegraph_symbols"  # Use CodeGraph's existing table
            )
            logger.info("✅ CodeGraphService using EnhancedVectorStore for semantic search")
        except Exception as e:
            logger.warning(f"EnhancedVectorStore not available, using fallback: {e}")
        
        # Initialize tree-sitter parser (PRD-62: 14+ language support)
        self._treesitter_parser = None
        try:
            from .parsers.treesitter_parser import TreeSitterParser
            self._treesitter_parser = TreeSitterParser()
            if self._treesitter_parser.available:
                logger.info("CodeGraphService using tree-sitter for multi-language parsing (14+ languages)")
            else:
                logger.warning("TreeSitterParser created but tree-sitter not available, using fallback parsers")
                self._treesitter_parser = None
        except Exception as e:
            logger.warning(f"TreeSitterParser not available, using legacy parsers: {e}")

        # Supported file extensions (expanded with tree-sitter support)
        if self._treesitter_parser and self._treesitter_parser.available:
            # Build from tree-sitter's supported languages
            self.language_extensions = {}
            for ext, lang in self._treesitter_parser.SUPPORTED_LANGUAGES.items():
                if lang not in self.language_extensions:
                    self.language_extensions[lang] = []
                self.language_extensions[lang].append(ext)
        else:
            # Fallback: original 3-language support
            self.language_extensions = {
                'python': ['.py'],
                'typescript': ['.ts', '.tsx'],
                'javascript': ['.js', '.jsx']
            }
        
        # Default exclude patterns
        self.default_excludes = [
            'node_modules', '__pycache__', '.git', '.venv', 'venv',
            'dist', 'build', '.next', 'coverage', '.pytest_cache',
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.egg-info'
        ]
    
    async def index_github_project(
        self,
        project_name: str,
        github_url: str,
        branch: str = 'main',
        auth_token: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
        workspace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a GitHub repository
        
        Args:
            project_name: Unique project identifier
            github_url: GitHub repository URL
            branch: Branch to index (default: main)
            auth_token: GitHub auth token for private repos
            exclude_patterns: Additional patterns to exclude
            
        Returns:
            Indexing results with statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting GitHub indexing: {project_name} from {github_url}")
        
        # Check if project already exists in this workspace
        existing = self.db.execute(
            text("SELECT id, status, updated_at FROM codegraph_projects WHERE name = :name AND workspace_id = :workspace_id"),
            {"name": project_name, "workspace_id": workspace_id}
        ).fetchone()
        
        if existing and existing.status == 'indexing':
            # If project is stuck in indexing, reset it to allow re-indexing
            # This handles cases where previous indexing failed or was interrupted
            logger.warning(f"Project '{project_name}' is stuck in 'indexing' state. Resetting to allow re-indexing.")
            self.db.execute(
                text("UPDATE codegraph_projects SET status = 'pending' WHERE id = :id"),
                {"id": existing.id}
            )
            self.db.commit()
        
        # Create or update project record
        if existing:
            project_id = existing.id
            # PRD-62: Incremental indexing — load existing file hashes for comparison
            self._existing_file_hashes = {}
            try:
                rows = self.db.execute(
                    text("SELECT file_path, file_hash FROM codegraph_files WHERE project_id = :id"),
                    {"id": project_id}
                ).fetchall()
                self._existing_file_hashes = {r.file_path: r.file_hash for r in rows}
                logger.info(f"Loaded {len(self._existing_file_hashes)} existing file hashes for incremental indexing")
            except Exception as e:
                logger.warning(f"Could not load file hashes, will do full re-index: {e}")

            # Only delete symbols for files that have changed (determined during parse loop)
            # Don't do a full wipe anymore — incremental indexing handles per-file updates
            self.db.execute(
                text("UPDATE codegraph_projects SET status = 'indexing', updated_at = NOW() WHERE id = :id"),
                {"id": project_id}
            )
        else:
            result = self.db.execute(
                text("""
                    INSERT INTO codegraph_projects (name, source_type, source_url, branch, status, exclude_patterns, workspace_id)
                    VALUES (:name, 'github', :url, :branch, 'indexing', CAST(:excludes AS json), :workspace_id)
                    RETURNING id
                """),
                {
                    "name": project_name,
                    "url": github_url,
                    "branch": branch,
                    "excludes": json.dumps(exclude_patterns or []),
                    "workspace_id": workspace_id
                }
            )
            project_id = result.fetchone()[0]

        # Commit project creation/update before DDL operations
        self.db.commit()

        # Ensure embedding column dimension matches current embedding model
        # (may issue ALTER TABLE DDL — must be in its own transaction)
        try:
            self._ensure_embedding_dimension()
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.warning(f"Could not ensure embedding dimension: {e}")
        
        try:
            # Clone repository to temp directory
            repo_path = await self._clone_github_repo(github_url, branch, auth_token)
            
            # Merge exclude patterns
            all_excludes = self.default_excludes + (exclude_patterns or [])
            
            # Discover and parse files
            symbols_data = []
            relationships_data = []  # NEW: Collect relationships
            files_data = []
            total_files = 0
            
            skipped_files = 0
            discovered_paths = set()

            for file_path in self._discover_files(repo_path, all_excludes):
                try:
                    rel_path = os.path.relpath(file_path, repo_path)
                    discovered_paths.add(rel_path)

                    # PRD-62: Incremental indexing — skip unchanged files
                    existing_hashes = getattr(self, '_existing_file_hashes', {})
                    if existing_hashes:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content_check = f.read()
                        new_hash = hashlib.sha256(content_check.encode()).hexdigest()
                        if existing_hashes.get(rel_path) == new_hash:
                            skipped_files += 1
                            total_files += 1
                            continue
                        # File changed — delete old symbols/relationships for this file
                        self.db.execute(
                            text("""
                                DELETE FROM codegraph_relationships WHERE project_id = :pid
                                AND (from_symbol_id IN (SELECT id FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp)
                                  OR to_symbol_id IN (SELECT id FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp))
                            """),
                            {"pid": project_id, "fp": rel_path}
                        )
                        self.db.execute(
                            text("DELETE FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp"),
                            {"pid": project_id, "fp": rel_path}
                        )

                    parse_result = await self._parse_file(file_path, repo_path)

                    if parse_result:
                        # Store file metadata
                        files_data.append({
                            "project_id": project_id,
                            "file_path": rel_path,
                            "file_hash": parse_result.file_hash,
                            "file_size": os.path.getsize(file_path),
                            "lines_of_code": parse_result.lines_of_code,
                            "language": parse_result.language,
                            "workspace_id": workspace_id
                        })
                        
                        # Store symbols
                        for symbol in parse_result.symbols:
                            symbols_data.append({
                                "project_id": project_id,
                                "symbol_type": symbol.symbol_type,
                                "name": symbol.name,
                                "qualified_name": symbol.qualified_name,
                                "file_path": rel_path,
                                "line_number": symbol.line_number,
                                "signature": symbol.signature,
                                "docstring": symbol.docstring,
                                "code_snippet": symbol.code_snippet,
                                "code_snippet": symbol.code_snippet,
                                "metadata": symbol.metadata,
                                "workspace_id": workspace_id
                            })
                        
                        # NEW: Store relationships
                        for rel in parse_result.relationships:
                            relationships_data.append({
                                "project_id": project_id,
                                "from_symbol": rel.from_symbol,
                                "to_symbol": rel.to_symbol,
                                "relationship_type": rel.relationship_type,
                                "metadata": rel.metadata,
                                "workspace_id": workspace_id
                            })
                        
                        total_files += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
                    continue
            
            # PRD-62: Remove symbols for files that were deleted from repo
            if existing_hashes:
                deleted_paths = set(existing_hashes.keys()) - discovered_paths
                for del_path in deleted_paths:
                    self.db.execute(
                        text("""
                            DELETE FROM codegraph_relationships WHERE project_id = :pid
                            AND (from_symbol_id IN (SELECT id FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp)
                              OR to_symbol_id IN (SELECT id FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp))
                        """),
                        {"pid": project_id, "fp": del_path}
                    )
                    self.db.execute(
                        text("DELETE FROM codegraph_symbols WHERE project_id = :pid AND file_path = :fp"),
                        {"pid": project_id, "fp": del_path}
                    )
                    self.db.execute(
                        text("DELETE FROM codegraph_files WHERE project_id = :pid AND file_path = :fp"),
                        {"pid": project_id, "fp": del_path}
                    )
                if deleted_paths:
                    logger.info(f"Removed {len(deleted_paths)} deleted files from index")

            if skipped_files:
                logger.info(f"Incremental indexing: skipped {skipped_files} unchanged files, re-parsed {total_files - skipped_files} changed files")

            # Batch insert files
            if files_data:
                for file_data in files_data:
                    self.db.execute(
                        text("""
                            INSERT INTO codegraph_files 
                            (project_id, file_path, file_hash, file_size, lines_of_code, language, workspace_id)
                            VALUES (:project_id, :file_path, :file_hash, :file_size, :lines_of_code, :language, :workspace_id)
                            ON CONFLICT (project_id, file_path) DO UPDATE
                            SET file_hash = EXCLUDED.file_hash,
                                indexed_at = NOW()
                        """),
                        file_data
                    )
            
            # Batch insert symbols with embeddings
            if symbols_data:
                await self._store_symbols_with_embeddings(symbols_data)
            
            # NEW: Store relationships (after symbols are stored)
            if relationships_data:
                await self._store_relationships(project_id, relationships_data)
            
            # Update project stats
            duration = time.time() - start_time
            self.db.execute(
                text("""
                    UPDATE codegraph_projects
                    SET status = 'active',
                        total_files = :total_files,
                        total_symbols = :total_symbols,
                        total_relationships = :total_relationships,
                        last_indexed = NOW(),
                        index_duration_seconds = :duration,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {
                    "id": project_id,
                    "total_files": total_files,
                    "total_symbols": len(symbols_data),
                    "total_relationships": len(relationships_data),
                    "duration": duration
                }
            )
            self.db.commit()
            
            # Cleanup temp directory
            shutil.rmtree(repo_path, ignore_errors=True)
            
            logger.info(f"✅ Indexed {project_name}: {total_files} files, {len(symbols_data)} symbols in {duration:.1f}s")
            
            return {
                "project_id": project_id,
                "project_name": project_name,
                "total_files": total_files,
                "total_symbols": len(symbols_data),
                "duration_seconds": round(duration, 2),
                "status": "success"
            }
            
        except Exception as e:
            # Rollback any failed transaction before trying the status update
            try:
                self.db.rollback()
            except Exception:
                pass
            try:
                self.db.execute(
                    text("UPDATE codegraph_projects SET status = 'failed', updated_at = NOW() WHERE id = :id"),
                    {"id": project_id}
                )
                self.db.commit()
            except Exception as update_err:
                logger.warning(f"Could not mark project as failed: {update_err}")

            logger.error(f"Failed to index {project_name}: {e}")
            raise
    
    async def _clone_github_repo(
        self,
        github_url: str,
        branch: str,
        auth_token: Optional[str]
    ) -> str:
        """Clone GitHub repository to temp directory"""
        from git import Repo
        
        temp_dir = tempfile.mkdtemp(prefix="codegraph_")
        
        try:
            # Add auth token to URL if provided
            if auth_token and 'github.com' in github_url:
                # Convert https://github.com/user/repo.git to https://token@github.com/user/repo.git
                github_url = github_url.replace('https://', f'https://{auth_token}@')
            
            logger.info(f"Cloning {github_url} (branch: {branch}) to {temp_dir}")
            
            Repo.clone_from(
                github_url,
                temp_dir,
                branch=branch,
                depth=1  # Shallow clone for speed
            )
            
            return temp_dir
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to clone repository: {e}")
    
    def _discover_files(self, root_dir: str, exclude_patterns: List[str]) -> List[str]:
        """Discover all code files in directory"""
        files = []
        root_path = Path(root_dir)
        
        # Get all supported extensions
        valid_extensions = []
        for extensions in self.language_extensions.values():
            valid_extensions.extend(extensions)
        
        for file_path in root_path.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix not in valid_extensions:
                continue
            
            # Check exclude patterns
            rel_path = str(file_path.relative_to(root_path))
            if any(self._matches_pattern(rel_path, pattern) for pattern in exclude_patterns):
                continue
            
            files.append(str(file_path))
        
        return files
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches exclude pattern"""
        # Simple pattern matching
        if '*' in pattern:
            pattern_regex = pattern.replace('.', '\\.').replace('*', '.*')
            return re.match(pattern_regex, path) is not None
        else:
            return pattern in path
    
    async def _parse_file(self, file_path: str, repo_root: str) -> Optional[ParseResult]:
        """Parse a single file and extract symbols.

        Uses tree-sitter parser (14+ languages) when available,
        falls back to legacy Python AST / regex parsers.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Calculate file hash
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            lines_of_code = len([line for line in content.split('\n') if line.strip()])

            extension = Path(file_path).suffix

            # Try tree-sitter first (PRD-62: 14+ language support)
            if self._treesitter_parser:
                lang_name = self._treesitter_parser.get_language_for_extension(extension)
                if lang_name:
                    ts_result = self._treesitter_parser.parse_file(file_path, content, repo_root)
                    if not ts_result.errors:
                        # Convert tree-sitter results to CodeSymbol/CodeRelationship
                        symbols = [
                            CodeSymbol(
                                symbol_type=s.symbol_type,
                                name=s.name,
                                qualified_name=f"{s.file_path}::{s.name}",
                                file_path=s.file_path,
                                line_number=s.line_number,
                                signature=s.signature,
                                docstring=s.docstring,
                                code_snippet=s.code_snippet,
                                metadata=s.metadata,
                            )
                            for s in ts_result.symbols
                        ]
                        relationships = [
                            CodeRelationship(
                                from_symbol=r.from_symbol,
                                to_symbol=r.to_symbol_name,
                                relationship_type=r.relationship_type,
                                metadata=r.metadata,
                            )
                            for r in ts_result.relationships
                        ]
                        return ParseResult(
                            symbols=symbols,
                            relationships=relationships,
                            file_hash=file_hash,
                            lines_of_code=lines_of_code,
                            language=ts_result.language,
                        )
                    else:
                        logger.debug(f"tree-sitter parse errors for {file_path}, falling back: {ts_result.errors}")

            # Fallback: legacy parsers (Python AST + regex TS/JS)
            language = self._detect_language(extension)

            if language == 'python':
                symbols, relationships = self._parse_python(content, file_path, repo_root)
            elif language in ['typescript', 'javascript']:
                symbols, relationships = self._parse_typescript_javascript(content, file_path, repo_root, language)
            else:
                return None

            return ParseResult(
                symbols=symbols,
                relationships=relationships,
                file_hash=file_hash,
                lines_of_code=lines_of_code,
                language=language
            )

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def _detect_language(self, extension: str) -> Optional[str]:
        """Detect language from file extension"""
        for language, extensions in self.language_extensions.items():
            if extension in extensions:
                return language
        return None
    
    def _parse_python(self, content: str, file_path: str, repo_root: str) -> Tuple[List[CodeSymbol], List[CodeRelationship]]:
        """Parse Python file using AST - extract symbols AND relationships"""
        symbols = []
        relationships = []
        
        try:
            tree = ast.parse(content)
            rel_path = os.path.relpath(file_path, repo_root)
            
            # First pass: Extract all symbols
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function
                    signature = self._get_function_signature(node)
                    docstring = ast.get_docstring(node)
                    code_snippet = ast.get_source_segment(content, node) or ""
                    
                    # Extract function calls within this function
                    calls = self._extract_function_calls(node)
                    
                    symbols.append(CodeSymbol(
                        symbol_type='function',
                        name=node.name,
                        qualified_name=f"{rel_path}::{node.name}",
                        file_path=rel_path,
                        line_number=node.lineno,
                        signature=signature,
                        docstring=docstring,
                        code_snippet=code_snippet[:5000],  # Limit snippet size (increased for full functions)
                        metadata={
                            'args': [arg.arg for arg in node.args.args],
                            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                            'calls': calls  # Track what this function calls
                        }
                    ))
                    
                    # Create 'calls' relationships
                    for called_func in calls:
                        relationships.append(CodeRelationship(
                            from_symbol=f"{rel_path}::{node.name}",
                            to_symbol=called_func,
                            relationship_type='calls',
                            metadata={'line': node.lineno}
                        ))
                
                elif isinstance(node, ast.ClassDef):
                    # Extract class
                    docstring = ast.get_docstring(node)
                    code_snippet = ast.get_source_segment(content, node) or ""
                    
                    # Get base classes
                    bases = [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases]
                    
                    symbols.append(CodeSymbol(
                        symbol_type='class',
                        name=node.name,
                        qualified_name=f"{rel_path}::{node.name}",
                        file_path=rel_path,
                        line_number=node.lineno,
                        signature=f"class {node.name}({', '.join(bases)})",
                        docstring=docstring,
                        code_snippet=code_snippet[:5000],  # Increased for full class definitions
                        metadata={
                            'bases': bases,
                            'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                        }
                    ))
                    
                    # Create 'extends' relationships for base classes
                    for base in bases:
                        if base and base != 'object':
                            relationships.append(CodeRelationship(
                                from_symbol=f"{rel_path}::{node.name}",
                                to_symbol=base,
                                relationship_type='extends',
                                metadata={'line': node.lineno}
                            ))
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Extract imports
                    imports = self._extract_imports(node)
                    for imported in imports:
                        relationships.append(CodeRelationship(
                            from_symbol=rel_path,
                            to_symbol=imported,
                            relationship_type='imports',
                            metadata={'line': node.lineno}
                        ))
        
        except Exception as e:
            logger.warning(f"Failed to parse Python AST: {e}")
        
        return symbols, relationships
    
    def _extract_function_calls(self, function_node: ast.FunctionDef) -> List[str]:
        """Extract all function calls within a function"""
        calls = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    calls.append(node.func.attr)
        return list(set(calls))  # Remove duplicates
    
    def _extract_imports(self, node: ast.AST) -> List[str]:
        """Extract imported modules/functions"""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Generate function signature from AST node"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        return_annotation = ""
        if node.returns:
            return_annotation = f" -> {ast.unparse(node.returns)}"
        
        return f"def {node.name}({', '.join(args)}){return_annotation}"
    
    def _parse_typescript_javascript(
        self,
        content: str,
        file_path: str,
        repo_root: str,
        language: str
    ) -> Tuple[List[CodeSymbol], List[CodeRelationship]]:
        """Parse TypeScript/JavaScript using regex (simplified for MVP)"""
        symbols = []
        relationships = []
        rel_path = os.path.relpath(file_path, repo_root)
        
        # Regex patterns for functions and classes
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\((.*?)\)'
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
        const_function_pattern = r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\((.*?)\)\s*=>'
        
        lines = content.split('\n')
        
        # Extract functions
        for i, line in enumerate(lines, 1):
            # Regular functions
            match = re.search(function_pattern, line)
            if match:
                name, params = match.groups()
                symbols.append(CodeSymbol(
                    symbol_type='function',
                    name=name,
                    qualified_name=f"{rel_path}::{name}",
                    file_path=rel_path,
                    line_number=i,
                    signature=f"function {name}({params})",
                    docstring=None,
                    code_snippet=line[:500],
                    metadata={'language': language}
                ))
            
            # Arrow functions
            match = re.search(const_function_pattern, line)
            if match:
                name, params = match.groups()
                symbols.append(CodeSymbol(
                    symbol_type='function',
                    name=name,
                    qualified_name=f"{rel_path}::{name}",
                    file_path=rel_path,
                    line_number=i,
                    signature=f"const {name} = ({params}) =>",
                    docstring=None,
                    code_snippet=line[:500],
                    metadata={'language': language, 'arrow_function': True}
                ))
            
            # Classes
            match = re.search(class_pattern, line)
            if match:
                name, base = match.groups()
                signature = f"class {name}" + (f" extends {base}" if base else "")
                symbols.append(CodeSymbol(
                    symbol_type='class',
                    name=name,
                    qualified_name=f"{rel_path}::{name}",
                    file_path=rel_path,
                    line_number=i,
                    signature=signature,
                    docstring=None,
                    code_snippet=line[:500],
                    metadata={'language': language, 'extends': base}
                ))
                
                # Create 'extends' relationship if base class exists
                if base:
                    relationships.append(CodeRelationship(
                        from_symbol=f"{rel_path}::{name}",
                        to_symbol=base,
                        relationship_type='extends',
                        metadata={'line': i}
                    ))
            
            # Extract imports (simplified)
            import_match = re.search(r'import\s+.*?\s+from\s+[\'"](.+?)[\'"]', line)
            if import_match:
                imported_module = import_match.group(1)
                relationships.append(CodeRelationship(
                    from_symbol=rel_path,
                    to_symbol=imported_module,
                    relationship_type='imports',
                    metadata={'line': i}
                ))
        
        return symbols, relationships
    
    def _ensure_embedding_dimension(self):
        """Ensure the embedding column dimension matches the current embedding model from database settings"""
        try:
            # Read dimension directly from database settings (no hardcoding)
            from core.llm.embedding_manager import get_system_setting
            dimension_str = get_system_setting("vector_store_dimensions")
            
            if not dimension_str:
                # Fallback to embedding manager if settings not found
                current_dim = self.embedding_manager.get_dimension()
                logger.warning(f"vector_store_dimensions not found in settings, using embedding_manager dimension: {current_dim}")
            else:
                current_dim = int(dimension_str)
                logger.info(f"Read embedding dimension from database settings: {current_dim}")
            
            # Check current column dimension
            result = self.db.execute(
                text("""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = 'codegraph_symbols'::regclass 
                    AND attname = 'embedding'
                """)
            ).fetchone()
            
            if result and result[0]:
                # atttypmod format: -1 means no dimension, positive means dimension
                # For vector types, atttypmod = dimension + 4
                current_col_dim = result[0] - 4 if result[0] > 0 else None
            else:
                current_col_dim = None
            
            if current_col_dim != current_dim:
                logger.info(f"Altering embedding column dimension from {current_col_dim} to {current_dim}")
                
                # Drop the index if it exists
                try:
                    self.db.execute(text("DROP INDEX IF EXISTS idx_codegraph_symbols_embedding"))
                except Exception as e:
                    logger.warning(f"Could not drop index: {e}")
                
                # Alter the column
                # Validate current_dim is a safe integer before interpolating into DDL
                # (DDL statements cannot use bind parameters for type definitions)
                current_dim = int(current_dim)
                if not (1 <= current_dim <= 10000):
                    raise ValueError(f"Invalid embedding dimension: {current_dim}")
                self.db.execute(
                    text("ALTER TABLE codegraph_symbols ALTER COLUMN embedding TYPE vector(" + str(current_dim) + ")")
                )

                # Recreate the index
                try:
                    self.db.execute(
                        text("""
                            CREATE INDEX idx_codegraph_symbols_embedding
                            ON codegraph_symbols
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                        """)
                    )
                except Exception as e:
                    logger.warning(f"Could not recreate index: {e}")
                
                self.db.commit()
                logger.info(f"✅ Updated embedding column dimension to {current_dim} and recreated index")
            else:
                logger.debug(f"Embedding column dimension already correct: {current_dim}")
        except Exception as e:
            logger.warning(f"Could not verify/update embedding dimension: {e}. Continuing anyway...")
    
    async def _store_symbols_with_embeddings(self, symbols_data: List[Dict[str, Any]]):
        """Store symbols in database with embeddings"""
        # Batch generate embeddings
        for i in range(0, len(symbols_data), 100):  # Process in batches of 100
            batch = symbols_data[i:i+100]
            
            # Generate embeddings for batch
            texts = []
            for symbol in batch:
                # Create rich text for embedding
                text_parts = [
                    f"Symbol: {symbol['name']}",
                    f"Type: {symbol['symbol_type']}",
                    f"Signature: {symbol['signature']}"
                ]
                if symbol['docstring']:
                    text_parts.append(f"Documentation: {symbol['docstring']}")
                text_parts.append(f"Code: {symbol['code_snippet'][:500]}")
                
                texts.append("\n".join(text_parts))
            
            try:
                # Generate embeddings using centralized manager
                embeddings = []
                
                # Debug: Check embedding_manager type
                if not self.embedding_manager:
                    raise ValueError("embedding_manager is None")
                
                logger.debug(f"🔍 embedding_manager type: {type(self.embedding_manager)}")
                logger.debug(f"🔍 embedding_manager has generate_embedding: {hasattr(self.embedding_manager, 'generate_embedding')}")
                
                if not hasattr(self.embedding_manager, 'generate_embedding'):
                    raise ValueError(f"embedding_manager has no generate_embedding method. Type: {type(self.embedding_manager)}")
                
                # Check if generate_embedding is callable
                gen_method = getattr(self.embedding_manager, 'generate_embedding', None)
                logger.debug(f"🔍 generate_embedding type: {type(gen_method)}, callable: {callable(gen_method)}")
                logger.debug(f"🔍 generate_embedding value (first 200 chars): {repr(gen_method)[:200]}")
                
                if not callable(gen_method):
                    raise ValueError(
                        f"embedding_manager.generate_embedding is not callable. "
                        f"Type: {type(gen_method)}, Value: {repr(gen_method)[:200]}"
                    )
                
                logger.debug(f"🔍 About to call generate_embedding for {len(texts)} texts")
                for idx, text_content in enumerate(texts):
                    logger.debug(f"🔍 Calling generate_embedding for text {idx+1}/{len(texts)}")
                    embedding = await gen_method(text_content)
                    # Ensure embedding is a list/array, not a string
                    if isinstance(embedding, str):
                        raise ValueError(f"Embedding returned as string instead of array for text {idx+1}")
                    # Convert numpy array to list if needed
                    try:
                        import numpy as np
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                    except ImportError:
                        pass  # numpy not available, assume it's already a list
                    if not isinstance(embedding, (list, tuple)):
                        raise ValueError(f"Embedding is not a list/array: {type(embedding)}")
                    embeddings.append(embedding)
                    logger.debug(f"🔍 Got embedding of length {len(embedding)}, type: {type(embedding)}")
                
                logger.debug(f"🔍 Successfully generated {len(embeddings)} embeddings, now storing...")
                
                # Store symbols with embeddings
                for j, symbol in enumerate(batch):
                    try:
                        embedding = embeddings[j]
                        # Double-check embedding type before conversion
                        if isinstance(embedding, str):
                            raise ValueError(f"Embedding {j} is a string, not an array")
                        # Convert numpy array to list if needed
                        try:
                            import numpy as np
                            if isinstance(embedding, np.ndarray):
                                embedding = embedding.tolist()
                        except ImportError:
                            pass  # numpy not available, assume it's already a list
                        
                        # Use builtin str to avoid shadowing issues from **symbol unpacking
                        # Create embedding_str BEFORE unpacking symbol to avoid any shadowing
                        embedding_str = '[' + ','.join([builtins.str(x) for x in embedding]) + ']'
                        
                        # Prepare params dict BEFORE unpacking symbol to isolate str shadowing
                        params = {
                            "project_id": symbol.get('project_id'),
                            "symbol_type": symbol.get('symbol_type'),
                            "name": symbol.get('name'),
                            "qualified_name": symbol.get('qualified_name'),
                            "file_path": symbol.get('file_path'),
                            "line_number": symbol.get('line_number'),
                            "signature": symbol.get('signature'),
                            "docstring": symbol.get('docstring'),
                            "code_snippet": symbol.get('code_snippet'),
                            "embedding": embedding_str,
                            "metadata": json.dumps(symbol.get('metadata', {})),
                            "workspace_id": symbol.get('workspace_id')
                        }
                        
                        self.db.execute(
                            text("""
                                INSERT INTO codegraph_symbols
                                (project_id, symbol_type, name, qualified_name, file_path, line_number,
                                 signature, docstring, code_snippet, embedding, metadata, workspace_id)
                                VALUES (:project_id, :symbol_type, :name, :qualified_name, :file_path,
                                        :line_number, :signature, :docstring, :code_snippet,
                                        CAST(:embedding AS vector), :metadata, :workspace_id)
                            """),
                            params
                        )
                        logger.debug(f"🔍 Stored symbol {j+1}/{len(batch)}")
                    except Exception as e:
                        import traceback
                        logger.debug(f"🔍 Failed to store symbol {j}: {e}")
                        logger.debug(f"🔍 Traceback: {traceback.format_exc()}")
                        logger.debug(f"🔍 Symbol keys: {list(symbol.keys()) if isinstance(symbol, dict) else 'N/A'}")
                        raise
                
                self.db.commit()
                logger.debug(f"🔍 Successfully committed batch of {len(batch)} symbols")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                # Store without embeddings as fallback
                for symbol in batch:
                    self.db.execute(
                        text("""
                            INSERT INTO codegraph_symbols
                            (project_id, symbol_type, name, qualified_name, file_path, line_number,
                             signature, docstring, code_snippet, metadata)
                            VALUES (:project_id, :symbol_type, :name, :qualified_name, :file_path,
                                    :line_number, :signature, :docstring, :code_snippet, :metadata)
                        """),
                        {
                            **symbol,
                            "metadata": str(symbol.get('metadata', {}))
                        }
                    )
                self.db.commit()
    
    async def _store_relationships(self, project_id: int, relationships_data: List[Dict[str, Any]]):
        """
        Store code relationships in database
        
        Maps qualified symbol names to symbol IDs and creates relationships.
        """
        import json
        
        try:
            # Build TWO maps: qualified_name -> id AND name -> ids
            symbol_map = {}  # qualified_name -> symbol_id
            name_map = {}    # simple name -> [symbol_ids]
            
            result = self.db.execute(
                text("SELECT id, qualified_name, name FROM codegraph_symbols WHERE project_id = :project_id"),
                {"project_id": project_id}
            )
            for row in result:
                symbol_map[row.qualified_name] = row.id
                # Store by simple name too (for fuzzy matching)
                if row.name not in name_map:
                    name_map[row.name] = []
                name_map[row.name].append(row.id)
            
            # Insert relationships (only if both symbols exist)
            relationships_inserted = 0
            skipped_external = 0
            skipped_notfound = 0
            
            # DEBUG: Log first few symbol names for comparison
            if relationships_data:
                logger.info(f"🔍 DEBUG: Sample qualified names in symbol_map: {list(symbol_map.keys())[:3]}")
                sample_rels = [f"{r['from_symbol']} -> {r['to_symbol']}" for r in relationships_data[:3]]
                logger.info(f"🔍 DEBUG: Sample relationship names: {sample_rels}")
            
            matched_fuzzy = 0
            for idx, rel in enumerate(relationships_data):
                # Try exact qualified name match first
                from_symbol_id = symbol_map.get(rel['from_symbol'])
                to_symbol_id = symbol_map.get(rel['to_symbol'])
                
                # If to_symbol not found by qualified name, try fuzzy match by simple name
                if not to_symbol_id:
                    # Extract simple name from qualified or use as-is
                    to_name = rel['to_symbol'].split('::')[-1] if '::' in rel['to_symbol'] else rel['to_symbol']
                    to_candidates = name_map.get(to_name, [])
                    # If only one match, use it (unambiguous)
                    if len(to_candidates) == 1:
                        to_symbol_id = to_candidates[0]
                        matched_fuzzy += 1
                        if matched_fuzzy <= 3:
                            logger.info(f"✅ Fuzzy matched '{rel['to_symbol']}' -> '{to_name}' (id={to_symbol_id})")
               
                # Debug first few
                if idx < 3:
                    logger.info(f"🔍 Rel {idx}: from={rel['from_symbol']} (id={from_symbol_id}), to={rel['to_symbol']} (id={to_symbol_id})")
               
                # Only create relationship if both symbols are found
                # (to_symbol might be external library, base class not in project, etc.)
                if from_symbol_id and to_symbol_id:
                    try:
                        self.db.execute(
                            text("""
                                INSERT INTO codegraph_relationships 
                                (project_id, from_symbol_id, to_symbol_id, relationship_type, metadata, workspace_id)
                                VALUES (:project_id, :from_symbol_id, :to_symbol_id, :relationship_type, CAST(:metadata AS jsonb), :workspace_id)
                            """),
                            {
                                "project_id": project_id,
                                "from_symbol_id": from_symbol_id,
                                "to_symbol_id": to_symbol_id,
                                "relationship_type": rel['relationship_type'],
                                "metadata": json.dumps(rel['metadata']),
                                "workspace_id": rel.get('workspace_id')
                            }
                        )
                        relationships_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert relationship: {e}")
                        continue
                elif from_symbol_id and not to_symbol_id:
                    # PRD-62 Bug #4: Store as external_reference instead of silently skipping
                    to_name = rel['to_symbol'].split('::')[-1] if '::' in rel['to_symbol'] else rel['to_symbol']
                    try:
                        self.db.execute(
                            text("""
                                INSERT INTO codegraph_relationships
                                (project_id, from_symbol_id, to_symbol_id, relationship_type, metadata, workspace_id)
                                VALUES (:project_id, :from_symbol_id, NULL, 'external_reference', CAST(:metadata AS jsonb), :workspace_id)
                            """),
                            {
                                "project_id": project_id,
                                "from_symbol_id": from_symbol_id,
                                "relationship_type": 'external_reference',
                                "metadata": json.dumps({
                                    **rel.get('metadata', {}),
                                    'status': 'unresolved',
                                    'reason': 'not_in_project',
                                    'target_name': to_name,
                                    'original_type': rel['relationship_type'],
                                }),
                                "workspace_id": rel.get('workspace_id')
                            }
                        )
                        relationships_inserted += 1
                    except Exception:
                        pass
                    logger.debug(f"Unresolved reference: {to_name} (from {rel['from_symbol']})")
                    skipped_external += 1
                else:
                    skipped_notfound += 1

            self.db.commit()
            logger.info(f"Stored {relationships_inserted} relationships (from {len(relationships_data)} extracted, {matched_fuzzy} fuzzy-matched, {skipped_external} external refs, {skipped_notfound} unresolved)")
            
        except Exception as e:
            logger.error(f"Failed to store relationships: {e}")
            # Don't raise - relationships are supplementary, don't fail indexing
    
    async def search_symbols(
        self,
        project_name: str,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for code symbols by name (fuzzy matching)
        """
        start_time = time.time()
        
        # Get project ID
        project = self.db.execute(
            text("SELECT id FROM codegraph_projects WHERE name = :name AND workspace_id = :workspace_id"),
            {"name": project_name, "workspace_id": workspace_id}
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project '{project_name}' not found")
        
        project_id = project.id
        
        # Build search query
        # Use two separate fully-parameterized queries to avoid f-string SQL interpolation
        if symbol_type:
            search_query = text("""
                SELECT
                    id,
                    symbol_type,
                    name,
                    qualified_name,
                    file_path,
                    line_number,
                    signature,
                    docstring,
                    code_snippet
                FROM codegraph_symbols
                WHERE project_id = :project_id
                    AND symbol_type = :symbol_type
                    AND (
                        name ILIKE :query
                        OR qualified_name ILIKE :query
                        OR signature ILIKE :query
                    )
                ORDER BY
                    CASE
                        WHEN name = :exact_query THEN 1
                        WHEN name ILIKE :starts_with THEN 2
                        ELSE 3
                    END,
                    name
                LIMIT :limit
            """)
        else:
            search_query = text("""
                SELECT
                    id,
                    symbol_type,
                    name,
                    qualified_name,
                    file_path,
                    line_number,
                    signature,
                    docstring,
                    code_snippet
                FROM codegraph_symbols
                WHERE project_id = :project_id
                    AND (
                        name ILIKE :query
                        OR qualified_name ILIKE :query
                        OR signature ILIKE :query
                    )
                ORDER BY
                    CASE
                        WHEN name = :exact_query THEN 1
                        WHEN name ILIKE :starts_with THEN 2
                        ELSE 3
                    END,
                    name
                LIMIT :limit
            """)
        
        params = {
            "project_id": project_id,
            "query": f"%{query}%",
            "exact_query": query,
            "starts_with": f"{query}%",
            "limit": limit
        }
        
        if symbol_type:
            params["symbol_type"] = symbol_type
        
        results = self.db.execute(search_query, params).fetchall()
        
        # Format results
        symbols = []
        for row in results:
            symbols.append({
                "id": row.id,
                "symbol_type": row.symbol_type,
                "name": row.name,
                "qualified_name": row.qualified_name,
                "file_path": row.file_path,
                "line_number": row.line_number,
                "signature": row.signature,
                "docstring": row.docstring,
                "code_snippet": row.code_snippet
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        # Log query
        self.db.execute(
            text("""
                INSERT INTO codegraph_query_logs
                (project_id, query_type, query_text, results_count, execution_time_ms)
                VALUES (:project_id, 'symbol', :query, :count, :time)
            """),
            {
                "project_id": project_id,
                "query": query,
                "count": len(symbols),
                "time": execution_time
            }
        )
        self.db.commit()
        
        return {
            "project": project_name,
            "query": query,
            "count": len(symbols),
            "results": symbols,
            "execution_time_ms": round(execution_time, 2)
        }
    
    async def semantic_search(
        self,
        project_name: str,
        query: str,
        limit: int = 10,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Semantic search using centralized EnhancedVectorStore.
        
        MIGRATED: Now uses modules.search.EnhancedVectorStore for consistent vector search.
        Old SQL-based implementation kept below for rollback if needed.
        """
        start_time = time.time()
        
        # Get project ID
        project = self.db.execute(
            text("SELECT id FROM codegraph_projects WHERE name = :name AND workspace_id = :workspace_id"),
            {"name": project_name, "workspace_id": workspace_id}
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project '{project_name}' not found")
        
        project_id = project.id
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.generate_embedding(query)
        
        # Use centralized EnhancedVectorStore if available
        if self._vector_store:
            try:
                from modules.search import SearchMode, RankingStrategy, SearchFilter
                
                # Initialize vector store if needed
                if not self._vector_store.pool:
                    await self._vector_store.initialize()
                
                logger.info(f"🔎 Using EnhancedVectorStore for semantic search: limit={limit}")
                
                # Create filter for this project
                search_filter = SearchFilter(
                    metadata_filters={"project_id": str(project_id)}
                )
                
                # Perform search using centralized vector store
                search_results = await self._vector_store.search(
                    query_embedding=query_embedding,
                    mode=SearchMode.SEMANTIC,
                    ranking_strategy=RankingStrategy.SIMILARITY,
                    limit=limit,
                    search_filter=search_filter,
                    query_text=query
                )
                
                # Convert SearchResult objects to CodeGraph's expected format
                symbols = []
                for result in search_results:
                    doc = result.document
                    symbols.append({
                        "id": doc.metadata.get('id', doc.id),
                        "symbol_type": doc.metadata.get('symbol_type', ''),
                        "name": doc.metadata.get('name', ''),
                        "qualified_name": doc.metadata.get('qualified_name', ''),
                        "file_path": doc.metadata.get('file_path', ''),
                        "line_number": doc.metadata.get('line_number', 0),
                        "signature": doc.metadata.get('signature', ''),
                        "docstring": doc.metadata.get('docstring', ''),
                        "code_snippet": doc.content,
                        "similarity": float(result.similarity_score)
                    })
                
                execution_time = (time.time() - start_time) * 1000
                prompt_block = self._format_prompt_block(symbols, query)
                
                # Log query
                self.db.execute(
                    text("""
                        INSERT INTO codegraph_query_logs
                        (project_id, query_type, query_text, results_count, execution_time_ms)
                        VALUES (:project_id, 'semantic', :query, :count, :time)
                    """),
                    {
                        "project_id": project_id,
                        "query": query,
                        "count": len(symbols),
                        "time": execution_time
                    }
                )
                self.db.commit()
                
                logger.info(f"✅ Retrieved {len(symbols)} symbols using EnhancedVectorStore")
                
                return {
                    "project": project_name,
                    "query": query,
                    "count": len(symbols),
                    "results": symbols,
                    "prompt_block": prompt_block,
                    "execution_time_ms": round(execution_time, 2)
                }
                
            except Exception as e:
                logger.warning(f"EnhancedVectorStore search failed, falling back to SQL: {e}")
                # Fall through to SQL fallback below
        
        # FALLBACK: Original SQL-based implementation (kept for rollback)
        logger.info("Using fallback SQL-based semantic search")
        # Validate embedding values are numeric and build safe string for parameterized query
        embedding_str = '[' + ','.join(str(float(v)) for v in query_embedding) + ']'

        # Semantic search using parameterized query with CAST for the embedding vector
        search_query = text("""
            SELECT
                id,
                symbol_type,
                name,
                qualified_name,
                file_path,
                line_number,
                signature,
                docstring,
                code_snippet,
                1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM codegraph_symbols
            WHERE project_id = :project_id
                AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """)
        
        results = self.db.execute(
            search_query,
            {"project_id": project_id, "limit": limit, "embedding": embedding_str}
        ).fetchall()
        
        # Format results
        symbols = []
        for row in results:
            symbols.append({
                "id": row.id,
                "symbol_type": row.symbol_type,
                "name": row.name,
                "qualified_name": row.qualified_name,
                "file_path": row.file_path,
                "line_number": row.line_number,
                "signature": row.signature,
                "docstring": row.docstring,
                "code_snippet": row.code_snippet,
                "similarity": float(row.similarity)
            })
        
        execution_time = (time.time() - start_time) * 1000
        
        # Format as prompt block for LLMs
        prompt_block = self._format_prompt_block(symbols, query)
        
        # Log query
        self.db.execute(
            text("""
                INSERT INTO codegraph_query_logs
                (project_id, query_type, query_text, results_count, execution_time_ms)
                VALUES (:project_id, 'semantic', :query, :count, :time)
            """),
            {
                "project_id": project_id,
                "query": query,
                "count": len(symbols),
                "time": execution_time
            }
        )
        self.db.commit()
        
        logger.info(f"✅ Retrieved {len(symbols)} symbols with SQL fallback")
        
        return {
            "project": project_name,
            "query": query,
            "count": len(symbols),
            "results": symbols,
            "prompt_block": prompt_block,
            "execution_time_ms": round(execution_time, 2)
        }
    
    def _format_prompt_block(self, symbols: List[Dict], query: str) -> str:
        """Format search results as prompt block for LLM"""
        parts = [f"# CodeGraph Search Results\n\nQuery: {query}\n\n"]
        
        for i, symbol in enumerate(symbols, 1):
            parts.append(f"## Result {i}: {symbol['name']} ({symbol['symbol_type']})")
            parts.append(f"File: {symbol['file_path']}:{symbol['line_number']}")
            parts.append(f"Signature: {symbol['signature']}")
            
            if symbol.get('similarity'):
                parts.append(f"Relevance: {symbol['similarity']:.1%}")
            
            if symbol.get('docstring'):
                parts.append(f"\nDocumentation:\n{symbol['docstring']}")
            
            parts.append(f"\nCode:\n```\n{symbol['code_snippet']}\n```\n")
            parts.append("---\n")
        
        return "\n".join(parts)
    
    def list_projects(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all indexed projects for a workspace"""
        
        # Build query specific to workspace
        sql = """
            SELECT
                id,
                name,
                source_type,
                source_url,
                branch,
                language,
                total_files,
                total_symbols,
                total_relationships,
                last_indexed,
                index_duration_seconds,
                status,
                auto_reindex,
                created_at
            FROM codegraph_projects
            WHERE workspace_id = :workspace_id
            ORDER BY last_indexed DESC NULLS LAST
        """
        
        results = self.db.execute(text(sql), {"workspace_id": workspace_id}).fetchall()
        
        projects = []
        for row in results:
            projects.append({
                "id": row.id,
                "name": row.name,
                "source_type": row.source_type,
                "source_url": row.source_url,
                "branch": row.branch,
                "language": row.language,
                "total_files": row.total_files,
                "total_symbols": row.total_symbols,
                "total_relationships": row.total_relationships,
                "last_indexed": row.last_indexed.isoformat() if row.last_indexed else None,
                "index_duration_seconds": row.index_duration_seconds,
                "status": row.status,
                "auto_reindex": row.auto_reindex,
                "created_at": row.created_at.isoformat() if row.created_at else None
            })
        
        return projects
    
    def delete_project(self, project_id: int, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """Delete a project and all associated data"""
        
        # Build query with workspace isolation if provided
        check_sql = "SELECT name, total_files, total_symbols FROM codegraph_projects WHERE id = :id"
        params = {"id": project_id}
        
        if workspace_id:
            check_sql += " AND workspace_id = :workspace_id"
            params["workspace_id"] = workspace_id
            
        # Get project stats before deletion
        project = self.db.execute(
            text(check_sql),
            params
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Delete project (cascades to symbols, relationships, files, logs)
        self.db.execute(
            text("DELETE FROM codegraph_projects WHERE id = :id"),
            {"id": project_id}
        )
        self.db.commit()
        
        return {
            "message": "Project deleted successfully",
            "project_name": project.name,
            "files_removed": project.total_files,
            "symbols_removed": project.total_symbols
        }
    
    async def get_call_graph(
        self,
        project_name: str,
        symbol: str,
        depth: int = 1,
        direction: str = "outgoing",
        workspace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate call graph for a symbol showing function calls and dependencies
        
        Args:
            project_name: Name of the project
            symbol: Qualified symbol name (e.g., "module.py::ClassName::method")
            depth: How many levels to traverse (1-5)
            direction: 'outgoing' (calls made), 'incoming' (callers), 'both'
        """
        # Get project ID
        project = self.db.execute(
            text("SELECT id FROM codegraph_projects WHERE name = :name AND workspace_id = :workspace_id"),
            {"name": project_name, "workspace_id": workspace_id}
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project '{project_name}' not found")
        
        project_id = project.id
        
        # BFS to build call graph
        visited = set()
        nodes = []
        edges = []
        queue = [(symbol, 0)]  # (symbol, level)
        
        while queue:
            current_symbol, level = queue.pop(0)
            
            if current_symbol in visited or level > depth:
                continue
            
            visited.add(current_symbol)
            
            # Get symbol details
            symbol_info = self.db.execute(
                text("""
                    SELECT id, symbol_type, name, qualified_name, file_path, line_number
                    FROM codegraph_symbols
                    WHERE project_id = :project_id AND qualified_name = :symbol
                    LIMIT 1
                """),
                {"project_id": project_id, "symbol": current_symbol}
            ).fetchone()
            
            if not symbol_info:
                continue
            
            # Add node
            node = {
                "id": str(symbol_info.id),
                "symbol": symbol_info.qualified_name,
                "name": symbol_info.name,
                "type": symbol_info.symbol_type,
                "file": symbol_info.file_path,
                "line": symbol_info.line_number,
                "level": level
            }
            nodes.append(node)
            
            # Get relationships
            if direction in ["outgoing", "both"]:
                # Get symbols this one calls
                outgoing = self.db.execute(
                    text("""
                        SELECT s.qualified_name, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON r.to_symbol_id = s.id
                        WHERE r.from_symbol_id = :symbol_id
                          AND r.relationship_type IN ('calls', 'imports', 'extends')
                    """),
                    {"symbol_id": symbol_info.id}
                ).fetchall()
                
                for rel in outgoing:
                    edges.append({
                        "from": current_symbol,
                        "to": rel.qualified_name,
                        "type": rel.relationship_type
                    })
                    
                    if level < depth:
                        queue.append((rel.qualified_name, level + 1))
            
            if direction in ["incoming", "both"]:
                # Get symbols that call this one
                incoming = self.db.execute(
                    text("""
                        SELECT s.qualified_name, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON r.from_symbol_id = s.id
                        WHERE r.to_symbol_id = :symbol_id
                          AND r.relationship_type IN ('calls', 'imports', 'extends')
                    """),
                    {"symbol_id": symbol_info.id}
                ).fetchall()
                
                for rel in incoming:
                    edges.append({
                        "from": rel.qualified_name,
                        "to": current_symbol,
                        "type": rel.relationship_type
                    })
                    
                    if level < depth:
                        queue.append((rel.qualified_name, level + 1))
        
        return {
            "project": project_name,
            "root_symbol": symbol,
            "depth": depth,
            "direction": direction,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges
        }

    # ── PRD-62: Architecture Analysis (US-007) ────────────────────────

    async def analyze_architecture(
        self,
        project_id: int,
        workspace_id: Optional[str] = None,
        focus_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get high-level architecture overview: modules, key symbols, dependency patterns.

        Args:
            project_id: Project to analyze
            workspace_id: Workspace for isolation
            focus_path: Optional directory prefix to focus on
        """
        # Verify project exists
        project = self.db.execute(
            text("SELECT id, name FROM codegraph_projects WHERE id = :id AND workspace_id = :wid"),
            {"id": project_id, "wid": workspace_id}
        ).fetchone()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Fetch symbols (optionally filtered by path)
        if focus_path:
            symbols = self.db.execute(
                text("""
                    SELECT id, name, qualified_name, symbol_type, file_path, line_number
                    FROM codegraph_symbols
                    WHERE project_id = :pid AND workspace_id = :wid AND file_path LIKE :fp
                """),
                {"pid": project_id, "wid": workspace_id, "fp": f"{focus_path}%"}
            ).fetchall()
        else:
            symbols = self.db.execute(
                text("""
                    SELECT id, name, qualified_name, symbol_type, file_path, line_number
                    FROM codegraph_symbols
                    WHERE project_id = :pid AND workspace_id = :wid
                """),
                {"pid": project_id, "wid": workspace_id}
            ).fetchall()

        # Fetch relationships
        relationships = self.db.execute(
            text("""
                SELECT r.from_symbol_id, r.to_symbol_id, r.relationship_type
                FROM codegraph_relationships r
                WHERE r.project_id = :pid AND r.workspace_id = :wid
                AND r.relationship_type != 'external_reference'
            """),
            {"pid": project_id, "wid": workspace_id}
        ).fetchall()

        # Group symbols by file
        files_map: Dict[str, List[Dict]] = {}
        symbol_type_counts: Dict[str, int] = {}
        for s in symbols:
            fp = s.file_path
            if fp not in files_map:
                files_map[fp] = []
            files_map[fp].append({
                "id": s.id, "name": s.name, "type": s.symbol_type, "line": s.line_number
            })
            symbol_type_counts[s.symbol_type] = symbol_type_counts.get(s.symbol_type, 0) + 1

        # Relationship type counts
        rel_type_counts: Dict[str, int] = {}
        incoming_counts: Dict[int, int] = {}
        for r in relationships:
            rel_type_counts[r.relationship_type] = rel_type_counts.get(r.relationship_type, 0) + 1
            if r.to_symbol_id:
                incoming_counts[r.to_symbol_id] = incoming_counts.get(r.to_symbol_id, 0) + 1

        # Top referenced symbols (by incoming relationship count)
        symbol_id_map = {s.id: s for s in symbols}
        top_referenced = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_symbols = []
        for sym_id, count in top_referenced:
            s = symbol_id_map.get(sym_id)
            if s:
                top_symbols.append({
                    "name": s.name, "qualified_name": s.qualified_name,
                    "type": s.symbol_type, "file": s.file_path,
                    "incoming_references": count,
                })

        return {
            "project_id": project_id,
            "project_name": project.name,
            "total_symbols": len(symbols),
            "total_relationships": len(relationships),
            "total_files": len(files_map),
            "symbol_type_counts": symbol_type_counts,
            "relationship_type_counts": rel_type_counts,
            "top_referenced_symbols": top_symbols,
            "files": {fp: syms for fp, syms in sorted(files_map.items())},
        }

    async def find_dependencies(
        self,
        project_id: int,
        symbol_name: str,
        direction: str = 'both',
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Find all symbols that depend on a given symbol, or that it depends on.

        Args:
            project_id: Project to search
            symbol_name: Symbol name (simple or qualified)
            direction: 'dependents', 'dependencies', or 'both'
            workspace_id: Workspace for isolation
        """
        # Find the target symbol(s)
        target_symbols = self.db.execute(
            text("""
                SELECT id, name, qualified_name, file_path, symbol_type
                FROM codegraph_symbols
                WHERE project_id = :pid AND workspace_id = :wid
                AND (name = :name OR qualified_name = :name)
            """),
            {"pid": project_id, "wid": workspace_id, "name": symbol_name}
        ).fetchall()

        if not target_symbols:
            return {
                "symbol": symbol_name, "direction": direction,
                "dependents": [], "dependencies": [], "error": "Symbol not found"
            }

        target_ids = [s.id for s in target_symbols]
        dependents = []
        dependencies = []

        if direction in ('dependents', 'both'):
            # Symbols that reference target (incoming edges)
            for tid in target_ids:
                rows = self.db.execute(
                    text("""
                        SELECT DISTINCT s.name, s.qualified_name, s.file_path, s.symbol_type, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON s.id = r.from_symbol_id
                        WHERE r.to_symbol_id = :tid AND r.project_id = :pid
                        AND r.relationship_type != 'external_reference'
                    """),
                    {"tid": tid, "pid": project_id}
                ).fetchall()
                for row in rows:
                    dependents.append({
                        "name": row.name, "qualified_name": row.qualified_name,
                        "file": row.file_path, "type": row.symbol_type,
                        "relationship": row.relationship_type,
                    })

        if direction in ('dependencies', 'both'):
            # Symbols that target depends on (outgoing edges)
            for tid in target_ids:
                rows = self.db.execute(
                    text("""
                        SELECT DISTINCT s.name, s.qualified_name, s.file_path, s.symbol_type, r.relationship_type
                        FROM codegraph_relationships r
                        JOIN codegraph_symbols s ON s.id = r.to_symbol_id
                        WHERE r.from_symbol_id = :tid AND r.project_id = :pid
                        AND r.relationship_type != 'external_reference'
                    """),
                    {"tid": tid, "pid": project_id}
                ).fetchall()
                for row in rows:
                    dependencies.append({
                        "name": row.name, "qualified_name": row.qualified_name,
                        "file": row.file_path, "type": row.symbol_type,
                        "relationship": row.relationship_type,
                    })

        return {
            "symbol": symbol_name,
            "direction": direction,
            "dependents": dependents,
            "dependencies": dependencies,
            "dependents_count": len(dependents),
            "dependencies_count": len(dependencies),
        }
