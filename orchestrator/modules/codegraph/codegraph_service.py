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
        from shared.llm import create_embedding_manager
        self.embedding_manager = create_embedding_manager()
        logger.info(f"CodeGraphService using {self.embedding_manager.get_provider_info()['provider']} embeddings")
        
        # Supported file extensions
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
        exclude_patterns: Optional[List[str]] = None
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
        
        # Check if project already exists
        existing = self.db.execute(
            text("SELECT id, status, updated_at FROM codegraph_projects WHERE name = :name"),
            {"name": project_name}
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
            # Delete old symbols and relationships before re-indexing (to handle dimension changes)
            logger.info(f"Deleting old symbols and embeddings for project {project_name} before re-indexing...")
            
            # Count records before deletion
            rel_count = self.db.execute(
                text("SELECT COUNT(*) FROM codegraph_relationships WHERE project_id = :id"),
                {"id": project_id}
            ).scalar()
            sym_count = self.db.execute(
                text("SELECT COUNT(*) FROM codegraph_symbols WHERE project_id = :id"),
                {"id": project_id}
            ).scalar()
            file_count = self.db.execute(
                text("SELECT COUNT(*) FROM codegraph_files WHERE project_id = :id"),
                {"id": project_id}
            ).scalar()
            
            logger.info(f"Found {rel_count} relationships, {sym_count} symbols, {file_count} files to delete")
            
            # Delete old data
            self.db.execute(
                text("DELETE FROM codegraph_relationships WHERE project_id = :id"),
                {"id": project_id}
            )
            self.db.execute(
                text("DELETE FROM codegraph_symbols WHERE project_id = :id"),
                {"id": project_id}
            )
            self.db.execute(
                text("DELETE FROM codegraph_files WHERE project_id = :id"),
                {"id": project_id}
            )
            self.db.execute(
                text("UPDATE codegraph_projects SET status = 'indexing', updated_at = NOW() WHERE id = :id"),
                {"id": project_id}
            )
            
            # Commit deletion immediately
            self.db.commit()
            logger.info(f"✅ Deleted {rel_count} relationships, {sym_count} symbols, {file_count} files")
            
            # Ensure embedding column dimension matches current embedding model
            self._ensure_embedding_dimension()
        else:
            result = self.db.execute(
                text("""
                    INSERT INTO codegraph_projects (name, source_type, source_url, branch, status, exclude_patterns)
                    VALUES (:name, 'github', :url, :branch, 'indexing', CAST(:excludes AS json))
                    RETURNING id
                """),
                {
                    "name": project_name,
                    "url": github_url,
                    "branch": branch,
                    "excludes": json.dumps(exclude_patterns or [])
                }
            )
            project_id = result.fetchone()[0]
            
            # Ensure embedding column dimension matches current embedding model
            self._ensure_embedding_dimension()
        
        self.db.commit()
        
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
            
            for file_path in self._discover_files(repo_path, all_excludes):
                try:
                    parse_result = await self._parse_file(file_path, repo_path)
                    
                    if parse_result:
                        # Store file metadata
                        rel_path = os.path.relpath(file_path, repo_path)
                        files_data.append({
                            "project_id": project_id,
                            "file_path": rel_path,
                            "file_hash": parse_result.file_hash,
                            "file_size": os.path.getsize(file_path),
                            "lines_of_code": parse_result.lines_of_code,
                            "language": parse_result.language
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
                                "metadata": symbol.metadata
                            })
                        
                        # NEW: Store relationships
                        for rel in parse_result.relationships:
                            relationships_data.append({
                                "project_id": project_id,
                                "from_symbol": rel.from_symbol,
                                "to_symbol": rel.to_symbol,
                                "relationship_type": rel.relationship_type,
                                "metadata": rel.metadata
                            })
                        
                        total_files += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
                    continue
            
            # Batch insert files
            if files_data:
                for file_data in files_data:
                    self.db.execute(
                        text("""
                            INSERT INTO codegraph_files 
                            (project_id, file_path, file_hash, file_size, lines_of_code, language)
                            VALUES (:project_id, :file_path, :file_hash, :file_size, :lines_of_code, :language)
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
            # Mark project as failed
            self.db.execute(
                text("UPDATE codegraph_projects SET status = 'failed', updated_at = NOW() WHERE id = :id"),
                {"id": project_id}
            )
            self.db.commit()
            
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
        """Parse a single file and extract symbols"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Calculate file hash
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            lines_of_code = len([line for line in content.split('\n') if line.strip()])
            
            # Detect language
            extension = Path(file_path).suffix
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
                        code_snippet=code_snippet[:1000],  # Limit snippet size
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
                        code_snippet=code_snippet[:1000],
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
            from shared.llm.embedding_manager import get_system_setting
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
                self.db.execute(
                    text(f"ALTER TABLE codegraph_symbols ALTER COLUMN embedding TYPE vector({current_dim})")
                )
                
                # Recreate the index
                try:
                    self.db.execute(
                        text(f"""
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
                            "metadata": json.dumps(symbol.get('metadata', {}))
                        }
                        
                        self.db.execute(
                            text("""
                                INSERT INTO codegraph_symbols
                                (project_id, symbol_type, name, qualified_name, file_path, line_number,
                                 signature, docstring, code_snippet, embedding, metadata)
                                VALUES (:project_id, :symbol_type, :name, :qualified_name, :file_path,
                                        :line_number, :signature, :docstring, :code_snippet,
                                        CAST(:embedding AS vector), :metadata)
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
                                (project_id, from_symbol_id, to_symbol_id, relationship_type, metadata)
                                VALUES (:project_id, :from_symbol_id, :to_symbol_id, :relationship_type, CAST(:metadata AS jsonb))
                            """),
                            {
                                "project_id": project_id,
                                "from_symbol_id": from_symbol_id,
                                "to_symbol_id": to_symbol_id,
                                "relationship_type": rel['relationship_type'],
                                "metadata": json.dumps(rel['metadata'])
                            }
                        )
                        relationships_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert relationship: {e}")
                        continue
                else:
                    skipped_external += 1
            
            self.db.commit()
            logger.info(f"✅ Stored {relationships_inserted} relationships (from {len(relationships_data)} extracted, {matched_fuzzy} fuzzy-matched, {skipped_external} skipped)")
            
        except Exception as e:
            logger.error(f"Failed to store relationships: {e}")
            # Don't raise - relationships are supplementary, don't fail indexing
    
    async def search_symbols(
        self,
        project_name: str,
        query: str,
        symbol_type: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for code symbols by name (fuzzy matching)
        """
        start_time = time.time()
        
        # Get project ID
        project = self.db.execute(
            text("SELECT id FROM codegraph_projects WHERE name = :name"),
            {"name": project_name}
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project '{project_name}' not found")
        
        project_id = project.id
        
        # Build search query
        type_filter = "AND symbol_type = :symbol_type" if symbol_type else ""
        
        search_query = text(f"""
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
                {type_filter}
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
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Semantic search using vector similarity
        """
        start_time = time.time()
        
        # Get project ID
        project = self.db.execute(
            text("SELECT id FROM codegraph_projects WHERE name = :name"),
            {"name": project_name}
        ).fetchone()
        
        if not project:
            raise ValueError(f"Project '{project_name}' not found")
        
        project_id = project.id
        
        # Generate query embedding
        query_embedding = await self.embedding_manager.generate_embedding(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Semantic search
        search_query = text(f"""
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
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM codegraph_symbols
            WHERE project_id = :project_id
                AND embedding IS NOT NULL
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT :limit
        """)
        
        results = self.db.execute(
            search_query,
            {"project_id": project_id, "limit": limit}
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
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all indexed projects"""
        results = self.db.execute(
            text("""
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
                ORDER BY last_indexed DESC NULLS LAST
            """)
        ).fetchall()
        
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
    
    def delete_project(self, project_id: int) -> Dict[str, Any]:
        """Delete a project and all associated data"""
        # Get project stats before deletion
        project = self.db.execute(
            text("SELECT name, total_files, total_symbols FROM codegraph_projects WHERE id = :id"),
            {"id": project_id}
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
        direction: str = "outgoing"
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
            text("SELECT id FROM codegraph_projects WHERE name = :name"),
            {"name": project_name}
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


