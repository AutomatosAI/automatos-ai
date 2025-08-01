
"""
Enhanced Context Manager with Advanced RAG and Code Similarity Search
====================================================================

This enhanced context manager provides sophisticated retrieval-augmented generation
with code similarity search, project context awareness, and learning from past executions.

Key Enhancements:
- Advanced RAG with multiple vector databases
- Code similarity search using specialized embeddings
- Project context awareness and dependency tracking
- Learning from past execution patterns
- Semantic code search and documentation retrieval
- Multi-modal context integration (code, docs, logs)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import pickle

import numpy as np
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    PythonCodeTextSplitter,
    MarkdownTextSplitter
)
from langchain.schema import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PythonLoader,
    JSONLoader
)
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    ContextualCompressionRetriever
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('context_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CodeContext:
    """Enhanced code context with metadata"""
    file_path: str
    content: str
    language: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    dependencies: List[str]
    complexity_score: float
    last_modified: datetime
    embedding: Optional[np.ndarray] = None

@dataclass
class ProjectContext:
    """Project-level context information"""
    project_path: str
    project_name: str
    language: str
    framework: str
    dependencies: List[str]
    architecture_pattern: str
    code_files: List[CodeContext]
    documentation: List[str]
    test_coverage: float
    last_updated: datetime

@dataclass
class ExecutionContext:
    """Context from past executions"""
    execution_id: str
    task_description: str
    code_generated: str
    success: bool
    error_messages: List[str]
    performance_metrics: Dict[str, float]
    timestamp: datetime
    embedding: Optional[np.ndarray] = None

class CodeAnalyzer:
    """Advanced code analysis for context extraction"""
    
    def __init__(self):
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
    
    async def analyze_code_file(self, file_path: str) -> Optional[CodeContext]:
        """Analyze a code file and extract context"""
        try:
            path = Path(file_path)
            if not path.exists() or path.suffix not in self.supported_languages:
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            language = self.supported_languages[path.suffix]
            
            # Extract code elements based on language
            if language == 'python':
                return await self._analyze_python_code(file_path, content)
            elif language in ['javascript', 'typescript']:
                return await self._analyze_js_code(file_path, content, language)
            else:
                return await self._analyze_generic_code(file_path, content, language)
                
        except Exception as e:
            logger.error(f"Failed to analyze code file {file_path}: {str(e)}")
            return None
    
    async def _analyze_python_code(self, file_path: str, content: str) -> CodeContext:
        """Analyze Python code specifically"""
        import ast
        
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Calculate complexity (simplified)
            complexity_score = len(functions) * 0.1 + len(classes) * 0.2 + len(imports) * 0.05
            
            # Extract dependencies from imports
            dependencies = [imp for imp in imports if not imp.startswith('.')]
            
            return CodeContext(
                file_path=file_path,
                content=content,
                language='python',
                functions=functions,
                classes=classes,
                imports=imports,
                dependencies=dependencies,
                complexity_score=complexity_score,
                last_modified=datetime.fromtimestamp(os.path.getmtime(file_path))
            )
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in Python file {file_path}: {str(e)}")
            return await self._analyze_generic_code(file_path, content, 'python')
    
    async def _analyze_js_code(self, file_path: str, content: str, language: str) -> CodeContext:
        """Analyze JavaScript/TypeScript code"""
        import re
        
        # Extract functions using regex (simplified)
        function_pattern = r'(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))'
        functions = re.findall(function_pattern, content)
        functions = [f[0] or f[1] for f in functions if f[0] or f[1]]
        
        # Extract classes
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)
        
        # Extract imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        imports = re.findall(import_pattern, content)
        
        complexity_score = len(functions) * 0.1 + len(classes) * 0.2 + len(imports) * 0.05
        
        return CodeContext(
            file_path=file_path,
            content=content,
            language=language,
            functions=functions,
            classes=classes,
            imports=imports,
            dependencies=imports,
            complexity_score=complexity_score,
            last_modified=datetime.fromtimestamp(os.path.getmtime(file_path))
        )
    
    async def _analyze_generic_code(self, file_path: str, content: str, language: str) -> CodeContext:
        """Generic code analysis for unsupported languages"""
        
        # Basic analysis using line counts and patterns
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        complexity_score = len(non_empty_lines) * 0.01
        
        return CodeContext(
            file_path=file_path,
            content=content,
            language=language,
            functions=[],
            classes=[],
            imports=[],
            dependencies=[],
            complexity_score=complexity_score,
            last_modified=datetime.fromtimestamp(os.path.getmtime(file_path))
        )

class VectorStoreManager:
    """Manage multiple vector stores for different types of content"""
    
    def __init__(self):
        # Initialize different embeddings for different content types
        self.text_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.code_embeddings = HuggingFaceEmbeddings(
            model_name="microsoft/codebert-base",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize vector stores
        self.text_vectorstore = None
        self.code_vectorstore = None
        self.execution_vectorstore = None
        
        # Initialize FAISS indices for fast similarity search
        self.code_index = None
        self.execution_index = None
        
        # Code similarity model
        try:
            self.code_similarity_model = SentenceTransformer('microsoft/codebert-base')
        except Exception as e:
            logger.warning(f"Failed to load CodeBERT model: {str(e)}")
            self.code_similarity_model = None
        
        logger.info("Vector store manager initialized")
    
    async def initialize_stores(self, persist_directory: str = "./vector_stores"):
        """Initialize all vector stores"""
        try:
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize text vector store (for documentation, logs, etc.)
            text_persist_dir = os.path.join(persist_directory, "text")
            self.text_vectorstore = Chroma(
                embedding_function=self.text_embeddings,
                persist_directory=text_persist_dir
            )
            
            # Initialize code vector store
            code_persist_dir = os.path.join(persist_directory, "code")
            self.code_vectorstore = Chroma(
                embedding_function=self.code_embeddings,
                persist_directory=code_persist_dir
            )
            
            # Initialize execution context store
            execution_persist_dir = os.path.join(persist_directory, "execution")
            self.execution_vectorstore = Chroma(
                embedding_function=self.text_embeddings,
                persist_directory=execution_persist_dir
            )
            
            logger.info("Vector stores initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector stores: {str(e)}")
            raise
    
    async def add_code_context(self, code_contexts: List[CodeContext]):
        """Add code contexts to vector store"""
        try:
            documents = []
            
            for ctx in code_contexts:
                # Create document with enhanced metadata
                doc = Document(
                    page_content=ctx.content,
                    metadata={
                        "file_path": ctx.file_path,
                        "language": ctx.language,
                        "functions": json.dumps(ctx.functions),
                        "classes": json.dumps(ctx.classes),
                        "imports": json.dumps(ctx.imports),
                        "dependencies": json.dumps(ctx.dependencies),
                        "complexity_score": ctx.complexity_score,
                        "last_modified": ctx.last_modified.isoformat(),
                        "type": "code"
                    }
                )
                documents.append(doc)
            
            if documents and self.code_vectorstore:
                await asyncio.to_thread(self.code_vectorstore.add_documents, documents)
                logger.info(f"Added {len(documents)} code contexts to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add code contexts: {str(e)}")
    
    async def add_execution_context(self, execution_contexts: List[ExecutionContext]):
        """Add execution contexts to vector store"""
        try:
            documents = []
            
            for ctx in execution_contexts:
                # Combine task description and code for better retrieval
                content = f"Task: {ctx.task_description}\n\nCode:\n{ctx.code_generated}"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "execution_id": ctx.execution_id,
                        "task_description": ctx.task_description,
                        "success": ctx.success,
                        "error_messages": json.dumps(ctx.error_messages),
                        "performance_metrics": json.dumps(ctx.performance_metrics),
                        "timestamp": ctx.timestamp.isoformat(),
                        "type": "execution"
                    }
                )
                documents.append(doc)
            
            if documents and self.execution_vectorstore:
                await asyncio.to_thread(self.execution_vectorstore.add_documents, documents)
                logger.info(f"Added {len(documents)} execution contexts to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add execution contexts: {str(e)}")
    
    async def similarity_search_code(
        self, 
        query: str, 
        k: int = 5,
        language_filter: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar code with optional language filtering"""
        try:
            if not self.code_vectorstore:
                return []
            
            # Build filter
            filter_dict = {"type": "code"}
            if language_filter:
                filter_dict["language"] = language_filter
            
            # Perform similarity search
            results = await asyncio.to_thread(
                self.code_vectorstore.similarity_search_with_score,
                query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Code similarity search failed: {str(e)}")
            return []
    
    async def similarity_search_executions(
        self, 
        query: str, 
        k: int = 3,
        success_only: bool = True
    ) -> List[Tuple[Document, float]]:
        """Search for similar past executions"""
        try:
            if not self.execution_vectorstore:
                return []
            
            # Build filter
            filter_dict = {"type": "execution"}
            if success_only:
                filter_dict["success"] = True
            
            # Perform similarity search
            results = await asyncio.to_thread(
                self.execution_vectorstore.similarity_search_with_score,
                query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Execution similarity search failed: {str(e)}")
            return []
    
    async def find_similar_code_by_function(
        self, 
        function_name: str, 
        language: Optional[str] = None
    ) -> List[Document]:
        """Find code containing similar functions"""
        try:
            if not self.code_vectorstore:
                return []
            
            # Search for documents containing the function
            filter_dict = {"type": "code"}
            if language:
                filter_dict["language"] = language
            
            # This is a simplified approach - in production, you'd want more sophisticated function matching
            results = await asyncio.to_thread(
                self.code_vectorstore.similarity_search,
                f"function {function_name}",
                k=10,
                filter=filter_dict
            )
            
            # Filter results that actually contain the function
            filtered_results = []
            for doc in results:
                functions = json.loads(doc.metadata.get("functions", "[]"))
                if function_name in functions or function_name in doc.page_content:
                    filtered_results.append(doc)
            
            return filtered_results[:5]
            
        except Exception as e:
            logger.error(f"Function similarity search failed: {str(e)}")
            return []

class EnhancedContextManager:
    """
    Enhanced Context Manager with Advanced RAG and Code Similarity Search
    
    This manager provides:
    - Advanced RAG with multiple vector databases
    - Code similarity search using specialized embeddings
    - Project context awareness and dependency tracking
    - Learning from past execution patterns
    - Semantic code search and documentation retrieval
    """
    
    def __init__(self):
        self.vector_manager = VectorStoreManager()
        self.code_analyzer = CodeAnalyzer()
        
        # Initialize LLM for context compression
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Context caches
        self.project_contexts: Dict[str, ProjectContext] = {}
        self.execution_history: List[ExecutionContext] = []
        
        # Text splitters for different content types
        self.code_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        logger.info("Enhanced Context Manager initialized")
    
    async def initialize(self, persist_directory: str = "./vector_stores"):
        """Initialize the context manager"""
        await self.vector_manager.initialize_stores(persist_directory)
        logger.info("Context manager initialization completed")
    
    async def index_project(self, project_path: str) -> ProjectContext:
        """Index an entire project for context awareness"""
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                raise ValueError(f"Project path does not exist: {project_path}")
            
            logger.info(f"Indexing project: {project_path}")
            
            # Analyze project structure
            project_context = await self._analyze_project_structure(project_path)
            
            # Index code files
            code_contexts = []
            for file_path in self._find_code_files(project_path):
                code_context = await self.code_analyzer.analyze_code_file(str(file_path))
                if code_context:
                    code_contexts.append(code_context)
            
            project_context.code_files = code_contexts
            
            # Add to vector stores
            await self.vector_manager.add_code_context(code_contexts)
            
            # Index documentation
            await self._index_documentation(project_path)
            
            # Cache project context
            self.project_contexts[str(project_path)] = project_context
            
            logger.info(f"Project indexing completed: {len(code_contexts)} files indexed")
            
            return project_context
            
        except Exception as e:
            logger.error(f"Project indexing failed: {str(e)}")
            raise
    
    async def enhanced_rag_retrieve(
        self,
        query: str,
        include_code_similarity: bool = True,
        project_context: Optional[str] = None,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """Enhanced RAG retrieval with code similarity and project context"""
        
        try:
            logger.info(f"Enhanced RAG retrieval for query: {query[:100]}...")
            
            results = {
                "context": "",
                "code_examples": [],
                "similar_executions": [],
                "project_insights": {},
                "embeddings": None,
                "sources": []
            }
            
            # 1. Search for similar code
            if include_code_similarity:
                code_results = await self.vector_manager.similarity_search_code(
                    query, k=max_results
                )
                
                for doc, score in code_results:
                    results["code_examples"].append({
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "file_path": doc.metadata.get("file_path", ""),
                        "language": doc.metadata.get("language", ""),
                        "functions": json.loads(doc.metadata.get("functions", "[]")),
                        "similarity_score": float(score)
                    })
                    results["sources"].append(doc.metadata.get("file_path", ""))
            
            # 2. Search for similar past executions
            execution_results = await self.vector_manager.similarity_search_executions(
                query, k=3
            )
            
            for doc, score in execution_results:
                results["similar_executions"].append({
                    "task_description": doc.metadata.get("task_description", ""),
                    "execution_id": doc.metadata.get("execution_id", ""),
                    "success": doc.metadata.get("success", False),
                    "similarity_score": float(score),
                    "timestamp": doc.metadata.get("timestamp", "")
                })
            
            # 3. Add project context if available
            if project_context and project_context in self.project_contexts:
                proj_ctx = self.project_contexts[project_context]
                results["project_insights"] = {
                    "project_name": proj_ctx.project_name,
                    "language": proj_ctx.language,
                    "framework": proj_ctx.framework,
                    "architecture_pattern": proj_ctx.architecture_pattern,
                    "dependencies": proj_ctx.dependencies[:10],  # Limit to first 10
                    "total_files": len(proj_ctx.code_files),
                    "test_coverage": proj_ctx.test_coverage
                }
            
            # 4. Build comprehensive context
            context_parts = []
            
            if results["project_insights"]:
                context_parts.append(f"Project Context: {json.dumps(results['project_insights'], indent=2)}")
            
            if results["code_examples"]:
                context_parts.append("Relevant Code Examples:")
                for i, example in enumerate(results["code_examples"][:3]):  # Top 3
                    context_parts.append(f"Example {i+1} ({example['language']}):")
                    context_parts.append(example["content"])
            
            if results["similar_executions"]:
                context_parts.append("Similar Past Executions:")
                for execution in results["similar_executions"]:
                    if execution["success"]:
                        context_parts.append(f"- {execution['task_description']} (Success)")
            
            results["context"] = "\n\n".join(context_parts)
            
            # 5. Generate embeddings for the query
            try:
                query_embedding = await asyncio.to_thread(
                    self.vector_manager.text_embeddings.embed_query, 
                    query
                )
                results["embeddings"] = query_embedding
            except Exception as e:
                logger.warning(f"Failed to generate query embeddings: {str(e)}")
            
            logger.info(f"Enhanced RAG retrieval completed: {len(results['code_examples'])} code examples, {len(results['similar_executions'])} similar executions")
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced RAG retrieval failed: {str(e)}")
            return {
                "context": f"Error retrieving context: {str(e)}",
                "code_examples": [],
                "similar_executions": [],
                "project_insights": {},
                "embeddings": None,
                "sources": []
            }
    
    async def retrieve_similar_code(
        self,
        code_snippet: str,
        language: Optional[str] = None,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve code similar to the given snippet"""
        
        try:
            # Search for similar code
            results = await self.vector_manager.similarity_search_code(
                code_snippet, k=10, language_filter=language
            )
            
            similar_code = []
            for doc, score in results:
                if score >= similarity_threshold:
                    similar_code.append({
                        "content": doc.page_content,
                        "file_path": doc.metadata.get("file_path", ""),
                        "language": doc.metadata.get("language", ""),
                        "functions": json.loads(doc.metadata.get("functions", "[]")),
                        "classes": json.loads(doc.metadata.get("classes", "[]")),
                        "similarity_score": float(score),
                        "complexity_score": doc.metadata.get("complexity_score", 0.0)
                    })
            
            return similar_code
            
        except Exception as e:
            logger.error(f"Similar code retrieval failed: {str(e)}")
            return []
    
    async def learn_from_execution(
        self,
        task_description: str,
        code_generated: str,
        success: bool,
        error_messages: List[str] = None,
        performance_metrics: Dict[str, float] = None
    ):
        """Learn from execution results to improve future context retrieval"""
        
        try:
            execution_context = ExecutionContext(
                execution_id=hashlib.md5(f"{task_description}{time.time()}".encode()).hexdigest(),
                task_description=task_description,
                code_generated=code_generated,
                success=success,
                error_messages=error_messages or [],
                performance_metrics=performance_metrics or {},
                timestamp=datetime.utcnow()
            )
            
            # Add to execution history
            self.execution_history.append(execution_context)
            
            # Add to vector store for future retrieval
            await self.vector_manager.add_execution_context([execution_context])
            
            logger.info(f"Learned from execution: {execution_context.execution_id} (Success: {success})")
            
        except Exception as e:
            logger.error(f"Failed to learn from execution: {str(e)}")
    
    async def get_project_dependencies(self, project_path: str) -> List[str]:
        """Get project dependencies for context"""
        
        try:
            if project_path in self.project_contexts:
                return self.project_contexts[project_path].dependencies
            
            # Try to extract dependencies from common files
            dependencies = []
            project_path = Path(project_path)
            
            # Python requirements
            req_files = ["requirements.txt", "Pipfile", "pyproject.toml"]
            for req_file in req_files:
                req_path = project_path / req_file
                if req_path.exists():
                    deps = await self._extract_python_dependencies(req_path)
                    dependencies.extend(deps)
            
            # Node.js dependencies
            package_json = project_path / "package.json"
            if package_json.exists():
                deps = await self._extract_node_dependencies(package_json)
                dependencies.extend(deps)
            
            return list(set(dependencies))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to get project dependencies: {str(e)}")
            return []
    
    async def search_documentation(
        self,
        query: str,
        project_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search through project documentation"""
        
        try:
            if not self.vector_manager.text_vectorstore:
                return []
            
            # Build filter for documentation
            filter_dict = {"type": "documentation"}
            if project_path:
                filter_dict["project_path"] = project_path
            
            results = await asyncio.to_thread(
                self.vector_manager.text_vectorstore.similarity_search_with_score,
                query,
                k=5,
                filter=filter_dict
            )
            
            documentation = []
            for doc, score in results:
                documentation.append({
                    "content": doc.page_content,
                    "file_path": doc.metadata.get("file_path", ""),
                    "title": doc.metadata.get("title", ""),
                    "similarity_score": float(score)
                })
            
            return documentation
            
        except Exception as e:
            logger.error(f"Documentation search failed: {str(e)}")
            return []
    
    # Helper methods
    async def _analyze_project_structure(self, project_path: Path) -> ProjectContext:
        """Analyze project structure and extract metadata"""
        
        # Detect project type and framework
        language, framework = await self._detect_project_type(project_path)
        
        # Get dependencies
        dependencies = await self.get_project_dependencies(str(project_path))
        
        # Detect architecture pattern (simplified)
        architecture_pattern = await self._detect_architecture_pattern(project_path)
        
        return ProjectContext(
            project_path=str(project_path),
            project_name=project_path.name,
            language=language,
            framework=framework,
            dependencies=dependencies,
            architecture_pattern=architecture_pattern,
            code_files=[],  # Will be populated later
            documentation=[],
            test_coverage=0.0,  # Would be calculated from test results
            last_updated=datetime.utcnow()
        )
    
    async def _detect_project_type(self, project_path: Path) -> Tuple[str, str]:
        """Detect project language and framework"""
        
        # Check for Python
        if (project_path / "requirements.txt").exists() or \
           (project_path / "setup.py").exists() or \
           (project_path / "pyproject.toml").exists():
            
            # Detect Python framework
            if (project_path / "manage.py").exists():
                return "python", "django"
            elif any((project_path / "app.py").exists(), 
                    (project_path / "main.py").exists()):
                return "python", "flask"
            else:
                return "python", "generic"
        
        # Check for Node.js
        if (project_path / "package.json").exists():
            try:
                with open(project_path / "package.json", 'r') as f:
                    package_data = json.load(f)
                    deps = {**package_data.get("dependencies", {}), 
                           **package_data.get("devDependencies", {})}
                    
                    if "react" in deps:
                        return "javascript", "react"
                    elif "vue" in deps:
                        return "javascript", "vue"
                    elif "angular" in deps:
                        return "javascript", "angular"
                    elif "express" in deps:
                        return "javascript", "express"
                    else:
                        return "javascript", "generic"
            except:
                return "javascript", "generic"
        
        # Check for Java
        if (project_path / "pom.xml").exists():
            return "java", "maven"
        elif (project_path / "build.gradle").exists():
            return "java", "gradle"
        
        # Default
        return "unknown", "generic"
    
    async def _detect_architecture_pattern(self, project_path: Path) -> str:
        """Detect architecture pattern (simplified)"""
        
        # Look for common directory structures
        subdirs = [d.name for d in project_path.iterdir() if d.is_dir()]
        
        if "models" in subdirs and "views" in subdirs and "controllers" in subdirs:
            return "mvc"
        elif "src" in subdirs and "tests" in subdirs:
            return "standard"
        elif "microservices" in subdirs or "services" in subdirs:
            return "microservices"
        elif "components" in subdirs:
            return "component-based"
        else:
            return "unknown"
    
    def _find_code_files(self, project_path: Path) -> List[Path]:
        """Find all code files in project"""
        
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
        code_files = []
        
        for ext in code_extensions:
            code_files.extend(project_path.rglob(f"*{ext}"))
        
        # Filter out common directories to ignore
        ignore_dirs = {'node_modules', '__pycache__', '.git', 'venv', 'env', 'build', 'dist'}
        
        filtered_files = []
        for file_path in code_files:
            if not any(ignore_dir in file_path.parts for ignore_dir in ignore_dirs):
                filtered_files.append(file_path)
        
        return filtered_files[:100]  # Limit to first 100 files
    
    async def _index_documentation(self, project_path: Path):
        """Index project documentation"""
        
        try:
            doc_files = []
            
            # Find documentation files
            for pattern in ["*.md", "*.rst", "*.txt"]:
                doc_files.extend(project_path.rglob(pattern))
            
            documents = []
            for doc_file in doc_files[:20]:  # Limit to first 20 docs
                try:
                    with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Split content
                    if doc_file.suffix == '.md':
                        chunks = self.markdown_splitter.split_text(content)
                    else:
                        chunks = self.text_splitter.split_text(content)
                    
                    for chunk in chunks:
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "file_path": str(doc_file),
                                "title": doc_file.stem,
                                "type": "documentation",
                                "project_path": str(project_path)
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    logger.warning(f"Failed to process documentation file {doc_file}: {str(e)}")
            
            # Add to vector store
            if documents and self.vector_manager.text_vectorstore:
                await asyncio.to_thread(self.vector_manager.text_vectorstore.add_documents, documents)
                logger.info(f"Indexed {len(documents)} documentation chunks")
                
        except Exception as e:
            logger.error(f"Documentation indexing failed: {str(e)}")
    
    async def _extract_python_dependencies(self, req_file: Path) -> List[str]:
        """Extract Python dependencies from requirements file"""
        
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            dependencies = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before version specifiers)
                    package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('~=')[0]
                    dependencies.append(package.strip())
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to extract Python dependencies: {str(e)}")
            return []
    
    async def _extract_node_dependencies(self, package_json: Path) -> List[str]:
        """Extract Node.js dependencies from package.json"""
        
        try:
            with open(package_json, 'r') as f:
                package_data = json.load(f)
            
            dependencies = []
            for dep_type in ["dependencies", "devDependencies"]:
                if dep_type in package_data:
                    dependencies.extend(package_data[dep_type].keys())
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Failed to extract Node.js dependencies: {str(e)}")
            return []

# Legacy compatibility
class ContextManager(EnhancedContextManager):
    """Legacy compatibility class"""
    
    def __init__(self):
        super().__init__()
        logger.warning("Using legacy ContextManager class. Please migrate to EnhancedContextManager.")
    
    def rag_retrieve(self, query: str) -> str:
        """Legacy RAG retrieve method"""
        try:
            # Run async method in sync context
            import asyncio
            
            # Initialize if not already done
            if not self.vector_manager.text_vectorstore:
                asyncio.run(self.initialize())
            
            result = asyncio.run(self.enhanced_rag_retrieve(query))
            return result.get("context", "No context found")
            
        except Exception as e:
            logger.error(f"Legacy RAG retrieve failed: {str(e)}")
            return f"Error retrieving context: {str(e)}"

if __name__ == "__main__":
    async def main():
        # Example usage
        context_manager = EnhancedContextManager()
        await context_manager.initialize()
        
        # Index a project
        project_path = "/tmp/example_project"
        if os.path.exists(project_path):
            project_context = await context_manager.index_project(project_path)
            print(f"Indexed project: {project_context.project_name}")
        
        # Perform enhanced RAG retrieval
        result = await context_manager.enhanced_rag_retrieve(
            "How to implement authentication in a web application?",
            include_code_similarity=True
        )
        
        print(f"Context: {result['context'][:500]}...")
        print(f"Code examples found: {len(result['code_examples'])}")
        print(f"Similar executions: {len(result['similar_executions'])}")
    
    asyncio.run(main())
