"""
Project Context Analysis
========================

Analyze project structure, detect frameworks, and track dependencies.
Migrated from context_engineering/manager.py

This enables:
- Understanding project architecture
- Detecting language and framework
- Tracking dependencies
- Providing intelligent context for agents
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Project-level context information"""
    project_path: str
    project_name: str
    language: str
    framework: str
    dependencies: List[str] = field(default_factory=list)
    architecture_pattern: str = "unknown"
    code_file_count: int = 0
    documentation_files: List[str] = field(default_factory=list)
    test_coverage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectContextAnalyzer:
    """
    Analyze project structure and extract metadata.
    
    Detects:
    - Programming language
    - Framework (Django, FastAPI, React, etc.)
    - Architecture pattern (MVC, microservices, etc.)
    - Dependencies
    """
    
    # File extensions for code detection
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
    
    # Directories to ignore
    IGNORE_DIRS = {'node_modules', '__pycache__', '.git', 'venv', 'env', 'build', 'dist', '.next', '.cache'}
    
    def __init__(self):
        pass
    
    async def analyze_project(self, project_path: str) -> ProjectContext:
        """
        Analyze a project and return comprehensive context.
        
        Args:
            project_path: Path to the project root
            
        Returns:
            ProjectContext with all detected information
        """
        path = Path(project_path)
        
        if not path.exists():
            logger.warning(f"Project path does not exist: {project_path}")
            return ProjectContext(
                project_path=project_path,
                project_name=path.name,
                language="unknown",
                framework="unknown"
            )
        
        # Detect language and framework
        language, framework = self._detect_project_type(path)
        
        # Get dependencies
        dependencies = self._get_dependencies(path, language)
        
        # Detect architecture pattern
        architecture = self._detect_architecture_pattern(path)
        
        # Count code files
        code_files = self._find_code_files(path)
        
        # Find documentation
        doc_files = self._find_documentation_files(path)
        
        return ProjectContext(
            project_path=str(path.absolute()),
            project_name=path.name,
            language=language,
            framework=framework,
            dependencies=dependencies,
            architecture_pattern=architecture,
            code_file_count=len(code_files),
            documentation_files=[str(f.relative_to(path)) for f in doc_files[:20]],
            last_updated=datetime.utcnow(),
            metadata={
                "analyzed_at": datetime.utcnow().isoformat(),
                "code_extensions_found": list(set(f.suffix for f in code_files))
            }
        )
    
    def _detect_project_type(self, project_path: Path) -> Tuple[str, str]:
        """Detect project language and framework"""
        
        # Check for Python
        if (project_path / "requirements.txt").exists() or \
           (project_path / "setup.py").exists() or \
           (project_path / "pyproject.toml").exists():
            return self._detect_python_framework(project_path)
        
        # Check for Node.js/JavaScript
        if (project_path / "package.json").exists():
            return self._detect_node_framework(project_path)
        
        # Check for Java
        if (project_path / "pom.xml").exists():
            return "java", "maven"
        elif (project_path / "build.gradle").exists():
            return "java", "gradle"
        
        # Check for Go
        if (project_path / "go.mod").exists():
            return "go", "module"
        
        # Check for Rust
        if (project_path / "Cargo.toml").exists():
            return "rust", "cargo"
        
        # Default based on file extensions
        return self._detect_by_files(project_path)
    
    def _detect_python_framework(self, project_path: Path) -> Tuple[str, str]:
        """Detect Python framework"""
        
        if (project_path / "manage.py").exists():
            return "python", "django"
        
        # Check for FastAPI/Flask in requirements
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            try:
                content = req_file.read_text().lower()
                if "fastapi" in content:
                    return "python", "fastapi"
                elif "flask" in content:
                    return "python", "flask"
                elif "django" in content:
                    return "python", "django"
            except:
                pass
        
        # Check pyproject.toml
        pyproject = project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text().lower()
                if "fastapi" in content:
                    return "python", "fastapi"
                elif "flask" in content:
                    return "python", "flask"
                elif "django" in content:
                    return "python", "django"
            except:
                pass
        
        return "python", "generic"
    
    def _detect_node_framework(self, project_path: Path) -> Tuple[str, str]:
        """Detect Node.js framework"""
        
        try:
            with open(project_path / "package.json", 'r') as f:
                package_data = json.load(f)
                deps = {
                    **package_data.get("dependencies", {}),
                    **package_data.get("devDependencies", {})
                }
                
                # Check for TypeScript
                is_typescript = "typescript" in deps
                lang = "typescript" if is_typescript else "javascript"
                
                # Detect framework
                if "next" in deps:
                    return lang, "nextjs"
                elif "react" in deps:
                    return lang, "react"
                elif "vue" in deps:
                    return lang, "vue"
                elif "@angular/core" in deps:
                    return lang, "angular"
                elif "express" in deps:
                    return lang, "express"
                elif "nestjs" in deps or "@nestjs/core" in deps:
                    return lang, "nestjs"
                else:
                    return lang, "generic"
        except:
            return "javascript", "generic"
    
    def _detect_by_files(self, project_path: Path) -> Tuple[str, str]:
        """Detect language based on file extensions"""
        
        extension_counts = {}
        for file in project_path.rglob("*"):
            if file.is_file() and file.suffix in self.CODE_EXTENSIONS:
                if not any(ignore in file.parts for ignore in self.IGNORE_DIRS):
                    extension_counts[file.suffix] = extension_counts.get(file.suffix, 0) + 1
        
        if not extension_counts:
            return "unknown", "generic"
        
        # Get most common extension
        most_common = max(extension_counts.items(), key=lambda x: x[1])
        
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        lang = ext_to_lang.get(most_common[0], "unknown")
        return lang, "generic"
    
    def _detect_architecture_pattern(self, project_path: Path) -> str:
        """Detect architecture pattern based on directory structure"""
        
        try:
            subdirs = {d.name.lower() for d in project_path.iterdir() if d.is_dir()}
        except:
            return "unknown"
        
        # MVC pattern
        if {"models", "views", "controllers"}.issubset(subdirs):
            return "mvc"
        
        # Django-style
        if "templates" in subdirs and "static" in subdirs:
            return "django-style"
        
        # Microservices
        if "microservices" in subdirs or "services" in subdirs:
            return "microservices"
        
        # Component-based (React/Vue style)
        if "components" in subdirs:
            if "pages" in subdirs or "views" in subdirs:
                return "component-based"
        
        # Clean Architecture / Layered
        if {"domain", "application", "infrastructure"}.issubset(subdirs):
            return "clean-architecture"
        
        # Monorepo
        if "packages" in subdirs or "apps" in subdirs:
            return "monorepo"
        
        # Standard src/tests structure
        if "src" in subdirs and "tests" in subdirs:
            return "standard"
        
        return "unknown"
    
    def _get_dependencies(self, project_path: Path, language: str) -> List[str]:
        """Extract project dependencies"""
        
        dependencies = []
        
        if language == "python":
            # requirements.txt
            req_file = project_path / "requirements.txt"
            if req_file.exists():
                try:
                    for line in req_file.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name
                            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                            dependencies.append(pkg)
                except:
                    pass
        
        elif language in ("javascript", "typescript"):
            # package.json
            pkg_file = project_path / "package.json"
            if pkg_file.exists():
                try:
                    with open(pkg_file) as f:
                        data = json.load(f)
                        dependencies.extend(data.get("dependencies", {}).keys())
                except:
                    pass
        
        return dependencies[:50]  # Limit to first 50
    
    def _find_code_files(self, project_path: Path) -> List[Path]:
        """Find all code files in project"""
        
        code_files = []
        
        for ext in self.CODE_EXTENSIONS:
            for file in project_path.rglob(f"*{ext}"):
                if not any(ignore in file.parts for ignore in self.IGNORE_DIRS):
                    code_files.append(file)
                    if len(code_files) >= 500:  # Limit
                        return code_files
        
        return code_files
    
    def _find_documentation_files(self, project_path: Path) -> List[Path]:
        """Find documentation files"""
        
        doc_files = []
        doc_patterns = ["*.md", "*.rst", "*.txt", "*.adoc"]
        
        for pattern in doc_patterns:
            for file in project_path.rglob(pattern):
                if not any(ignore in file.parts for ignore in self.IGNORE_DIRS):
                    doc_files.append(file)
                    if len(doc_files) >= 100:
                        return doc_files
        
        return doc_files


# Singleton
_analyzer: Optional[ProjectContextAnalyzer] = None


def get_project_analyzer() -> ProjectContextAnalyzer:
    """Get project context analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = ProjectContextAnalyzer()
    return _analyzer

