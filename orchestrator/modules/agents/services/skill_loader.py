"""
SkillLoader Service - PRD-22 Core Implementation
================================================

This service implements Git-backed skill loading with progressive disclosure.

Key Features:
- Clone and manage Git repositories containing skills
- Progressive 3-level loading (metadata → core → resources)
- In-memory caching with LRU eviction
- SKILL.md parsing (YAML frontmatter + Markdown body)
- Version tracking and rollback
- Security validation

Author: Automatos AI Team
Date: October 29, 2025
"""

import os
import re
import yaml
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from functools import lru_cache
from collections import OrderedDict

from core.security.git_sanitizer import (
    validate_git_url as _secure_validate_git_url,
    validate_branch as _secure_validate_branch,
    build_git_clone_cmd,
)
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import and_

# Import models exclusively from database.models (single source of truth)
from core.models import (
    Skill, SkillFile, SkillSource, SkillVersion, SkillAuditLog
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class SkillLoaderConfig:
    """Configuration for SkillLoader"""
    
    # Base directory for skill cache
    SKILLS_BASE_DIR = os.path.expanduser("~/.automatos/skills/")
    
    # Git operation timeouts
    GIT_CLONE_TIMEOUT = 300  # 5 minutes
    GIT_PULL_TIMEOUT = 120   # 2 minutes
    
    # Cache settings
    METADATA_CACHE_SIZE = 1000    # Level 1 cache size
    CORE_CACHE_SIZE = 100         # Level 2 cache size
    RESOURCE_CACHE_SIZE = 50      # Level 3 LRU cache size
    
    # Git URL whitelist (security)
    ALLOWED_GIT_DOMAINS = [
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        # Add your organization's Git servers here
    ]
    
    # Validation patterns
    DANGEROUS_PATTERNS = [
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"compile\s*\(",
        r"system\s*\(",
        r"popen\s*\(",
        r"rm\s+-rf",
        r"DROP\s+TABLE",
    ]
    
    # Token estimation (rough approximation: 1 token ≈ 4 characters)
    CHARS_PER_TOKEN = 4


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    return len(text) // SkillLoaderConfig.CHARS_PER_TOKEN


def validate_git_url(git_url: str) -> Tuple[bool, Optional[str]]:
    """Validate Git URL — delegates to centralized git_sanitizer (PRD-70 FIX-01)."""
    return _secure_validate_git_url(git_url, SkillLoaderConfig.ALLOWED_GIT_DOMAINS)


def scan_for_dangerous_patterns(content: str) -> Tuple[bool, List[str]]:
    """
    Scan content for dangerous code patterns.

    PRD-71: Delegates to the full 42-pattern scanner from plugin_security_scanner.
    Falls back to the original 8-pattern list if the import fails.

    Returns:
        (is_safe, list_of_matches)
    """
    try:
        from core.services.plugin_security_scanner import quick_scan
        findings = quick_scan(content, filename="SKILL.md")
        if findings:
            matches = [f.description for f in findings]
            return False, matches
        return True, []
    except ImportError:
        # Fallback: original 8 patterns
        matches = []
        for pattern in SkillLoaderConfig.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(pattern)
        return len(matches) == 0, matches


def parse_yaml_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """
    Extract YAML frontmatter and markdown body from SKILL.md.
    
    Expected format:
    ---
    name: skill-name
    description: Skill description
    version: 1.0.0
    ---
    
    # Markdown content here
    
    Returns:
        (yaml_dict, markdown_body)
    """
    # Match YAML frontmatter between --- markers
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)
    
    if not match:
        logger.warning("No YAML frontmatter found in SKILL.md")
        return None, content
    
    yaml_str = match.group(1)
    markdown_body = match.group(2)
    
    try:
        yaml_data = yaml.safe_load(yaml_str)
        return yaml_data, markdown_body
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML frontmatter: {e}")
        return None, markdown_body


def extract_repo_name(git_url: str) -> str:
    """Extract repository name from Git URL"""
    # Handle various Git URL formats
    # https://github.com/anthropics/skills.git -> skills
    # git@github.com:anthropics/skills.git -> skills
    
    name = git_url.rstrip('/').split('/')[-1]
    if name.endswith('.git'):
        name = name[:-4]
    return name


# ============================================================================
# LRU Cache for Resource Loading (Level 3)
# ============================================================================

class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, max_size: int):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end (most recent)"""
        if key not in self.cache:
            return None
        
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put value and evict oldest if needed"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest (first item)
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()


# ============================================================================
# Main SkillLoader Class
# ============================================================================

class SkillLoader:
    """
    SkillLoader - Core service for Git-backed skill management.
    
    Implements:
    - Git repository cloning and updating
    - Progressive 3-level skill loading
    - In-memory caching
    - Skill indexing and validation
    - Version tracking
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize SkillLoader"""
        self.db = db_session
        
        # Create base directory
        os.makedirs(SkillLoaderConfig.SKILLS_BASE_DIR, exist_ok=True)
        
        # Initialize caches
        self.metadata_cache: Dict[str, Dict] = {}  # Level 1: {skill_name: metadata}
        self.core_content_cache: Dict[str, str] = {}  # Level 2: {skill_name: core_content}
        self.resource_cache = LRUCache(SkillLoaderConfig.RESOURCE_CACHE_SIZE)  # Level 3: LRU
        
        logger.info("SkillLoader initialized")
    
    # ========================================================================
    # Git Repository Management
    # ========================================================================
    
    def add_git_repository(
        self,
        git_url: str,
        source_name: Optional[str] = None,
        branch: str = "main",
        auto_update: bool = False,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Clone a Git repository and index all skills.
        
        Args:
            git_url: Git repository URL
            source_name: Friendly name (auto-generated if None)
            branch: Git branch to clone (default: main)
            auto_update: Enable auto-updates (default: False)
            db: Database session (uses self.db if None)
            
        Returns:
            {
                "source_id": int,
                "source_name": str,
                "skills_discovered": int,
                "skills": [{"name": str, "id": int}, ...],
                "status": "success" | "error",
                "error": str | None
            }
        """
        start_time = datetime.now()
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # 1. Validate Git URL
            is_valid, error_msg = validate_git_url(git_url)
            if not is_valid:
                logger.error(f"Invalid Git URL: {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg
                }
            
            # 2. Generate source name
            if not source_name:
                repo_name = extract_repo_name(git_url)
                source_name = f"git-{repo_name}"
            
            # 3. Check if source already exists
            existing_source = db.query(SkillSource).filter(
                SkillSource.source_name == source_name
            ).first()
            
            if existing_source:
                return {
                    "status": "error",
                    "error": f"Source '{source_name}' already exists. Use update_git_repository() instead."
                }
            
            # 4. Clone repository
            local_path = os.path.join(SkillLoaderConfig.SKILLS_BASE_DIR, source_name)
            
            logger.info(f"Cloning {git_url} to {local_path}...")
            
            clone_result = self._git_clone(git_url, local_path, branch)
            
            if not clone_result["success"]:
                return {
                    "status": "error",
                    "error": f"Git clone failed: {clone_result['error']}"
                }
            
            commit_sha = clone_result["commit_sha"]
            
            # 5. Create SkillSource record
            skill_source = SkillSource(
                source_name=source_name,
                source_type="git",
                git_url=git_url,
                git_branch=branch,
                git_commit_sha=commit_sha,
                local_cache_path=local_path,
                auto_update=auto_update,
                status="syncing"
            )
            
            db.add(skill_source)
            db.flush()  # Get ID
            
            # 6. Index repository
            index_result = self._index_repository(db, skill_source.id, local_path)

            # 6b. Clear caches so new/updated skills are served immediately
            self.clear_caches()

            # 7. Update source with results
            skill_source.skills_discovered = len(index_result["skills"])
            skill_source.status = "active"
            skill_source.last_sync_at = datetime.now()
            skill_source.last_sync_status = "success"
            
            db.commit()
            
            # 8. Log audit
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._log_audit(
                db=db,
                source_id=skill_source.id,
                action="sync",
                status="success",
                action_details={
                    "git_url": git_url,
                    "commit_sha": commit_sha,
                    "skills_discovered": len(index_result["skills"])
                },
                execution_time_ms=execution_time_ms
            )
            
            logger.info(f"Successfully added Git repository: {source_name} ({len(index_result['skills'])} skills)")
            
            return {
                "status": "success",
                "source_id": skill_source.id,
                "source_name": source_name,
                "skills_discovered": len(index_result["skills"]),
                "skills": index_result["skills"],
                "commit_sha": commit_sha
            }
            
        except Exception as e:
            logger.error(f"Error adding Git repository: {e}", exc_info=True)
            
            if db:
                db.rollback()
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_git_repository(
        self,
        source_name: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Update a Git repository (git pull) and re-index skills.
        
        Args:
            source_name: Name of the skill source
            db: Database session
            
        Returns:
            {
                "status": "success" | "error",
                "changes": {"added": [], "updated": [], "removed": []},
                "old_commit": str,
                "new_commit": str
            }
        """
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # 1. Get source
            source = db.query(SkillSource).filter(
                SkillSource.source_name == source_name
            ).first()
            
            if not source:
                return {"status": "error", "error": f"Source '{source_name}' not found"}
            
            if source.source_type != "git":
                return {"status": "error", "error": "Source is not a Git repository"}
            
            old_commit = source.git_commit_sha
            
            # 2. Git pull
            logger.info(f"Updating Git repository: {source_name}")
            
            pull_result = self._git_pull(source.local_cache_path)
            
            if not pull_result["success"]:
                return {"status": "error", "error": f"Git pull failed: {pull_result['error']}"}
            
            new_commit = pull_result["commit_sha"]
            
            # 3. Check if there are changes
            if old_commit == new_commit:
                logger.info(f"No changes in repository {source_name}")
                return {
                    "status": "success",
                    "changes": {"added": [], "updated": [], "removed": []},
                    "old_commit": old_commit,
                    "new_commit": new_commit,
                    "message": "No changes"
                }
            
            # 4. Re-index repository
            source.status = "syncing"
            db.commit()
            
            index_result = self._index_repository(db, source.id, source.local_cache_path)
            
            # 5. Update source
            source.git_commit_sha = new_commit
            source.skills_discovered = len(index_result["skills"])
            source.status = "active"
            source.last_sync_at = datetime.now()
            source.last_sync_status = "success"
            
            db.commit()
            
            # 6. Clear caches
            self.clear_caches()
            
            logger.info(f"Successfully updated repository: {source_name}")
            
            return {
                "status": "success",
                "changes": index_result.get("changes", {}),
                "old_commit": old_commit,
                "new_commit": new_commit
            }
            
        except Exception as e:
            logger.error(f"Error updating Git repository: {e}", exc_info=True)
            
            if db:
                db.rollback()
            
            return {"status": "error", "error": str(e)}
    
    def rollback_git_repository(
        self,
        source_name: str,
        commit_sha: str,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Rollback a Git repository to a specific commit.
        
        Args:
            source_name: Name of the skill source
            commit_sha: Commit SHA to rollback to
            db: Database session
            
        Returns:
            {"status": "success" | "error", "error": str | None}
        """
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # 1. Get source
            source = db.query(SkillSource).filter(
                SkillSource.source_name == source_name
            ).first()
            
            if not source:
                return {"status": "error", "error": f"Source '{source_name}' not found"}
            
            # 2. Git checkout
            logger.info(f"Rolling back {source_name} to {commit_sha}")
            
            checkout_result = self._git_checkout(source.local_cache_path, commit_sha)
            
            if not checkout_result["success"]:
                return {"status": "error", "error": f"Git checkout failed: {checkout_result['error']}"}
            
            # 3. Re-index
            source.status = "syncing"
            db.commit()
            
            index_result = self._index_repository(db, source.id, source.local_cache_path)
            
            # 4. Update source
            source.git_commit_sha = commit_sha
            source.skills_discovered = len(index_result["skills"])
            source.status = "active"
            source.last_sync_at = datetime.now()
            
            db.commit()
            
            # 5. Clear caches
            self.clear_caches()
            
            logger.info(f"Successfully rolled back {source_name} to {commit_sha}")
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Error rolling back Git repository: {e}", exc_info=True)
            
            if db:
                db.rollback()
            
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # Git Operations (Private)
    # ========================================================================
    
    def _git_clone(self, git_url: str, local_path: str, branch: str) -> Dict[str, Any]:
        """Execute git clone with sanitized inputs (PRD-70 FIX-01)."""
        try:
            # Validate branch to prevent argument injection
            ok, err = _secure_validate_branch(branch)
            if not ok:
                return {"success": False, "error": f"Invalid branch: {err}"}

            # Remove directory if exists
            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            # Build safe command with -- separator
            cmd = build_git_clone_cmd(git_url, local_path, branch=branch, depth=50)

            result = subprocess.run(
                cmd,
                timeout=SkillLoaderConfig.GIT_CLONE_TIMEOUT,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            # Get commit SHA
            commit_sha = self._get_commit_sha(local_path)
            
            return {
                "success": True,
                "commit_sha": commit_sha
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Git clone timeout after {SkillLoaderConfig.GIT_CLONE_TIMEOUT} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _git_pull(self, local_path: str) -> Dict[str, Any]:
        """Execute git pull"""
        try:
            cmd = ["git", "-C", local_path, "pull"]
            
            result = subprocess.run(
                cmd,
                timeout=SkillLoaderConfig.GIT_PULL_TIMEOUT,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            # Get new commit SHA
            commit_sha = self._get_commit_sha(local_path)
            
            return {
                "success": True,
                "commit_sha": commit_sha
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Git pull timeout after {SkillLoaderConfig.GIT_PULL_TIMEOUT} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _git_checkout(self, local_path: str, commit_sha: str) -> Dict[str, Any]:
        """Execute git checkout"""
        try:
            cmd = ["git", "-C", local_path, "checkout", commit_sha]
            
            result = subprocess.run(
                cmd,
                timeout=60,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr
                }
            
            return {"success": True}
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_commit_sha(self, local_path: str) -> str:
        """Get current commit SHA"""
        try:
            cmd = ["git", "-C", local_path, "rev-parse", "HEAD"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to get commit SHA: {e}")
            return "unknown"
    
    # ========================================================================
    # Repository Indexing
    # ========================================================================
    
    def _index_repository(
        self,
        db: Session,
        source_id: int,
        repo_path: str
    ) -> Dict[str, Any]:
        """
        Index all skills in a repository.
        
        Discovers SKILL.md files and creates/updates Skill records.
        
        Returns:
            {
                "skills": [{"id": int, "name": str}, ...],
                "changes": {"added": [], "updated": [], "removed": []}
            }
        """
        skills = []
        changes = {"added": [], "updated": [], "removed": []}
        
        # Find all SKILL.md files
        repo_path_obj = Path(repo_path)
        skill_files = list(repo_path_obj.rglob("SKILL.md"))
        
        logger.info(f"Found {len(skill_files)} SKILL.md files in {repo_path}")
        
        for skill_file in skill_files:
            try:
                # Read SKILL.md
                content = skill_file.read_text(encoding='utf-8')
                
                # Parse YAML frontmatter
                yaml_data, markdown_body = parse_yaml_frontmatter(content)
                
                if not yaml_data:
                    logger.warning(f"Skipping {skill_file} - no valid YAML frontmatter")
                    continue
                
                # Validate required fields
                if 'name' not in yaml_data or 'description' not in yaml_data:
                    logger.warning(f"Skipping {skill_file} - missing required fields (name, description)")
                    continue
                
                skill_name = yaml_data['name']
                skill_dir = skill_file.parent
                
                # Security check
                is_safe, dangerous_matches = scan_for_dangerous_patterns(content)
                if not is_safe:
                    logger.warning(f"Skipping {skill_name} - contains dangerous patterns: {dangerous_matches}")
                    continue
                
                # Check if skill exists
                existing_skill = db.query(Skill).filter(
                    Skill.name == skill_name,
                    Skill.skill_source == str(source_id)
                ).first()
                
                if existing_skill:
                    # Update existing skill
                    existing_skill.description = yaml_data.get('description', '')
                    existing_skill.skill_version = yaml_data.get('version', '1.0.0')
                    existing_skill.prompt_template = markdown_body
                    existing_skill.filesystem_path = str(skill_dir)
                    existing_skill.tags = yaml_data.get('tags', [])
                    existing_skill.skill_metadata = yaml_data
                    existing_skill.last_sync_at = datetime.now()
                    
                    skill = existing_skill
                    changes["updated"].append(skill_name)
                else:
                    # Create new skill
                    skill = Skill(
                        name=skill_name,
                        description=yaml_data.get('description', ''),
                        skill_type=yaml_data.get('skill_type', 'custom'),
                        category=yaml_data.get('category', 'general'),
                        skill_version=yaml_data.get('version', '1.0.0'),
                        skill_source=str(source_id),
                        prompt_template=markdown_body,
                        filesystem_path=str(skill_dir),
                        tags=yaml_data.get('tags', []),
                        skill_metadata=yaml_data,
                        is_active=True,
                        last_sync_at=datetime.now()
                    )
                    
                    db.add(skill)
                    db.flush()  # Get ID
                    
                    changes["added"].append(skill_name)
                
                # Index files for progressive disclosure
                self._index_skill_files(db, skill.id, skill_dir)
                
                skills.append({"id": skill.id, "name": skill_name})
                
            except Exception as e:
                logger.error(f"Error indexing {skill_file}: {e}", exc_info=True)
                continue
        
        db.commit()
        
        logger.info(f"Indexed {len(skills)} skills: {changes}")
        
        return {
            "skills": skills,
            "changes": changes
        }
    
    def _index_skill_files(
        self,
        db: Session,
        skill_id: int,
        skill_dir: Path
    ):
        """
        Index all files in a skill directory for progressive disclosure.
        """
        # Delete existing file records
        db.query(SkillFile).filter(SkillFile.skill_id == skill_id).delete()
        
        skill_dir_path = Path(skill_dir)
        
        # Index all files
        for file_path in skill_dir_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            relative_path = file_path.relative_to(skill_dir_path)
            
            # Determine file type and load level
            if file_path.name == "SKILL.md":
                file_type = "core"
                load_level = 2
            elif file_path.suffix == ".md":
                file_type = "resource"
                load_level = 3
            elif file_path.parent.name == "scripts":
                file_type = "script"
                load_level = 3
            elif file_path.parent.name in ["examples", "data"]:
                file_type = "example"
                load_level = 3
            else:
                # Skip unknown files
                continue
            
            # Read file for token estimation
            try:
                content = file_path.read_text(encoding='utf-8')
                estimated_tokens = estimate_tokens(content)
                file_size = len(content.encode('utf-8'))
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                estimated_tokens = 0
                file_size = 0
            
            # Create SkillFile record
            skill_file = SkillFile(
                skill_id=skill_id,
                file_path=str(relative_path),
                file_type=file_type,
                load_level=load_level,
                file_size_bytes=file_size,
                estimated_tokens=estimated_tokens
            )
            
            db.add(skill_file)
        
        db.commit()
    
    # ========================================================================
    # Progressive Disclosure - Level 1: Metadata
    # ========================================================================
    
    def load_skill_metadata(
        self,
        skill_name: str,
        db: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load Level 1: Skill metadata only (YAML frontmatter).
        
        This is always loaded - very lightweight (~50 tokens per skill).
        
        Args:
            skill_name: Name of the skill
            db: Database session
            
        Returns:
            Metadata dict or None if skill not found
        """
        # Check cache first
        if skill_name in self.metadata_cache:
            logger.debug(f"Metadata cache hit: {skill_name}")
            return self.metadata_cache[skill_name]
        
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # Query database
            skill = db.query(Skill).filter(
                Skill.name == skill_name,
                Skill.is_active == True
            ).first()
            
            if not skill:
                logger.warning(f"Skill not found: {skill_name}")
                return None
            
            # If skill_metadata exists (PRD-22 skill), use it
            if skill.skill_metadata:
                metadata = skill.skill_metadata
            else:
                # Legacy skill - construct metadata from DB fields
                metadata = {
                    "name": skill.name,
                    "description": skill.description,
                    "version": skill.skill_version or "1.0.0",
                    "skill_type": skill.skill_type,
                    "category": skill.category,
                    "tags": skill.tags or []
                }
            
            # Add ID for convenience
            metadata["id"] = skill.id
            
            # Cache it
            self.metadata_cache[skill_name] = metadata
            
            logger.debug(f"Loaded metadata for: {skill_name}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading skill metadata: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # Builtin-core runtime freshness
    # ========================================================================

    # Map builtin-core skill names to their on-disk .md paths
    _BUILTIN_PATHS: Dict[str, Path] = {
        "platform-management": Path(__file__).resolve().parents[3] / "core" / "seeds" / "platform-management-skill.md",
    }

    def _refresh_builtin_if_stale(self, skill, db: Session) -> Optional[str]:
        """Check if a builtin-core skill's DB row is stale vs the on-disk .md.

        Compares SHA-256 of the on-disk content against skill.content_hash.
        If different, updates prompt_template + content_hash inline (~5ms).
        Returns the (possibly refreshed) prompt_template, or None to fall through.
        """
        import hashlib

        disk_path = self._BUILTIN_PATHS.get(skill.name)
        if not disk_path or not disk_path.exists():
            return skill.prompt_template

        raw = disk_path.read_text(encoding="utf-8").strip()
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            markdown_body = parts[2].strip() if len(parts) > 2 else raw
        else:
            markdown_body = raw

        disk_hash = hashlib.sha256(markdown_body.encode("utf-8")).hexdigest()

        if skill.content_hash == disk_hash:
            return skill.prompt_template

        # Stale — update DB inline
        skill.prompt_template = markdown_body
        skill.content_hash = disk_hash
        try:
            db.commit()
            logger.info("Refreshed builtin-core skill '%s' from disk (hash=%s…)", skill.name, disk_hash[:12])
        except Exception:
            db.rollback()
            logger.warning("Failed to refresh builtin skill '%s' — using DB version", skill.name, exc_info=True)

        return skill.prompt_template

    # ========================================================================
    # Progressive Disclosure - Level 2: Core Content
    # ========================================================================
    
    def load_skill_core(
        self,
        skill_name: str,
        db: Optional[Session] = None
    ) -> Optional[str]:
        """
        Load Level 2: Core skill content (SKILL.md markdown body).
        
        This is loaded when the skill is relevant to a task (~2000 tokens).
        
        Args:
            skill_name: Name of the skill
            db: Database session
            
        Returns:
            Core content string or None if skill not found
        """
        # Check cache first
        if skill_name in self.core_content_cache:
            logger.debug(f"Core content cache hit: {skill_name}")
            return self.core_content_cache[skill_name]
        
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # Query database
            skill = db.query(Skill).filter(
                Skill.name == skill_name,
                Skill.is_active == True
            ).first()
            
            if not skill:
                logger.warning(f"Skill not found: {skill_name}")
                return None

            # Runtime freshness check for builtin-core skills
            if skill.skill_source == "builtin-core":
                core_content = self._refresh_builtin_if_stale(skill, db)
                if core_content:
                    self.core_content_cache[skill_name] = core_content
                    return core_content

            # Get core content
            core_content = None

            if skill.prompt_template:
                # PRD-22 skill - use prompt_template
                core_content = skill.prompt_template
            elif skill.filesystem_path:
                # Try to read SKILL.md from filesystem
                skill_md_path = Path(skill.filesystem_path) / "SKILL.md"
                if skill_md_path.exists():
                    content = skill_md_path.read_text(encoding='utf-8')
                    yaml_data, markdown_body = parse_yaml_frontmatter(content)
                    core_content = markdown_body
            
            if not core_content:
                # Legacy skill - use implementation field
                core_content = skill.implementation or skill.description
            
            # Cache it
            self.core_content_cache[skill_name] = core_content
            
            # Log token estimate
            if core_content:
                tokens = estimate_tokens(core_content)
                logger.info(f"Loaded core content for {skill_name}: ~{tokens} tokens")
            else:
                logger.warning(f"No core content found for {skill_name}")
            
            return core_content
            
        except Exception as e:
            logger.error(f"Error loading skill core content: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # Progressive Disclosure - Level 3: Resources
    # ========================================================================
    
    def load_skill_resource(
        self,
        skill_name: str,
        resource_path: str,
        db: Optional[Session] = None
    ) -> Optional[str]:
        """
        Load Level 3: Specific resource file on-demand.
        
        This is only loaded when explicitly requested (variable tokens).
        
        Args:
            skill_name: Name of the skill
            resource_path: Relative path to resource file
            db: Database session
            
        Returns:
            Resource content string or None if not found
        """
        cache_key = f"{skill_name}:{resource_path}"
        
        # Check LRU cache
        cached = self.resource_cache.get(cache_key)
        if cached:
            logger.debug(f"Resource cache hit: {cache_key}")
            return cached
        
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # Get skill
            skill = db.query(Skill).filter(
                Skill.name == skill_name,
                Skill.is_active == True
            ).first()
            
            if not skill or not skill.filesystem_path:
                logger.warning(f"Skill not found or no filesystem path: {skill_name}")
                return None
            
            # Construct full path
            full_path = Path(skill.filesystem_path) / resource_path
            
            if not full_path.exists():
                logger.warning(f"Resource not found: {full_path}")
                return None
            
            # Read resource
            content = full_path.read_text(encoding='utf-8')
            
            # Cache in LRU
            self.resource_cache.put(cache_key, content)
            
            # Log access
            tokens = estimate_tokens(content)
            logger.info(f"Loaded resource {resource_path} for {skill_name}: ~{tokens} tokens")
            
            self._log_audit(
                db=db,
                skill_id=skill.id,
                action="load_resource",
                status="success",
                action_details={"resource_path": resource_path, "tokens": tokens}
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading skill resource: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # Script Execution Support
    # ========================================================================
    
    def get_skill_script_path(
        self,
        skill_name: str,
        script_name: str,
        db: Optional[Session] = None
    ) -> Optional[str]:
        """
        Get absolute path to a skill script for execution.
        
        Args:
            skill_name: Name of the skill
            script_name: Name of the script file
            db: Database session
            
        Returns:
            Absolute path to script or None if not found
        """
        db = db or self.db
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # Get skill
            skill = db.query(Skill).filter(
                Skill.name == skill_name,
                Skill.is_active == True
            ).first()
            
            if not skill or not skill.filesystem_path:
                return None
            
            skill_dir = Path(skill.filesystem_path)
            
            # Try direct path
            script_path = skill_dir / script_name
            if script_path.exists():
                return str(script_path)
            
            # Try scripts/ subdirectory
            script_path = skill_dir / "scripts" / script_name
            if script_path.exists():
                return str(script_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting script path: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def clear_caches(self):
        """Clear all caches"""
        self.metadata_cache.clear()
        self.core_content_cache.clear()
        self.resource_cache.clear()
        logger.info("Cleared all skill caches")
    
    def _log_audit(
        self,
        db: Session,
        action: str,
        status: str,
        skill_id: Optional[int] = None,
        source_id: Optional[int] = None,
        action_details: Optional[Dict] = None,
        execution_time_ms: Optional[int] = None
    ):
        """Log audit entry"""
        try:
            audit_log = SkillAuditLog(
                skill_id=skill_id,
                source_id=source_id,
                action=action,
                action_details=action_details,
                status=status,
                execution_time_ms=execution_time_ms
            )
            db.add(audit_log)
            db.commit()
        except Exception as e:
            logger.error(f"Error logging audit: {e}", exc_info=True)


# ============================================================================
# Singleton Pattern
# ============================================================================

_skill_loader_instance: Optional[SkillLoader] = None


def get_skill_loader(db_session: Optional[Session] = None) -> SkillLoader:
    """
    Get singleton SkillLoader instance.
    
    Args:
        db_session: Database session (required on first call)
        
    Returns:
        SkillLoader instance
    """
    global _skill_loader_instance
    
    if _skill_loader_instance is None:
        _skill_loader_instance = SkillLoader(db_session)
    
    return _skill_loader_instance


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Import Anthropic's skills
    
    from core.database.database import get_db
    
    db = next(get_db())
    skill_loader = get_skill_loader(db)
    
    # Add Git repository
    result = skill_loader.add_git_repository(
        git_url="https://github.com/anthropics/skills.git",
        source_name="anthropic-official",
        branch="main",
        auto_update=True
    )
    
    print(f"Import result: {result}")
    
    if result["status"] == "success":
        # Load a skill
        metadata = skill_loader.load_skill_metadata("advanced-code-review")
        print(f"\nMetadata: {metadata}")
        
        core_content = skill_loader.load_skill_core("advanced-code-review")
        print(f"\nCore content length: {len(core_content)} chars")
