"""
Workspace Manager - Manages isolated workspaces for workflow executions

Each workflow execution gets a unique workspace directory to prevent file pollution
and enable parallel execution. Workspace is automatically cleaned up after execution.

Key Features:
- Unique workspace per execution: /tmp/workflow_{execution_id}/workspace
- Permanent results storage: /tmp/automatos/results/{execution_id}/
- Automatic cleanup after execution
- Safe file operations within workspace boundaries

Note: For production SaaS, results will move to S3/GCS with workspace_id isolation.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages workspace lifecycle for workflow executions.

    Each execution gets:
    - Temporary workspace: /tmp/workflow_{execution_id}/workspace (auto-deleted)
    - Permanent results dir: /tmp/automatos/results/{execution_id}/ (persistent)

    Future: Will use S3 with workspace_id isolation for production SaaS.
    """
    
    # Base directories
    TEMP_BASE = Path("/tmp")
    RESULTS_BASE = Path("/tmp/automatos/results")  # Changed from /var for local testing
    
    def __init__(self, execution_id: int):
        """
        Initialize workspace manager for a specific execution.
        
        Args:
            execution_id: The workflow execution ID
        """
        self.execution_id = execution_id
        
        # Workspace paths
        self.workspace_root = self.TEMP_BASE / f"workflow_{execution_id}" / "workspace"
        self.results_dir = self.RESULTS_BASE / str(execution_id)
        
        # Track created files/dirs for cleanup
        self.created_files: List[Path] = []
        self.created_dirs: List[Path] = []
        
        logger.info(f"📁 WorkspaceManager initialized for execution {execution_id}")
        logger.info(f"   Workspace: {self.workspace_root}")
        logger.info(f"   Results:   {self.results_dir}")
    
    def create_workspace(self) -> str:
        """
        Create the workspace directory structure.
        
        Returns:
            Path to the workspace root directory
        """
        try:
            # Create workspace directory
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            self.created_dirs.append(self.workspace_root)
            
            # Create results directory
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Workspace created: {self.workspace_root}")
            logger.info(f"✅ Results directory created: {self.results_dir}")
            
            return str(self.workspace_root)
            
        except Exception as e:
            logger.error(f"❌ Failed to create workspace: {e}")
            raise
    
    def get_workspace_path(self) -> str:
        """Get the workspace root path"""
        return str(self.workspace_root)
    
    def get_results_path(self) -> str:
        """Get the results directory path"""
        return str(self.results_dir)
    
    def save_result_file(self, source_file: str, filename: Optional[str] = None) -> str:
        """
        Copy a file from workspace to permanent results directory.
        
        Args:
            source_file: Path to file in workspace (can be relative or absolute)
            filename: Optional custom filename in results dir (defaults to source basename)
            
        Returns:
            Path to the saved result file
        """
        try:
            # Resolve source file path
            source_path = Path(source_file)
            if not source_path.is_absolute():
                source_path = self.workspace_root / source_path
            
            # Check if file exists
            if not source_path.exists():
                logger.warning(f"⚠️  Source file not found: {source_path}")
                return ""
            
            # Determine destination filename
            dest_filename = filename or source_path.name
            dest_path = self.results_dir / dest_filename
            
            # Copy file to results directory
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"💾 Saved result: {dest_filename} ({source_path.stat().st_size} bytes)")
            return str(dest_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save result file: {e}")
            return ""
    
    def save_all_results(self, file_patterns: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Auto-save all matching files from workspace to results directory.
        
        Args:
            file_patterns: List of glob patterns (e.g., ['*.pdf', '*.docx', '*.md'])
                          If None, saves all files
        
        Returns:
            Dictionary mapping original filename to saved path
        """
        saved_files = {}
        
        try:
            if not self.workspace_root.exists():
                logger.warning(f"⚠️  Workspace doesn't exist: {self.workspace_root}")
                return saved_files
            
            # Default patterns if not specified
            patterns = file_patterns or ['*']
            
            # Find and save all matching files
            for pattern in patterns:
                for file_path in self.workspace_root.rglob(pattern):
                    if file_path.is_file():
                        # Preserve directory structure in results
                        relative_path = file_path.relative_to(self.workspace_root)
                        dest_path = self.results_dir / relative_path
                        
                        # Create parent directories if needed
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        shutil.copy2(file_path, dest_path)
                        saved_files[str(relative_path)] = str(dest_path)
                        
                        logger.info(f"💾 Saved: {relative_path} ({file_path.stat().st_size} bytes)")
            
            if saved_files:
                logger.info(f"✅ Saved {len(saved_files)} result file(s)")
            else:
                logger.warning(f"⚠️  No files matched patterns: {patterns}")
                
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return saved_files
    
    def create_result_manifest(self, execution_metadata: Dict[str, Any]) -> str:
        """
        Create a manifest file with execution metadata and results.
        
        Args:
            execution_metadata: Metadata about the execution (status, duration, etc.)
            
        Returns:
            Path to the manifest file
        """
        try:
            manifest_path = self.results_dir / "manifest.json"
            
            manifest = {
                "execution_id": self.execution_id,
                "timestamp": datetime.now().isoformat(),
                "workspace": str(self.workspace_root),
                "results_directory": str(self.results_dir),
                "metadata": execution_metadata,
                "files": [
                    {
                        "name": f.name,
                        "path": str(f),
                        "size_bytes": f.stat().st_size,
                        "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
                    }
                    for f in self.results_dir.iterdir() if f.is_file() and f.name != "manifest.json"
                ]
            }
            
            import json
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"📋 Created manifest: {manifest_path}")
            return str(manifest_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create manifest: {e}")
            return ""
    
    def cleanup_workspace(self, force: bool = False) -> bool:
        """
        Clean up the temporary workspace directory.
        
        Args:
            force: If True, deletes even if results directory is empty
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Safety check: don't delete if no results were saved (unless forced)
            if not force and self.results_dir.exists():
                result_files = list(self.results_dir.iterdir())
                if not result_files:
                    logger.warning(
                        f"⚠️  Skipping workspace cleanup - no results saved! "
                        f"Use force=True to override."
                    )
                    return False
            
            # Delete workspace directory
            if self.workspace_root.exists():
                # Delete the entire workflow directory (including workspace subdirectory)
                workflow_dir = self.workspace_root.parent
                shutil.rmtree(workflow_dir)
                logger.info(f"🧹 Workspace cleaned up: {workflow_dir}")
                return True
            else:
                logger.info(f"✅ Workspace already clean: {self.workspace_root}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup workspace: {e}")
            return False
    
    def get_result_files(self) -> List[Dict[str, Any]]:
        """
        Get list of all files in the results directory.
        
        Returns:
            List of file metadata dictionaries
        """
        files = []
        
        try:
            if not self.results_dir.exists():
                return files
            
            for file_path in self.results_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.results_dir)
                    files.append({
                        "name": file_path.name,
                        "path": str(relative_path),
                        "full_path": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"❌ Failed to list result files: {e}")
            return files
    
    @staticmethod
    def cleanup_old_workspaces(max_age_hours: int = 24) -> int:
        """
        Clean up abandoned workspace directories older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for workspace directories
            
        Returns:
            Number of workspaces cleaned up
        """
        cleaned = 0
        
        try:
            temp_base = WorkspaceManager.TEMP_BASE
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            # Find all workflow_* directories
            for workspace_dir in temp_base.glob("workflow_*"):
                if not workspace_dir.is_dir():
                    continue
                
                # Check age
                dir_age = current_time - workspace_dir.stat().st_mtime
                
                if dir_age > max_age_seconds:
                    try:
                        shutil.rmtree(workspace_dir)
                        cleaned += 1
                        logger.info(f"🧹 Cleaned up old workspace: {workspace_dir.name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to cleanup {workspace_dir}: {e}")
            
            if cleaned > 0:
                logger.info(f"✅ Cleaned up {cleaned} old workspace(s)")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old workspaces: {e}")
            return cleaned


# Utility function for easy access
def get_workspace_manager(execution_id: int) -> WorkspaceManager:
    """Get a workspace manager for the specified execution"""
    return WorkspaceManager(execution_id)

