"""
Agent Action Executor
=====================

Enables agents to perform real actions on the system:
- File system operations (read/write/delete)
- Shell command execution
- Directory operations

Security features:
- Sandboxed workspace
- Command whitelisting
- Permission checks
- Audit logging
"""

import os
import subprocess
import shutil
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import tempfile
import hashlib
import re

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass


class ActionExecutor:
    """
    Executes real actions on behalf of agents with security controls
    """
    
    # Define safe commands that agents can execute
    # NOTE: Intentionally restrictive — no interpreters (python, curl, wget, docker)
    # that could be used for arbitrary code execution or data exfiltration.
    SAFE_COMMANDS = {
        # File operations (read-only)
        'ls', 'cat', 'head', 'tail', 'grep', 'wc', 'file', 'stat',
        # System info (read-only)
        'pwd', 'whoami', 'date', 'echo', 'which',
        # Text processing (read-only)
        'cut', 'sort', 'uniq', 'tr',
    }

    # Shell metacharacters that enable command chaining/injection
    SHELL_METACHARACTERS = set(';|&$`><(){}')

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/',  # Prevent rm -rf /
        r':\(\)\{.*\}',    # Fork bomb
        r'>\s*/dev/sd',    # Direct disk write
        r'mkfs\.',         # Filesystem formatting
        r'dd\s+.*of=/dev', # Direct disk operations
    ]
    
    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize the action executor
        
        Args:
            workspace_root: Root directory for agent operations (sandboxed)
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root).resolve()
        else:
            # Create a temporary workspace
            self.workspace_root = Path(tempfile.mkdtemp(prefix="automatos_workspace_"))
        
        # Ensure workspace exists
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Audit log
        self.audit_log = []
        
        logger.info(f"Action executor initialized with workspace: {self.workspace_root}")
    
    def _audit_action(self, action: str, details: Dict[str, Any], result: Any, success: bool):
        """Log an action for audit purposes"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
            "result": str(result)[:500] if result else None,  # Limit result size
            "success": success
        }
        self.audit_log.append(entry)
        
        if success:
            logger.info(f"Action executed: {action} - {details}")
        else:
            logger.warning(f"Action failed: {action} - {details} - {result}")
    
    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a path within the workspace

        Args:
            path: Path to validate

        Returns:
            Resolved safe path

        Raises:
            SecurityError: If path is outside workspace
        """
        # Resolve the path
        if os.path.isabs(path):
            resolved = Path(path).resolve()
        else:
            resolved = (self.workspace_root / path).resolve()

        # Check if it's within workspace
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError:
            # Provide a clear, actionable error message
            raise SecurityError(
                f"SECURITY: Cannot access '{path}' - path is outside the workspace boundary. "
                f"The agent workspace is: {self.workspace_root}. "
                f"To access files outside the workspace, either:\n"
                f"  1. Use a relative path within the workspace, OR\n"
                f"  2. Copy the file into the workspace first, OR\n"
                f"  3. Configure AUTOMATOS_WORKSPACE environment variable to a different directory."
            )

        return resolved
    
    def read_file(self, file_path: str, encoding: str = 'utf-8') -> Tuple[bool, str]:
        """
        Read a file from the workspace
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            Success status and file content or error message
        """
        try:
            safe_path = self._validate_path(file_path)
            
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not safe_path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            
            content = safe_path.read_text(encoding=encoding)
            
            self._audit_action("read_file", {"path": str(safe_path), "encoding": encoding}, f"{len(content)} bytes", True)
            return True, content
            
        except Exception as e:
            self._audit_action("read_file", {"path": file_path, "encoding": encoding}, str(e), False)
            return False, str(e)
    
    def write_file(self, file_path: str, content: str, encoding: str = 'utf-8') -> Tuple[bool, str]:
        """
        Write content to a file in the workspace
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            
        Returns:
            Success status and success message or error
        """
        try:
            safe_path = self._validate_path(file_path)
            
            # Create parent directories if needed
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            safe_path.write_text(content, encoding=encoding)
            
            self._audit_action("write_file", {"path": str(safe_path), "size": len(content)}, "File written", True)
            return True, f"File written successfully: {safe_path}"
            
        except Exception as e:
            self._audit_action("write_file", {"path": file_path, "size": len(content)}, str(e), False)
            return False, str(e)
    
    def delete_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Delete a file from the workspace
        
        Args:
            file_path: Path to the file
            
        Returns:
            Success status and success message or error
        """
        try:
            safe_path = self._validate_path(file_path)
            
            if not safe_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if safe_path.is_dir():
                shutil.rmtree(safe_path)
            else:
                safe_path.unlink()
            
            self._audit_action("delete_file", {"path": str(safe_path)}, "File deleted", True)
            return True, f"File deleted successfully: {safe_path}"
            
        except Exception as e:
            self._audit_action("delete_file", {"path": file_path}, str(e), False)
            return False, str(e)
    
    def create_directory(self, dir_path: str) -> Tuple[bool, str]:
        """
        Create a directory in the workspace
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Success status and success message or error
        """
        try:
            safe_path = self._validate_path(dir_path)
            safe_path.mkdir(parents=True, exist_ok=True)
            
            self._audit_action("create_directory", {"path": str(safe_path)}, "Directory created", True)
            return True, f"Directory created: {safe_path}"
            
        except Exception as e:
            self._audit_action("create_directory", {"path": dir_path}, str(e), False)
            return False, str(e)
    
    def list_directory(self, dir_path: str = ".") -> Tuple[bool, List[str]]:
        """
        List contents of a directory
        
        Args:
            dir_path: Path to the directory
            
        Returns:
            Success status and list of files/directories or error
        """
        try:
            safe_path = self._validate_path(dir_path)
            
            if not safe_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            if not safe_path.is_dir():
                raise ValueError(f"Not a directory: {dir_path}")
            
            items = []
            for item in safe_path.iterdir():
                rel_path = item.relative_to(self.workspace_root)
                if item.is_dir():
                    items.append(f"{rel_path}/")
                else:
                    items.append(str(rel_path))
            
            self._audit_action("list_directory", {"path": str(safe_path)}, f"{len(items)} items", True)
            return True, sorted(items)
            
        except Exception as e:
            self._audit_action("list_directory", {"path": dir_path}, str(e), False)
            return False, [str(e)]
    
    def _validate_command(self, command: str) -> bool:
        """
        Validate a shell command for safety

        Args:
            command: Command to validate

        Returns:
            True if command is safe

        Raises:
            SecurityError: If command is dangerous
        """
        # Block shell metacharacters that enable injection
        if any(ch in command for ch in self.SHELL_METACHARACTERS):
            raise SecurityError("Shell metacharacters are not allowed in commands")

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                raise SecurityError(f"Dangerous command pattern detected")

        # Extract the base command
        parts = command.split()
        if not parts:
            raise ValueError("Empty command")

        base_command = parts[0]

        # Check if command is in whitelist
        if '/' in base_command:
            base_command = os.path.basename(base_command)

        if base_command not in self.SAFE_COMMANDS:
            raise SecurityError(f"Command not whitelisted: {base_command}")

        return True
    
    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        capture_output: bool = True,
        shell: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a command in the workspace (no shell — prevents injection)

        Args:
            command: Command string (will be split into args list)
            timeout: Command timeout in seconds
            capture_output: Whether to capture output
            shell: Must be False (shell=True is blocked for security)

        Returns:
            Success status and result dictionary
        """
        try:
            if shell:
                raise SecurityError("shell=True is not allowed for security reasons")

            # Validate command safety
            self._validate_command(command)

            # Split into args list — safe because metacharacters are already blocked
            import shlex
            cmd_args = shlex.split(command)

            # Execute in workspace directory (no shell)
            result = subprocess.run(
                cmd_args,
                shell=False,
                cwd=str(self.workspace_root),
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            
            output = {
                "stdout": result.stdout if capture_output else "",
                "stderr": result.stderr if capture_output else "",
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
            
            self._audit_action(
                "execute_command", 
                {"command": command, "timeout": timeout},
                f"Return code: {result.returncode}",
                result.returncode == 0
            )
            
            return result.returncode == 0, output
            
        except subprocess.TimeoutExpired:
            error = f"Command timed out after {timeout} seconds"
            self._audit_action("execute_command", {"command": command, "timeout": timeout}, error, False)
            return False, {"error": error}
            
        except Exception as e:
            self._audit_action("execute_command", {"command": command}, str(e), False)
            return False, {"error": str(e)}
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about the workspace"""
        try:
            # Calculate workspace size
            total_size = 0
            file_count = 0
            for path in self.workspace_root.rglob('*'):
                if path.is_file():
                    total_size += path.stat().st_size
                    file_count += 1
            
            return {
                "workspace_root": str(self.workspace_root),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "audit_log_entries": len(self.audit_log)
            }
        except Exception as e:
            logger.error(f"Error getting workspace info: {e}")
            return {"error": str(e)}
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries"""
        return self.audit_log[-limit:]
    
    def cleanup_workspace(self):
        """Clean up the workspace (use with caution)"""
        try:
            if self.workspace_root and self.workspace_root.exists():
                shutil.rmtree(self.workspace_root)
                logger.info(f"Cleaned up workspace: {self.workspace_root}")
        except Exception as e:
            logger.error(f"Error cleaning up workspace: {e}")


# Global instance for shared workspace
_action_executor = None

def get_action_executor() -> ActionExecutor:
    """Get or create the action executor singleton"""
    global _action_executor
    if _action_executor is None:
        # Use a persistent workspace in /tmp
        from config import config
        workspace = config.AUTOMATOS_WORKSPACE or "/tmp/automatos_workspace"
        _action_executor = ActionExecutor(workspace_root=workspace)
    return _action_executor
