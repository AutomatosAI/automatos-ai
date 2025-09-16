
"""
SSH Manager with Banking-Grade Security
=======================================

Provides secure SSH command execution with comprehensive security features,
audit logging, and connection management for the orchestration system.
"""

import asyncio
import logging
import os
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import asynccontextmanager

import paramiko
from paramiko import SSHClient, AutoAddPolicy, RSAKey, Ed25519Key
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SSHConnection:
    host: str
    port: int
    username: str
    key_path: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    security_level: SecurityLevel = SecurityLevel.MEDIUM

@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timestamp: datetime
    host: str
    success: bool

class SSHSecurityManager:
    """Handles SSH security policies and validation"""
    
    def __init__(self):
        self.blocked_commands = {
            'rm -rf /',
            'dd if=/dev/zero',
            'mkfs',
            'fdisk',
            'format',
            'shutdown',
            'reboot',
            'halt',
            'init 0',
            'init 6',
            'killall',
            'pkill -9',
        }
        
        self.restricted_paths = {
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/boot',
            '/sys',
            '/proc/sys',
        }
        
        self.allowed_commands = {
            'ls', 'cat', 'grep', 'find', 'ps', 'top', 'df', 'du',
            'git', 'docker', 'npm', 'pip', 'python', 'node',
            'mkdir', 'touch', 'cp', 'mv', 'chmod', 'chown',
            'systemctl', 'service', 'curl', 'wget'
        }
    
    def validate_command(self, command: str, security_level: SecurityLevel) -> Tuple[bool, str]:
        """Validate command against security policies"""
        command_lower = command.lower().strip()
        
        # Check for blocked commands
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return False, f"Blocked dangerous command: {blocked}"
        
        # Check for restricted paths
        for path in self.restricted_paths:
            if path in command:
                return False, f"Access to restricted path: {path}"
        
        # High security level - only allow whitelisted commands
        if security_level == SecurityLevel.HIGH:
            command_parts = command_lower.split()
            if command_parts and command_parts[0] not in self.allowed_commands:
                return False, f"Command not in whitelist: {command_parts[0]}"
        
        # Check for command injection patterns
        injection_patterns = [';', '&&', '||', '|', '`', '$(',  '$()', '${']
        if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            for pattern in injection_patterns:
                if pattern in command:
                    return False, f"Potential command injection detected: {pattern}"
        
        return True, "Command validated"

class SSHAuditLogger:
    """Comprehensive audit logging for SSH operations"""
    
    def __init__(self, log_file: str = "ssh_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("ssh_audit")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_connection(self, connection: SSHConnection, success: bool, error: str = None):
        """Log SSH connection attempts"""
        log_data = {
            "event": "ssh_connection",
            "host": connection.host,
            "port": connection.port,
            "username": connection.username,
            "success": success,
            "security_level": connection.security_level.value,
            "timestamp": datetime.now().isoformat(),
            "error": error
        }
        
        if success:
            self.logger.info(f"SSH Connection: {json.dumps(log_data)}")
        else:
            self.logger.error(f"SSH Connection Failed: {json.dumps(log_data)}")
    
    def log_command(self, result: CommandResult):
        """Log command execution"""
        log_data = {
            "event": "ssh_command",
            "host": result.host,
            "command": result.command,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "success": result.success,
            "timestamp": result.timestamp.isoformat(),
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr)
        }
        
        if result.success:
            self.logger.info(f"SSH Command: {json.dumps(log_data)}")
        else:
            self.logger.error(f"SSH Command Failed: {json.dumps(log_data)}")

class SSHConnectionPool:
    """Manages SSH connection pooling and reuse"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, paramiko.SSHClient] = {}
        self.connection_times: Dict[str, datetime] = {}
        self.lock = threading.Lock()
    
    def _get_connection_key(self, connection: SSHConnection) -> str:
        """Generate unique key for connection"""
        return f"{connection.username}@{connection.host}:{connection.port}"
    
    def get_connection(self, connection: SSHConnection) -> Optional[paramiko.SSHClient]:
        """Get existing connection from pool"""
        with self.lock:
            key = self._get_connection_key(connection)
            if key in self.connections:
                # Check if connection is still alive
                try:
                    transport = self.connections[key].get_transport()
                    if transport and transport.is_active():
                        return self.connections[key]
                    else:
                        # Remove dead connection
                        del self.connections[key]
                        del self.connection_times[key]
                except:
                    # Remove problematic connection
                    if key in self.connections:
                        del self.connections[key]
                    if key in self.connection_times:
                        del self.connection_times[key]
            return None
    
    def add_connection(self, connection: SSHConnection, client: paramiko.SSHClient):
        """Add connection to pool"""
        with self.lock:
            key = self._get_connection_key(connection)
            
            # Remove oldest connection if at max capacity
            if len(self.connections) >= self.max_connections:
                oldest_key = min(self.connection_times.keys(), 
                               key=lambda k: self.connection_times[k])
                try:
                    self.connections[oldest_key].close()
                except:
                    pass
                del self.connections[oldest_key]
                del self.connection_times[oldest_key]
            
            self.connections[key] = client
            self.connection_times[key] = datetime.now()
    
    def cleanup_old_connections(self, max_age_minutes: int = 30):
        """Clean up old connections"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            keys_to_remove = []
            
            for key, connection_time in self.connection_times.items():
                if connection_time < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    self.connections[key].close()
                except:
                    pass
                del self.connections[key]
                del self.connection_times[key]

class EnhancedSSHManager:
    """Enhanced SSH Manager with banking-grade security"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.security_manager = SSHSecurityManager()
        self.audit_logger = SSHAuditLogger()
        self.connection_pool = SSHConnectionPool()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self.cleanup_thread.start()
    
    def _cleanup_worker(self):
        """Background worker for connection cleanup"""
        while True:
            try:
                self.connection_pool.cleanup_old_connections()
                time.sleep(300)  # Clean up every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(60)
    
    async def create_connection(self, connection: SSHConnection) -> paramiko.SSHClient:
        """Create new SSH connection with security validation"""
        
        # Check for existing connection
        existing_client = self.connection_pool.get_connection(connection)
        if existing_client:
            return existing_client
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(AutoAddPolicy())
        
        try:
            # Configure connection parameters
            connect_kwargs = {
                'hostname': connection.host,
                'port': connection.port,
                'username': connection.username,
                'timeout': connection.timeout,
                'compress': True,
                'look_for_keys': False,
                'allow_agent': False
            }
            
            # Handle authentication
            if connection.key_path and os.path.exists(connection.key_path):
                # Key-based authentication
                if connection.key_path.endswith('.pem') or 'rsa' in connection.key_path.lower():
                    key = RSAKey.from_private_key_file(connection.key_path)
                else:
                    key = Ed25519Key.from_private_key_file(connection.key_path)
                connect_kwargs['pkey'] = key
            elif connection.password:
                # Password authentication
                connect_kwargs['password'] = connection.password
            else:
                raise ValueError("No authentication method provided")
            
            # Establish connection
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.connect(**connect_kwargs)
            )
            
            # Add to connection pool
            self.connection_pool.add_connection(connection, client)
            
            # Log successful connection
            self.audit_logger.log_connection(connection, True)
            
            return client
            
        except Exception as e:
            self.audit_logger.log_connection(connection, False, str(e))
            try:
                client.close()
            except:
                pass
            raise Exception(f"SSH connection failed: {str(e)}")
    
    async def execute_command(
        self, 
        connection: SSHConnection, 
        command: str,
        timeout: int = 300
    ) -> CommandResult:
        """Execute command with comprehensive security and logging"""
        
        start_time = time.time()
        
        # Validate command
        is_valid, validation_message = self.security_manager.validate_command(
            command, connection.security_level
        )
        
        if not is_valid:
            result = CommandResult(
                command=command,
                stdout="",
                stderr=f"Security validation failed: {validation_message}",
                exit_code=-1,
                execution_time=0,
                timestamp=datetime.now(),
                host=connection.host,
                success=False
            )
            self.audit_logger.log_command(result)
            return result
        
        try:
            # Get SSH client
            client = await self.create_connection(connection)
            
            # Execute command
            stdin, stdout, stderr = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.exec_command(command, timeout=timeout)
            )
            
            # Read output
            stdout_data = await asyncio.get_event_loop().run_in_executor(
                None, stdout.read
            )
            stderr_data = await asyncio.get_event_loop().run_in_executor(
                None, stderr.read
            )
            
            exit_code = stdout.channel.recv_exit_status()
            execution_time = time.time() - start_time
            
            result = CommandResult(
                command=command,
                stdout=stdout_data.decode('utf-8', errors='replace'),
                stderr=stderr_data.decode('utf-8', errors='replace'),
                exit_code=exit_code,
                execution_time=execution_time,
                timestamp=datetime.now(),
                host=connection.host,
                success=exit_code == 0
            )
            
            # Log command execution
            self.audit_logger.log_command(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = CommandResult(
                command=command,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                execution_time=execution_time,
                timestamp=datetime.now(),
                host=connection.host,
                success=False
            )
            self.audit_logger.log_command(result)
            return result
    
    async def execute_script(
        self, 
        connection: SSHConnection, 
        script_content: str,
        script_name: str = "temp_script.sh"
    ) -> CommandResult:
        """Execute a script file remotely"""
        
        # Upload script
        upload_command = f"cat > /tmp/{script_name} << 'EOF'\n{script_content}\nEOF"
        upload_result = await self.execute_command(connection, upload_command)
        
        if not upload_result.success:
            return upload_result
        
        # Make executable and run
        chmod_result = await self.execute_command(
            connection, f"chmod +x /tmp/{script_name}"
        )
        
        if not chmod_result.success:
            return chmod_result
        
        # Execute script
        return await self.execute_command(connection, f"/tmp/{script_name}")
    
    async def file_transfer(
        self, 
        connection: SSHConnection,
        local_path: str,
        remote_path: str,
        upload: bool = True
    ) -> bool:
        """Transfer files securely"""
        
        try:
            client = await self.create_connection(connection)
            sftp = client.open_sftp()
            
            if upload:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sftp.put(local_path, remote_path)
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sftp.get(remote_path, local_path)
                )
            
            sftp.close()
            return True
            
        except Exception as e:
            logger.error(f"File transfer failed: {e}")
            return False
    
    def close_all_connections(self):
        """Close all pooled connections"""
        with self.connection_pool.lock:
            for client in self.connection_pool.connections.values():
                try:
                    client.close()
                except:
                    pass
            self.connection_pool.connections.clear()
            self.connection_pool.connection_times.clear()

# Example usage and testing
if __name__ == "__main__":
    async def test_ssh_manager():
        ssh_manager = EnhancedSSHManager()
        
        # Test connection
        connection = SSHConnection(
            host="mcp.xplaincrypto.ai",
            port=22,
            username="root",
            key_path="/path/to/private/key",
            security_level=SecurityLevel.HIGH
        )
        
        # Test command execution
        result = await ssh_manager.execute_command(
            connection, "ls -la /home"
        )
        
        print(f"Command: {result.command}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Output: {result.stdout}")
        print(f"Errors: {result.stderr}")
        print(f"Success: {result.success}")
        
        ssh_manager.close_all_connections()
    
    # Run test
    # asyncio.run(test_ssh_manager())
