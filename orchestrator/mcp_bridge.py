
"""
Enhanced MCP Bridge with SSH Integration and Two-Tier Support
============================================================

This enhanced MCP bridge provides secure command execution with comprehensive sandboxing,
structured logging, health monitoring, rate limiting, and advanced security features.
Now includes SSH integration and support for the two-tiered orchestration system.
"""

import asyncio
import json
import logging
import os
import time
import hashlib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import docker
from dotenv import load_dotenv

from ssh_manager import EnhancedSSHManager, SSHConnection, SecurityLevel
from ai_module_parser import AIModuleParser
from security import get_audit_logger, create_security_event, EventType

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_bridge.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class CommandExecution(Base):
    __tablename__ = "command_executions"
    
    id = Column(String, primary_key=True)
    command = Column(Text, nullable=False)
    user_id = Column(String)
    source_ip = Column(String)
    execution_time = Column(Float)
    exit_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(String)

class RateLimitEntry(Base):
    __tablename__ = "rate_limits"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    request_count = Column(Integer, default=1)
    window_start = Column(DateTime, default=datetime.utcnow)

# Pydantic models
class CommandRequest(BaseModel):
    command: str
    timeout: int = Field(default=300, ge=1, le=3600)
    working_directory: Optional[str] = None
    environment_variables: Optional[Dict[str, str]] = None
    security_level: str = Field(default="medium")

class WorkflowRequest(BaseModel):
    repository_url: str
    task_prompt: Optional[str] = None
    target_host: Optional[str] = None
    environment_variables: Optional[Dict[str, str]] = None

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float

class EnhancedMCPBridge:
    """Enhanced MCP Bridge with SSH integration"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Enhanced MCP Bridge",
            description="Multi-Agent Orchestration Bridge with SSH capabilities",
            version="2.0.0"
        )
        
        # Initialize components
        self.ssh_manager = EnhancedSSHManager()
        self.ai_module_parser = AIModuleParser()
        self.audit_logger = get_audit_logger()
        
        # Configuration
        self.config = {
            "api_key": os.getenv("API_KEY", "default-api-key"),
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379")),
            "database_url": os.getenv("DATABASE_URL", "sqlite:///mcp_bridge.db"),
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        }
        
        # Initialize database
        self.engine = create_engine(self.config["database_url"])
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize Redis
        try:
            self.redis_client = redis.Redis(
                host=self.config["redis_host"],
                port=self.config["redis_port"],
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker connection failed: {e}")
            self.docker_client = None
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: deque())
        
        # Security
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        
        # Startup time
        self.startup_time = time.time()
        
        # Setup routes
        self._setup_routes()
        self._setup_middleware()
        
        logger.info("Enhanced MCP Bridge initialized")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def security_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Log request
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.NETWORK_ACCESS,
                user_id=request.headers.get("X-User-ID", "anonymous"),
                source_ip=request.client.host,
                resource=str(request.url),
                action=request.method,
                result="started",
                details={
                    "user_agent": request.headers.get("User-Agent"),
                    "content_length": request.headers.get("Content-Length")
                }
            ))
            
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.NETWORK_ACCESS,
                user_id=request.headers.get("X-User-ID", "anonymous"),
                source_ip=request.client.host,
                resource=str(request.url),
                action=request.method,
                result="completed",
                details={
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            ))
            
            return response
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthCheckResponse(
                status="healthy",
                timestamp=datetime.now(),
                version="2.0.0",
                uptime_seconds=time.time() - self.startup_time
            )
        
        @self.app.post("/execute")
        async def execute_command(
            request: CommandRequest,
            api_key: str = Depends(self.api_key_header),
            http_request: Request = None
        ):
            """Execute command via SSH"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Rate limiting
            if not await self._check_rate_limit(http_request.client.host):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            try:
                # Create SSH connection
                ssh_connection = SSHConnection(
                    host=os.getenv("DEPLOY_HOST", "mcp.xplaincrypto.ai"),
                    port=int(os.getenv("DEPLOY_PORT", "22")),
                    username=os.getenv("DEPLOY_USER", "root"),
                    key_path=os.getenv("DEPLOY_KEY_PATH", "/root/.ssh/id_rsa"),
                    security_level=SecurityLevel[request.security_level.upper()]
                )
                
                # Execute command
                result = await self.ssh_manager.execute_command(
                    ssh_connection,
                    request.command,
                    timeout=request.timeout
                )
                
                # Store execution record
                await self._store_execution_record(
                    command=request.command,
                    user_id=http_request.headers.get("X-User-ID", "anonymous"),
                    source_ip=http_request.client.host,
                    execution_time=result.execution_time,
                    exit_code=result.exit_code,
                    success=result.success
                )
                
                return {
                    "success": result.success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/workflow")
        async def run_workflow(
            request: WorkflowRequest,
            api_key: str = Depends(self.api_key_header),
            http_request: Request = None
        ):
            """Run unified workflow (AI module or task prompt)"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Rate limiting
            if not await self._check_rate_limit(http_request.client.host):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            try:
                # Import orchestrator here to avoid circular imports
                from orchestrator import EnhancedTwoTierOrchestrator
                
                orchestrator = EnhancedTwoTierOrchestrator()
                
                result = await orchestrator.run_unified_workflow(
                    repository_url=request.repository_url,
                    task_prompt=request.task_prompt,
                    target_host=request.target_host,
                    environment_variables=request.environment_variables
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/workflows")
        async def list_workflows(
            api_key: str = Depends(self.api_key_header)
        ):
            """List active workflows"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                from orchestrator import EnhancedTwoTierOrchestrator
                
                orchestrator = EnhancedTwoTierOrchestrator()
                workflows = await orchestrator.list_active_workflows()
                
                return {"workflows": workflows}
                
            except Exception as e:
                logger.error(f"Failed to list workflows: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/workflows/{workflow_id}")
        async def get_workflow_status(
            workflow_id: str,
            api_key: str = Depends(self.api_key_header)
        ):
            """Get workflow status"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                from orchestrator import EnhancedTwoTierOrchestrator
                
                orchestrator = EnhancedTwoTierOrchestrator()
                status = await orchestrator.get_workflow_status(workflow_id)
                
                return status
                
            except Exception as e:
                logger.error(f"Failed to get workflow status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/workflows/{workflow_id}")
        async def stop_workflow(
            workflow_id: str,
            api_key: str = Depends(self.api_key_header)
        ):
            """Stop workflow"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                from orchestrator import EnhancedTwoTierOrchestrator
                
                orchestrator = EnhancedTwoTierOrchestrator()
                result = await orchestrator.stop_workflow(workflow_id)
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to stop workflow: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/security/report")
        async def get_security_report(
            hours: int = 24,
            api_key: str = Depends(self.api_key_header)
        ):
            """Get security report"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                report = self.audit_logger.generate_security_report(hours=hours)
                return report
                
            except Exception as e:
                logger.error(f"Failed to generate security report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/ai-module/template/{module_type}")
        async def get_ai_module_template(
            module_type: str,
            api_key: str = Depends(self.api_key_header)
        ):
            """Get AI module template"""
            
            # Validate API key
            if not await self._validate_api_key(api_key):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                from ai_module_parser import ModuleType
                
                module_type_enum = ModuleType(module_type)
                template = self.ai_module_parser.generate_template(module_type_enum)
                
                return {
                    "module_type": module_type,
                    "template": template
                }
                
            except Exception as e:
                logger.error(f"Failed to generate template: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key == self.config["api_key"]
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting"""
        current_time = time.time()
        window_start = current_time - self.config["rate_limit_window"]
        
        # Clean old requests
        self.rate_limits[client_ip] = deque([
            req_time for req_time in self.rate_limits[client_ip]
            if req_time > window_start
        ])
        
        # Check limit
        if len(self.rate_limits[client_ip]) >= self.config["rate_limit_requests"]:
            return False
        
        # Add current request
        self.rate_limits[client_ip].append(current_time)
        return True
    
    async def _store_execution_record(
        self,
        command: str,
        user_id: str,
        source_ip: str,
        execution_time: float,
        exit_code: int,
        success: bool
    ):
        """Store command execution record"""
        
        try:
            execution_id = hashlib.sha256(
                f"{command}{user_id}{time.time()}".encode()
            ).hexdigest()[:16]
            
            db = self.SessionLocal()
            try:
                record = CommandExecution(
                    id=execution_id,
                    command=command,
                    user_id=user_id,
                    source_ip=source_ip,
                    execution_time=execution_time,
                    exit_code=exit_code,
                    success=str(success)
                )
                db.add(record)
                db.commit()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to store execution record: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the MCP Bridge server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

# Standalone execution
if __name__ == "__main__":
    bridge = EnhancedMCPBridge()
    bridge.run()
