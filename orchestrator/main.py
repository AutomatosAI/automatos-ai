
"""
Main FastAPI Application for Automotas AI
=========================================

Comprehensive API server with WebSocket support for real-time updates.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import database and models
from database import init_database, get_db
from models import Base

# Import API routers
from api.agents import router as agents_router
from api.workflows import router as workflows_router
from api.documents import router as documents_router
from api.system import router as system_router
from api.context import router as context_engineering_router
from api.skills import router as skills_router
from api.patterns import router as patterns_router
from api.statistics import router as statistics_router
from api.templates import router as templates_router

# Import WebSocket manager
from websocket_manager import manager, WebSocketEventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Automotas AI API Server...")
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Automotas AI API Server...")

# Create FastAPI app
app = FastAPI(
    title="Automotas AI API",
    description="Comprehensive API for Automotas AI agent management, workflows, and document processing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(agents_router)
app.include_router(workflows_router)
app.include_router(documents_router)
app.include_router(system_router)
app.include_router(context_engineering_router)
app.include_router(skills_router)
app.include_router(patterns_router)
app.include_router(statistics_router)
app.include_router(templates_router)


# Testing helper endpoint
@app.get("/api/test-data")
async def get_test_data():
    """Helper endpoint that returns available test data for API testing"""
    return {
        "agents": {"available_ids": [1, 2, 3, 4], "valid_types": ["code_architect", "security_expert", "performance_optimizer", "custom"]},
        "workflows": {"available_ids": [1, 2, 3]},
        "example_payloads": {"agent_create": {"name": "Test Agent", "agent_type": "code_architect"}}
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "automotas-ai-api"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Automotas AI API", "version": "1.0.0", "docs": "/docs"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(None, description="Optional client identifier")
):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connection_established",
            "data": {
                "message": "Connected to Automotas AI",
                "client_id": client_id,
                "features": [
                    "agent_updates",
                    "workflow_progress",
                    "document_processing",
                    "system_notifications"
                ]
            }
        }, websocket)
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            
            try:
                import json
                message = json.loads(data)
                message_type = message.get("type")
                
                # Handle different message types
                if message_type == "ping":
                    await manager.update_last_ping(websocket)
                    await manager.send_personal_message({
                        "type": "pong",
                        "data": {"message": "pong"}
                    }, websocket)
                
                elif message_type == "subscribe":
                    # Handle subscription to specific events
                    events = message.get("data", {}).get("events", [])
                    await manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "data": {
                            "events": events,
                            "message": f"Subscribed to {len(events)} event types"
                        }
                    }, websocket)
                
                elif message_type == "get_status":
                    # Send current system status
                    await manager.send_personal_message({
                        "type": "system_status",
                        "data": {
                            "active_connections": manager.get_connection_count(),
                            "server_status": "running",
                            "features_available": True
                        }
                    }, websocket)
                
                else:
                    # Echo unknown messages
                    await manager.send_personal_message({
                        "type": "echo",
                        "data": message
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                }, websocket)
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "data": {"message": "Error processing message"}
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Testing helper endpoint
@app.get("/api/test-data")
async def get_test_data():
    """Helper endpoint that returns available test data for API testing"""
    return {
        "agents": {"available_ids": [1, 2, 3, 4], "valid_types": ["code_architect", "security_expert", "performance_optimizer", "custom"]},
        "workflows": {"available_ids": [1, 2, 3]},
        "example_payloads": {"agent_create": {"name": "Test Agent", "agent_type": "code_architect"}}
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "automotas-ai-api",
        "version": "1.0.0",
        "timestamp": "2025-07-26T00:00:00Z"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Automotas AI API Server",
        "version": "1.0.0",
        "documentation": "/docs",
        "websocket": "/ws",
        "health": "/health",
        "endpoints": {
            "agents": "/api/agents",
            "workflows": "/api/workflows", 
            "documents": "/api/documents",
            "system": "/api/system"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Testing helper endpoint
@app.get("/api/test-data")
async def get_test_data():
    """Helper endpoint that returns available test data for API testing"""
    return {
        "message": "Available test data for API testing",
        "agents": {
            "available_ids": [1, 2, 3, 4],
            "valid_types": ["code_architect", "security_expert", "performance_optimizer", "data_analyst", "infrastructure_manager", "custom", "system", "specialized"],
            "example_create_payload": {
                "name": "Test Agent",
                "description": "A test agent",
                "agent_type": "code_architect",
                "configuration": {"test": True}
            }
        },
        "workflows": {
            "available_ids": [1, 2, 3],
            "valid_statuses": ["draft", "active", "completed", "failed", "archived"],
            "example_create_payload": {
                "name": "Test Workflow",
                "description": "A test workflow",
                "workflow_definition": {"steps": ["init", "process", "complete"]},
                "status": "draft"
            }
        },
        "system_config": {
            "required_fields": ["config_key", "config_value"],
            "example_create_payload": {
                "config_key": "test_setting",
                "config_value": "test_value"
            }
        },
        "rag_config": {
            "required_fields": ["name"],
            "example_create_payload": {
                "name": "test_rag_config",
                "description": "Test RAG configuration"
            }
        }
    }
