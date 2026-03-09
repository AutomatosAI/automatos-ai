"""
Main FastAPI Application for Automotas AI
=========================================

Comprehensive API server with WebSocket support for real-time updates. DO NOT COMMENT OUT ANYTHING IN THIS FILE.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from core.auth.hybrid import get_request_context_hybrid
import uvicorn
import uuid
import time
from datetime import datetime
from collections import defaultdict, deque

# Load .env file BEFORE anything else
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Import centralized config
from config import config

# Import database and models
from core.database.database import init_database, get_db, SessionLocal
from core.models import Base

# Import API routers
from api.agents import router as agents_router
from api.workflows import router as workflows_router
from api.workflow_templates import router as workflow_templates_router
from api.workflow_recipes import router as workflow_recipes_router, webhook_router as recipe_webhook_router
from api.webhooks import router as general_webhooks_router
from api.marketplace import router as marketplace_router
from api.documents import router as documents_router
from api.cache import router as cache_router
from api.system import router as system_router
from api.context_engineering import router as context_engineering_router
from api.memory import router as memory_router
from api.widget_memory import router as widget_memory_router  # US-013: Widget memory panel
from api.analytics import router as analytics_router
from api.workflow_history import router as workflow_history_router
from api.benchmarking import router as benchmarking_router
from api.memory_stats import router as memory_stats_router
from api.context_policy import router as context_policy_router
from api.codegraph import router as codegraph_router  # PRD-11: New CodeGraph implementation
from api.github_webhooks import router as github_webhooks_router  # GitHub PR automation
from api.api_playbooks import router as playbooks_router
from api.patterns import router as patterns_router
from api.context import router as context_router
from api.credentials import router as credentials_router  # PRD-18: Enhanced credentials
from api.system_settings import router as system_settings_router  # System Settings Management
from api.tools import router as tools_router
# PRD-36: Composio Integration (optional module)
try:
    from api.composio import router as composio_router
except ImportError:
    composio_router = None
# PRD-42: Cloud Document Sync with S3 Vectors (optional module)
try:
    from api.cloud_documents import router as cloud_documents_router
except ImportError:
    cloud_documents_router = None
from api.statistics import router as statistics_router
from api.permissions import router as permissions_router
from api.skills import router as skills_router
from api.templates import router as templates_router
from api.context_summarization import router as context_summarization_router  # Context Engineering 2.0
from api.team import router as team_router  # PRD-37: Team Management
from api.routing import router as routing_router  # PRD-50: Universal Orchestrator Router
from api.admin_plugins import router as admin_plugins_router  # PRD-42: Admin Plugin Marketplace
try:
    from api.admin_prompts import router as admin_prompts_router  # PRD-58: System Prompt Management
except ImportError:
    admin_prompts_router = None
from api.marketplace_plugins import router as marketplace_plugins_router  # PRD-42: Public Marketplace Plugins
from api.workspace_plugins import router as workspace_plugins_router  # PRD-42: Workspace Plugin Enablement
from api.workspace_skills import router as workspace_skills_router  # PRD-71: Workspace Skill Enablement
from api.agent_plugins import router as agent_plugins_router  # PRD-42: Agent Plugin Assignment
from api.personas import router as personas_router  # PRD-42: Persona API
from api.generated_images import router as generated_images_router  # Generated image serving
# Pilot Helper Widget: Jira bug reports (optional — Composio dependency)
try:
    from api.bug_reports import router as bug_reports_router
except ImportError:
    bug_reports_router = None
# US-012: Widget Email operations (optional — Composio dependency)
try:
    from api.widget_email import router as widget_email_router
except ImportError:
    widget_email_router = None
# PRD-37: SaaS Foundation stubs (optional — may not exist in all branches)
try:
    from api.auth import router as auth_router
except ImportError:
    auth_router = None
try:
    from api.api_keys import router as api_keys_router
except ImportError:
    api_keys_router = None
try:
    from api.evaluation import router as evaluation_router
except ImportError:
    evaluation_router = None
try:
    from api.widgets.router import router as widget_api_router
except ImportError:
    widget_api_router = None
try:
    from api.widget_marketplace import router as widget_marketplace_router
except ImportError:
    widget_marketplace_router = None

# PRD-56: Workspace Tasks
from api.tasks import router as tasks_router

# PRD-66: Workspace File Browser (Code Viewer Widget)
from api.workspace_files import router as workspace_files_router
# PRD-66: Workspace GitHub Integration (repo listing + cloning)
try:
    from api.workspace_github import router as workspace_github_router
except ImportError:
    logging.getLogger(__name__).warning("workspace_github router unavailable", exc_info=True)
    workspace_github_router = None

# Import MISSING API routers
from api.orchestrator import router as orchestrator_router
from api.analytics_api import router as analytics_api_router
from api.analytics_real import router as analytics_real_router
from api.insights import router as insights_router
from api.knowledge import router as knowledge_router
from api.knowledge_multimodal import router as knowledge_multimodal_router
from api.knowledge_graph import router as knowledge_graph_router
from api.learning import router as learning_router
from api.problems import router as problems_router
from api.query import router as query_router
from api.recommendations import router as recommendations_router
from api.solutions import router as solutions_router
from api.synthesis import router as synthesis_router
# WebSocket removed - using AI SDK SSE streaming instead
from api.chatbot_llm import router as chatbot_router
from api.chat import router as chat_router  # PRD-27: New streaming chat with history
# document_processing removed - use api/documents.py instead
from api.agent_endpoints import router as agent_endpoints_router
# PRD-37: Workspace context (optional module - may not exist in all branches)
try:
    from api.workspaces import router as workspaces_router
except ImportError:
    workspaces_router = None
# redis_websocket removed - using AI SDK SSE streaming instead
from api.models_endpoints import router as models_router  # PRD-15: Model management
from api.llm_marketplace import router as llm_marketplace_router  # PRD-54: LLM Marketplace
from api.openrouter_marketplace import router as openrouter_marketplace_router  # OpenRouter Model Cache
from api.llm_analytics import router as llm_analytics_router, admin_router as llm_admin_analytics_router  # PRD-54: LLM Analytics
from api.composio_analytics import router as composio_analytics_router  # PRD-54: Composio Analytics
from api.analytics_charts import router as analytics_charts_router  # PRD-54: PandasAI Charts
from api.user_api_keys import router as user_api_keys_router  # PRD-54: BYOK API Keys
from api.execution_history import router as execution_history_router  # Enhanced execution history
from api.database_knowledge import router as database_knowledge_router  # PRD-21: Database Knowledge
from api.database_analytics import router as database_analytics_router  # PRD-21: Real database analytics
from api.document_generation import router as document_generation_router  # PRD-63: Document Generation
from api.widget_workflows import router as widget_workflows_router  # US-014: Widget Workflow Control

# PRD-72: Activity Command Centre
try:
    from api.activity import router as activity_router
except ImportError:
    activity_router = None

# PRD-55: Autonomous Assistant Platform (optional modules)
try:
    from api.heartbeat import router as heartbeat_router
except ImportError:
    heartbeat_router = None
try:
    from api.channels import router as channels_router
except ImportError:
    channels_router = None

# PRD-72: Activity Command Centre
try:
    from api.activity import router as activity_router
except ImportError:
    activity_router = None

# PRD-74: Voice Chat
try:
    from api.chat_voice import router as chat_voice_router
except ImportError:
    chat_voice_router = None

# Import Dashboard Integration (PRD-06)
from api.dashboard_integration import (
    register_dashboard_routes,
    startup_dashboard,
    shutdown_dashboard
)

# WebSocket manager removed - using AI SDK SSE streaming instead
from core.utils.logging_adapter import (
    install_request_context_logging,
    set_request_id,
    clear_request_id,
    set_request_context,
    request_id_var,
    workspace_id_var,
    user_id_var,
    http_method_var,
    http_path_var,
)

# Configure logging — ships to log-relay → Loki when LOG_RELAY_URL is set
from core.monitoring.automatos_logging import setup_logging
setup_logging(service="automatos-backend")
logger = logging.getLogger(__name__)

# API Tracking (in-memory, last 100 calls per endpoint)
api_call_stats = defaultdict(lambda: {
    "call_count": 0,
    "total_time": 0,
    "avg_time": 0,
    "min_time": float('inf'),
    "max_time": 0,
    "recent_times": deque(maxlen=100),
    "error_count": 0,
    "last_called": None,
    "status_codes": defaultdict(int)
})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Automotas AI API Server...")
    try:
        # NOTE: Most tables are created by docker-compose on first install.
        # PRD-58: Ensure system_prompts tables exist + seed on every startup (idempotent).
        try:
            import core.models.system_prompts  # register models with Base.metadata
            from core.database.database import create_tables, get_db_session
            create_tables()
            # PRD-58 Phase 1B: Add futureagi_eval_enabled column if missing
            try:
                from sqlalchemy import text
                from core.database.database import engine
                with engine.connect() as conn:
                    conn.execute(text(
                        "ALTER TABLE system_prompts ADD COLUMN IF NOT EXISTS "
                        "futureagi_eval_enabled BOOLEAN NOT NULL DEFAULT FALSE"
                    ))
                    conn.commit()
            except Exception as col_err:
                logger.debug(f"Column migration check: {col_err}")
            from core.seeds.seed_system_prompts import seed_system_prompts
            with get_db_session() as db:
                seed_system_prompts(db)
        except Exception as e:
            logger.warning(f"PRD-58 table/seed init: {e}")

        # PRD-63: Ensure document_templates table exists + seed starter templates
        try:
            import core.models.core  # noqa: F811 — registers DocumentTemplate with Base
            from core.database.database import create_tables as _ct, get_db_session as _gdb
            _ct()  # idempotent — creates any missing tables
            from modules.documents.seed_templates import seed_starter_templates
            from core.models.workspaces import Workspace
            with _gdb() as db:
                workspace_ids = [w.id for w in db.query(Workspace.id).all()]
                for ws_id in workspace_ids:
                    seed_starter_templates(db, ws_id)
            logger.info(f"PRD-63: Document templates seeded for {len(workspace_ids)} workspace(s)")
        except Exception as e:
            logger.warning(f"PRD-63 template seed init: {e}")
        # PRD-64: Ensure semantic routing columns exist on agents table
        try:
            from sqlalchemy import text as _t64
            from core.database.database import engine as _engine64
            with _engine64.connect() as conn:
                conn.execute(_t64(
                    "ALTER TABLE agents ADD COLUMN IF NOT EXISTS "
                    "semantic_embedding JSONB"
                ))
                conn.execute(_t64(
                    "ALTER TABLE agents ADD COLUMN IF NOT EXISTS "
                    "semantic_text_hash VARCHAR(64)"
                ))
                conn.commit()
        except Exception as col_err:
            logger.debug(f"PRD-64 column migration check: {col_err}")

        # PRD-64: Seed semantic embeddings for agents (non-blocking fire-and-forget)
        try:
            import asyncio as _asyncio
            from core.routing.semantic_indexer import embed_workspace_agents as _embed_ws
            from core.models.workspaces import Workspace as _Workspace

            async def _embed_all_agents_on_startup():
                """Background task: embed agents in all workspaces."""
                try:
                    from core.database.database import SessionLocal as _SL
                    from core.llm.embedding_manager import get_embedding_manager
                    from core.models.core import Agent as _Agent

                    _db = _SL()
                    try:
                        # Log provider info for debugging
                        emgr = get_embedding_manager()
                        emgr._ensure_provider()
                        logger.info(f"PRD-64: Embedding provider: {emgr.get_provider_info()}")

                        ws_ids = [w.id for w in _db.query(_Workspace.id).all()]
                        total = 0
                        for ws_id in ws_ids:
                            try:
                                total += await _embed_ws(ws_id, _db)
                            except Exception:
                                logger.warning("PRD-64: Failed to embed workspace %s", ws_id, exc_info=True)

                        # Report embedding coverage
                        all_agents = _db.query(_Agent).filter(_Agent.status == "active").count()
                        with_embeddings = _db.query(_Agent).filter(
                            _Agent.status == "active",
                            _Agent.semantic_embedding.isnot(None),
                        ).count()
                        logger.info(
                            f"PRD-64: Semantic embeddings seeded — "
                            f"{total} new, {with_embeddings}/{all_agents} agents have embeddings"
                        )
                    finally:
                        _db.close()
                except Exception as e:
                    logger.warning(f"PRD-64: Startup embedding seed failed (non-fatal): {e}", exc_info=True)

            _asyncio.create_task(_embed_all_agents_on_startup())
        except Exception as e:
            logger.warning(f"PRD-64: Could not schedule startup embedding (non-fatal): {e}")

        logger.info("Database ready")

        # NOTE: Redis client uses lazy initialization via get_redis_client()
        # Services will auto-initialize on first use from environment variables
        logger.info("Redis client will lazy-initialize on first use")
        
        # Initialize Dashboard Services (PRD-06)
        await startup_dashboard(app)
        logger.info("Dashboard services initialized successfully")

        # Unified Scheduler: single fcntl lock guards heartbeat + recipe schedulers
        # Only one uvicorn worker acquires the lock — prevents 4x duplicate executions
        if config.HEARTBEAT_ENABLED or config.RECIPE_SCHEDULER_ENABLED:
            try:
                import fcntl
                lock_path = "/tmp/automatos_scheduler.lock"
                lock_file = open(lock_path, "w")
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # We got the lock — this worker owns ALL scheduled services
                    from services.scheduler import get_unified_scheduler
                    unified = get_unified_scheduler()
                    unified.start()
                    shared_sched = unified.apscheduler
                    app.state.scheduler_lock = lock_file  # keep file open to hold lock

                    if config.HEARTBEAT_ENABLED:
                        from services.heartbeat_service import get_heartbeat_service
                        await get_heartbeat_service().start(scheduler=shared_sched)
                        logger.info("HeartbeatService started on unified scheduler")

                    if config.RECIPE_SCHEDULER_ENABLED:
                        from services.recipe_scheduler import get_recipe_scheduler
                        await get_recipe_scheduler().start(scheduler=shared_sched)
                        logger.info("RecipeSchedulerService started on unified scheduler")

                    logger.info("Unified scheduler started (this worker owns it)")
                except BlockingIOError:
                    lock_file.close()
                    logger.info("Unified scheduler: another worker owns it, skipping")
            except Exception as e:
                logger.warning(f"Unified scheduler failed to start (non-fatal): {e}")

        # PRD-55: Start ChannelManager
        if config.CHANNELS_ENABLED:
            try:
                from channels.manager import get_channel_manager
                channel_mgr = get_channel_manager()
                await channel_mgr.start_all()
                logger.info("ChannelManager started successfully")
            except Exception as e:
                logger.warning(f"ChannelManager failed to start (non-fatal): {e}")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Automotas AI API Server...")

    # Stop unified scheduler (shuts down all heartbeat + recipe jobs at once)
    if config.HEARTBEAT_ENABLED or config.RECIPE_SCHEDULER_ENABLED:
        try:
            from services.scheduler import get_unified_scheduler
            get_unified_scheduler().stop()
            logger.info("Unified scheduler stopped")
        except Exception:
            pass

    # PRD-55: Stop ChannelManager
    if config.CHANNELS_ENABLED:
        try:
            from channels.manager import get_channel_manager
            await get_channel_manager().stop_all()
            logger.info("ChannelManager stopped")
        except Exception:
            pass

    # Shutdown Dashboard Services (PRD-06)
    await shutdown_dashboard(app)
    logger.info("Dashboard services shutdown complete")

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title="🤖 Automatos AI API",
    description="""
    ## 🚀 Comprehensive API for Automotas AI Platform
    
    > **World's Most Advanced Multi-Agent AI Orchestration Platform**
    
    ### 🎯 Core Features & Capabilities
    
    #### 🤖 **Agent Management**
    - **Agent Types**: Custom, System, and Specialized agents
    - **Agent Configuration**: Dynamic configuration with performance metrics
    - **Skills Integration**: Cognitive, technical, and communication skills
    - **Pattern Recognition**: Coordination, communication, and decision patterns
    
    #### 👥 **Multi-Agent Systems** 
    - **Collaborative Reasoning**: Consensus mechanisms & conflict resolution
    - **Agent Coordination**: Sequential, parallel, hierarchical, mesh, and adaptive strategies
    - **Behavior Monitoring**: Real-time emergent behavior analysis
    - **System Optimization**: Multi-objective performance optimization
    
    #### 🌐 **Field Theory Integration**
    - **Field Representations**: Scalar, vector, and tensor fields
    - **Field Propagation**: Gradient-based influence propagation
    - **Context Interactions**: Mathematical field-based modeling
    - **Dynamic Management**: Real-time field evolution and optimization
    
    #### 🔄 **Workflow Orchestration**
    - **Workflow Design**: Visual workflow creation and management
    - **Execution Engine**: Robust workflow execution with monitoring
    - **Agent Assignment**: Dynamic agent allocation to workflow tasks
    - **Progress Tracking**: Real-time execution monitoring
    
    #### 📄 **Document Processing**
    - **RAG Integration**: Retrieval-Augmented Generation systems
    - **Document Analysis**: Advanced text analysis and processing
    - **Knowledge Extraction**: Intelligent information extraction
    - **Multi-format Support**: PDF, DOC, TXT, and more
    
    #### 🧠 **Context Engineering**
    - **Information Theory**: Shannon entropy, mutual information
    - **Vector Operations**: Embeddings, similarity, clustering
    - **Mathematical Foundations**: Probability theory, graph theory, optimization
    - **Statistical Analysis**: Advanced statistical modeling and analysis
    
    #### 📊 **Evaluation & Analytics**
    - **Performance Metrics**: Multi-dimensional agent evaluation
    - **Quality Assessment**: Comprehensive quality scoring
    - **Emergence Tracking**: Emergent capability monitoring
    - **System Analytics**: Real-time system performance analytics
    
    #### 🧩 **Memory Systems**
    - **Hierarchical Memory**: Multi-level memory architectures
    - **Memory Management**: Intelligent memory allocation and retrieval
    - **Context Storage**: Long-term context preservation
    - **Memory Optimization**: Efficient memory usage strategies
    
    ### 🔗 **API Endpoints Overview**
    
    | Endpoint Group | Base URL | Description |
    |---|---|---|
    | 🤖 **Agents** | `/api/agents` | Agent lifecycle management |
    | 👥 **Multi-Agent** | `/api/multi-agent` | Collaborative systems |
    | 🌐 **Field Theory** | `/api/field-theory` | Context field management |
    | 🔄 **Workflows** | `/api/workflows` | Workflow orchestration |
    | 📄 **Documents** | `/api/documents` | Document processing |
    | 🧠 **Context Engineering** | `/api/context-engineering` | Mathematical foundations |
    | 📊 **Evaluation** | `/api/evaluation` | System evaluation |
    | 🧩 **Memory** | `/api/memory` | Memory management |
    | ⚙️ **System** | `/api/system` | System configuration |
    
    ### 🔌 **Real-time Features**
    - **WebSocket Endpoint**: `/ws` - Real-time communication
    - **Behavior Monitoring**: `/api/multi-agent/behavior/monitor/realtime`
    - **Live Notifications**: System-wide event streaming
    
    ### 🎛️ **Quick Start**
    1. **Health Check**: `GET /health` - Verify system status
    2. **Create Agent**: `POST /api/agents` - Create your first agent
    3. **Add Skills**: `POST /api/agents/{id}/skills` - Enhance agent capabilities
    4. **Create Workflow**: `POST /api/workflows` - Design workflow
    5. **Execute**: `POST /api/workflows/{id}/execute` - Run workflow
    
    ### 📚 **Authentication**
    - **API Key**: Include `X-API-Key` header for authenticated requests
    - **Session-based**: WebSocket connections support session authentication
    
    ### ⚡ **Performance & Scaling**
    - **Load Balancing**: Automatic agent load balancing
    - **Horizontal Scaling**: Multi-instance support
    - **Caching**: Intelligent result caching
    - **Rate Limiting**: Configurable rate limits per endpoint
    
    ### 📝 **Response Formats**
    All endpoints return consistent JSON responses with:
    - `status`: Success/error status
    - `data`: Response payload
    - `message`: Human-readable description
    - `timestamp`: ISO 8601 timestamp
    
    ---
    
    **🌟 Ready to build the future of AI? Start exploring the endpoints below!**
    """,
    version="1.0.0",
    contact={
        "name": "Automatos AI Development Team",
        "url": "https://github.com/AutomatosAI/automatos-ai",
        "email": "developers@automotas.ai"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": config.API_URL or "http://localhost:8000",
            "description": "Production server"
        }
    ],
    lifespan=lifespan,
    docs_url="/docs" if config.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if config.ENVIRONMENT != "production" else None,
    openapi_url="/openapi.json" if config.ENVIRONMENT != "production" else None,
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "docExpansion": "none",
        "operationsSorter": "alpha",
        "filter": True,
        "tryItOutEnabled": True,
        "syntaxHighlight.activate": True,
        "syntaxHighlight.theme": "arta",
        "displayOperationId": True,
        "showMutatedRequest": True,
        "defaultModelRendering": "example",
        "defaultModelExpandDepth": 1,
        "defaultModelsExpandDepth": 1,
        "showExtensions": True,
        "showCommonExtensions": True
    }
)

# CORS middleware - use centralized config
# Parse and clean CORS origins (handle comma-separated list with whitespace)
cors_origins = [origin.strip() for origin in config.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]
logger.info(f"🌐 CORS configured with allowed origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Workspace-ID", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Routing-Agent-ID", "X-Routing-Confidence", "X-Routing-Type", "X-Routing-Reasoning", "X-Routing-Request-ID"],
)

# PRD-38.4: Widget SDK middleware
try:
    from api.widgets.cors import WidgetCORSMiddleware
    app.add_middleware(WidgetCORSMiddleware)
except ImportError:
    pass

try:
    from api.widgets.rate_limit import WidgetRateLimitMiddleware
    app.add_middleware(WidgetRateLimitMiddleware)
except ImportError:
    pass

# Rate limiting (US-017)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

def _get_real_client_ip(request) -> str:
    """Extract real client IP, respecting X-Forwarded-For behind reverse proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=_get_real_client_ip, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request body size limit middleware (10MB default, 50MB for uploads)
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
UPLOAD_PATHS = ("/api/documents/upload", "/api/admin/plugins/upload", "/api/documents/templates/upload")

@app.middleware("http")
async def limit_request_body(request, call_next):
    from starlette.responses import JSONResponse
    content_length = request.headers.get("content-length")
    limit = MAX_UPLOAD_SIZE if any(request.url.path.startswith(p) for p in UPLOAD_PATHS) else MAX_BODY_SIZE
    if content_length:
        try:
            if int(content_length) > limit:
                return JSONResponse(status_code=413, content={"detail": "Payload too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
    return await call_next(request)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if config.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return response

# Install logging context filter and add request-id middleware
install_request_context_logging()

@app.middleware("http")
async def add_request_id_middleware(request, call_next):
    inbound = request.headers.get("X-Request-ID")
    token = set_request_id(inbound or uuid.uuid4().hex[:12])

    # PRD-73 Phase 2: Enrich logs with request context (workspace, user, method, path)
    # ContextVars are captured automatically by LogRelayHandler — zero cost at call sites
    http_method_var.set(request.method)
    http_path_var.set(request.url.path)

    # Extract workspace_id and user_id from auth headers (best-effort, pre-auth)
    ws_header = request.headers.get("x-workspace-id", "") or request.headers.get("x-workspace", "")
    if ws_header and ws_header != "__all__":
        workspace_id_var.set(ws_header)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID") or request_id_var.get("")
        return response
    finally:
        clear_request_id(token)

@app.middleware("http")
async def api_tracking_middleware(request, call_next):
    """Track API calls and response times"""
    # Skip tracking for websockets, static files, and OPTIONS (CORS preflight)
    if request.url.path.startswith(("/ws/", "/static/", "/docs", "/openapi.json")) or request.method == "OPTIONS":
        return await call_next(request)
    
    start_time = time.time()
    status_code = 500  # Default to error
    
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        logger.error(f"Error in request {request.url.path}: {e}")
        raise
    finally:
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # in ms
        # Use route template (e.g. /api/agents/{agent_id}) instead of raw path
        # to avoid unbounded memory growth from path parameters
        try:
            route = request.scope.get("route")
            route_path = getattr(route, "path", None) if route else None
            endpoint = f"{request.method} {route_path or request.url.path}"

            # Cap stats dict size to prevent unbounded memory growth
            if endpoint not in api_call_stats and len(api_call_stats) > 500:
                pass  # Skip tracking but don't suppress the response
            else:
                stats = api_call_stats[endpoint]
                stats["call_count"] += 1
                stats["total_time"] += response_time
                stats["avg_time"] = stats["total_time"] / stats["call_count"]
                stats["min_time"] = min(stats["min_time"], response_time)
                stats["max_time"] = max(stats["max_time"], response_time)
                stats["recent_times"].append(response_time)
                stats["last_called"] = datetime.now().isoformat()
                stats["status_codes"][status_code] += 1

                if status_code >= 400:
                    stats["error_count"] += 1
        except Exception:
            pass  # Never let stats tracking break the response


# Include API routers
app.include_router(agents_router)
app.include_router(models_router)  # PRD-15: Model management
app.include_router(widget_workflows_router)  # US-014: Widget workflow control (pause/resume/cancel) — must be before workflows_router to avoid /{id} catch-all
app.include_router(workflows_router)
app.include_router(workflow_templates_router)  # Legacy - backward compatibility
app.include_router(workflow_recipes_router)  # US-009: Renamed from templates
app.include_router(recipe_webhook_router)  # Recipe webhook triggers (no auth)
app.include_router(general_webhooks_router)  # General workspace webhooks (no auth)
app.include_router(marketplace_router)  # Community Marketplace
app.include_router(document_generation_router)  # PRD-63: Must be BEFORE documents_router (has /templates, /generated specific routes that would otherwise be caught by documents_router's /{document_id} catch-all → 422)
app.include_router(documents_router)
app.include_router(cache_router)  # Cache management and monitoring
app.include_router(system_router)
app.include_router(context_engineering_router)
app.include_router(memory_router)
app.include_router(widget_memory_router)  # US-013: Widget memory panel (/api/memory)
app.include_router(memory_stats_router)  # Real memory stats from database
app.include_router(analytics_router)
app.include_router(workflow_history_router)
app.include_router(execution_history_router)  # Enhanced execution history API
app.include_router(benchmarking_router)  # Workflow and agent analytics
app.include_router(context_policy_router)
app.include_router(codegraph_router)  # PRD-11: CodeGraph
app.include_router(github_webhooks_router)  # GitHub PR automation
app.include_router(playbooks_router)
app.include_router(patterns_router)
app.include_router(context_router)
app.include_router(credentials_router)  # PRD-18: Enhanced credentials with management
app.include_router(system_settings_router)  # System Settings Management
app.include_router(tools_router)
if composio_router is not None:
    app.include_router(composio_router)  # PRD-36: Composio Integration (500+ tools)
if cloud_documents_router is not None:
    app.include_router(cloud_documents_router)  # PRD-42: Cloud Document Sync
app.include_router(statistics_router)
app.include_router(permissions_router)
app.include_router(skills_router)
app.include_router(templates_router)
app.include_router(context_summarization_router)  # Context Engineering 2.0: Self-baking

# Include MISSING API routers
app.include_router(orchestrator_router)
app.include_router(analytics_api_router)
app.include_router(analytics_real_router)
app.include_router(insights_router)
app.include_router(knowledge_router)
app.include_router(knowledge_multimodal_router)
app.include_router(knowledge_graph_router)
app.include_router(learning_router)
app.include_router(problems_router)
app.include_router(query_router)
app.include_router(recommendations_router)
app.include_router(solutions_router)
app.include_router(synthesis_router)
# WebSocket routers removed - using AI SDK SSE streaming
app.include_router(chatbot_router)  # Legacy chatbot endpoint (kept for backward compatibility)
app.include_router(chat_router)  # PRD-27: New streaming chat with SSE, history, and artifacts
# document_processing_router removed - api/documents.py handles all document processing
app.include_router(agent_endpoints_router)
if workspaces_router is not None:
    app.include_router(workspaces_router)  # PRD-37: Workspace context
app.include_router(database_knowledge_router)  # PRD-21: Database Knowledge
app.include_router(database_analytics_router)  # PRD-21: Database Analytics
app.include_router(tasks_router)  # PRD-56: Workspace task management
app.include_router(workspace_files_router)  # PRD-66: Workspace file browser
if workspace_github_router is not None:
    app.include_router(workspace_github_router)  # PRD-66: Workspace GitHub integration
app.include_router(team_router)  # PRD-37: Team Management
app.include_router(routing_router)  # PRD-50: Universal Orchestrator Router
app.include_router(admin_plugins_router)  # PRD-42: Admin Plugin Marketplace
if admin_prompts_router is not None:
    app.include_router(admin_prompts_router)  # PRD-58: System Prompt Management
app.include_router(marketplace_plugins_router)  # PRD-42: Public Marketplace Plugins
app.include_router(llm_marketplace_router)  # PRD-54: LLM Provider Marketplace
app.include_router(openrouter_marketplace_router)  # OpenRouter Model Cache (separate sync)
app.include_router(llm_analytics_router)  # PRD-54: LLM Usage Analytics
app.include_router(llm_admin_analytics_router)  # PRD-54: Admin Cost Analytics
app.include_router(composio_analytics_router)  # PRD-54: Composio Analytics
app.include_router(analytics_charts_router)  # PRD-54: PandasAI Charts
app.include_router(user_api_keys_router)  # PRD-54: BYOK API Key Management
app.include_router(workspace_plugins_router)  # PRD-42: Workspace Plugin Enablement
app.include_router(workspace_skills_router)  # PRD-71: Workspace Skill Enablement
app.include_router(agent_plugins_router)  # PRD-42: Agent Plugin Assignment
app.include_router(personas_router)  # PRD-42: Persona API
app.include_router(generated_images_router)  # Generated image serving from S3
if bug_reports_router is not None:
    app.include_router(bug_reports_router)  # Pilot Helper Widget: Jira bug reports
if widget_email_router is not None:
    app.include_router(widget_email_router)  # US-012: Widget Email operations
if auth_router is not None:
    app.include_router(auth_router)  # PRD-37: Auth endpoints
if api_keys_router is not None:
    app.include_router(api_keys_router)  # PRD-37: API key management
if widget_api_router is not None:
    app.include_router(widget_api_router)  # PRD-38.4: Widget SDK API
if widget_marketplace_router is not None:
    app.include_router(widget_marketplace_router)  # PRD-38.5: Widget Marketplace
if evaluation_router is not None:
    app.include_router(evaluation_router)  # Evaluation methodologies

# PRD-72: Activity Command Centre
if activity_router is not None:
    app.include_router(activity_router)

# PRD-55: Autonomous Assistant Platform
if heartbeat_router is not None:
    app.include_router(heartbeat_router)
if channels_router is not None:
    app.include_router(channels_router)
if activity_router is not None:
    app.include_router(activity_router)  # PRD-72: Activity Command Centre

# PRD-74: Voice Chat
if chat_voice_router is not None:
    app.include_router(chat_voice_router)

# PRD-73: Monitoring Stack Integration
# Prometheus /metrics endpoint + request instrumentation
try:
    from core.monitoring.automatos_metrics import setup_metrics
    setup_metrics(app, service_name="automatos-backend")
    logger.info("Prometheus /metrics endpoint enabled")
except Exception as e:
    logger.warning(f"Prometheus metrics disabled: {e}")

# AlertManager webhook ingest → infrastructure_alerts table
try:
    from core.monitoring.automatos_alerts import create_alerts_router
    alerts_router = create_alerts_router(get_db)
    app.include_router(alerts_router, prefix="/api")
    logger.info("Alert ingest endpoint enabled at /api/alerts/ingest")
except Exception as e:
    logger.warning(f"Alert ingest disabled: {e}")

# PRD-73 Phase 2: Loki log query API for SENTINEL investigation
try:
    from core.monitoring.automatos_logs_api import create_logs_router
    logs_router = create_logs_router()
    app.include_router(logs_router, prefix="/api")
    logger.info("Loki log query API enabled at /api/logs/query")
except Exception as e:
    logger.warning(f"Loki log query API disabled: {e}")

# PRD-74: Voice Chat
if chat_voice_router is not None:
    app.include_router(chat_voice_router)

# Register Dashboard Routes (PRD-06)
register_dashboard_routes(app)

# Include legacy routes (from existing api_routes.py)
try:
    from api_routes import app as legacy_routes
    app.mount("/legacy", legacy_routes)
except Exception as e:
    logger.warning(f"Could not mount legacy routes: {e}")

# Exports directory served through authenticated endpoint instead of open StaticFiles mount.
# See api/exports.py (or serve via pre-signed URLs in production).
from fastapi.responses import FileResponse

@app.get("/exports/{file_path:path}", tags=["Exports"])
async def serve_export(file_path: str, ctx = Depends(get_request_context_hybrid)):
    """Serve exported files (charts, etc.) with authentication."""
    from pathlib import Path as _Path
    safe_base = _Path("exports").resolve()
    requested = (safe_base / file_path).resolve()
    # Prevent path traversal
    if not str(requested).startswith(str(safe_base)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(requested)

# WebSocket endpoint removed - using AI SDK SSE streaming instead
# See consumers/workflows/streaming.py and consumers/chatbot/streaming.py

# Health check endpoint
@app.get("/health",
         summary="🏥 System Health Check",
         description="Get comprehensive system health status including all components and services",
         tags=["🏥 System Health"],
         response_description="Detailed system health information")
async def health_check():
    """
    System health check with real probes for database, config, and resources.

    **Status Values:**
    - `healthy`: All systems operational
    - `degraded`: Some issues but functional
    - `unhealthy`: Critical issues detected
    """
    import psutil
    from sqlalchemy import text as _text

    components = {"api_server": "healthy"}

    # Database probe
    try:
        db = SessionLocal()
        try:
            db.execute(_text("SELECT 1"))
            components["database"] = "healthy"
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Health: database check failed: {e}")
        components["database"] = "unhealthy"

    # Critical config check
    has_db_url = bool(config.DATABASE_URL)
    components["config"] = "healthy" if has_db_url else "degraded"

    # Real system metrics via psutil
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        metrics = {
            "cpu_percent": round(cpu_pct, 1),
            "memory_used_percent": round(mem.percent, 1),
            "memory_available_mb": round(mem.available / (1024 * 1024), 0),
        }
    except Exception:
        metrics = {}

    # Derive overall status
    statuses = list(components.values())
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return {
        "status": overall_status,
        "service": "automatos-ai-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components,
        "metrics": metrics,
    }

@app.get("/api/health/endpoints",
         summary="📡 API Endpoint Health",
         description="Get health statistics for all API endpoints including call counts and response times",
         tags=["🏥 System Health"],
         response_description="API endpoint health and performance statistics")
async def api_endpoint_health():
    """
    ## 📡 API Endpoint Health Statistics
    
    Returns real-time statistics for all API endpoints:
    - Call counts
    - Average, min, max response times
    - Status code distribution
    - Error rates
    - Last called timestamp
    """
    try:
        endpoints = []
        total_calls = 0
        total_errors = 0
        
        for endpoint, stats in api_call_stats.items():
            # Calculate health status based on error rate and response time
            error_rate = (stats["error_count"] / stats["call_count"] * 100) if stats["call_count"] > 0 else 0
            avg_time = stats["avg_time"]
            
            if error_rate > 10 or avg_time > 1000:
                health = "unhealthy"
            elif error_rate > 5 or avg_time > 500:
                health = "degraded"
            else:
                health = "healthy"
            
            endpoints.append({
                "endpoint": endpoint,
                "health": health,
                "call_count": stats["call_count"],
                "avg_response_time": round(stats["avg_time"], 2),
                "min_response_time": round(stats["min_time"], 2) if stats["min_time"] != float('inf') else 0,
                "max_response_time": round(stats["max_time"], 2),
                "error_count": stats["error_count"],
                "error_rate": round(error_rate, 2),
                "last_called": stats["last_called"],
                "status_codes": dict(stats["status_codes"])
            })
            
            total_calls += stats["call_count"]
            total_errors += stats["error_count"]
        
        # Sort by call count (most used first)
        endpoints.sort(key=lambda x: x["call_count"], reverse=True)
        
        # Calculate overall health
        overall_error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0
        overall_status = "healthy"
        if overall_error_rate > 5:
            overall_status = "degraded"
        if overall_error_rate > 10:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_endpoints": len(endpoints),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "overall_error_rate": round(overall_error_rate, 2),
                "healthy_endpoints": len([e for e in endpoints if e["health"] == "healthy"]),
                "degraded_endpoints": len([e for e in endpoints if e["health"] == "degraded"]),
                "unhealthy_endpoints": len([e for e in endpoints if e["health"] == "unhealthy"])
            },
            "endpoints": endpoints[:20]  # Top 20 most used endpoints
        }
        
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": "Service check failed",
            "message": "Failed to retrieve API health statistics"
        }

# Root endpoint
@app.get("/", 
         summary="🏠 API Overview & Navigation",
         description="Get comprehensive API information, endpoints overview, and quick navigation links",
         tags=["🏠 Getting Started"],
         response_description="API overview with navigation information")
async def root():
    """
    ## 🚀 Welcome to Automotas AI API
    
    This endpoint provides a comprehensive overview of all available API endpoints,
    documentation links, and system information to help developers get started quickly.
    """
    return {
        "service": "Automatos AI API Server",
        "version": "1.0.0",
        "status": "operational",
        "description": "World's Most Advanced Multi-Agent AI Orchestration Platform",
        
        "📚 documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/api/v1/openapi.json"
        },
        
        "🔌 real_time": {
            "websocket": "/ws",
            "behavior_monitoring": "/api/multi-agent/behavior/monitor/realtime"
        },
        
        "🏥 health_monitoring": {
            "system_health": "/health",
            "system_metrics": "/api/system/metrics",
        },
        
        "🛠️ api_endpoints": {
            "🤖 agents": {
                "base_url": "/api/agents",
                "description": "Complete agent lifecycle management",
                "features": ["Create agents", "Manage skills", "Performance tracking", "Agent coordination"]
            },
            "🔄 workflows": {
                "base_url": "/api/workflows",
                "description": "Workflow orchestration and execution",
                "features": ["Workflow design", "Execution engine", "Progress tracking", "Agent assignment"]
            },
            "📄 documents": {
                "base_url": "/api/documents",
                "description": "Document processing and analysis",
                "features": ["RAG integration", "Document analysis", "Knowledge extraction", "Multi-format support"]
            },
            "🧠 context_engineering": {
                "base_url": "/api/context-engineering",
                "description": "Mathematical foundations for intelligent processing",
                "features": ["Information theory", "Vector operations", "Statistical analysis", "Optimization algorithms"]
            },
            "📊 evaluation": {
                "base_url": "/api/evaluation",
                "description": "System evaluation and benchmarking",
                "features": ["Performance metrics", "Quality assessment", "Emergence tracking", "Analytics"]
            },
            "🧩 memory": {
                "base_url": "/api/memory",
                "description": "Advanced memory management systems",
                "features": ["Hierarchical memory", "Context storage", "Memory optimization", "Intelligent retrieval"]
            },
            "⚙️ system": {
                "base_url": "/api/system",
                "description": "System configuration and management",
                "features": ["Configuration management", "RAG systems", "System monitoring", "Health checks"]
            }
        },
        
        "🎯 quick_start": {
            "1": "GET /health - Check system status",
            "2": "POST /api/agents - Create your first agent", 
            "3": "GET /api/agents/{agent_id} - Retrieve agent details",
            "4": "POST /api/workflows - Create a workflow",
            "5": "POST /api/workflows/{workflow_id}/execute - Execute workflow"
        },
        
        "🔐 authentication": {
            "method": "API Key",
            "header": "X-API-Key",
            "websocket": "Session-based authentication supported"
        },
        
        "📞 support": {
            "documentation": "https://docs.automatos.ai",
            "github": "https://github.com/AutomatosAI/automatos-ai",
            "community": "https://community.automatos.ai"
        },
        
        "⚡ performance": {
            "load_balancing": "Automatic agent load balancing",
            "scaling": "Horizontal multi-instance support",
            "caching": "Intelligent result caching",
            "rate_limiting": "Configurable per endpoint"
        },
        
        "🕐 timestamp": datetime.utcnow().isoformat()
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
        reload=config.IS_DEVELOPMENT,
        log_level="info"
    )
