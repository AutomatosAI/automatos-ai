"""
CodeGraph API Endpoints
=======================

REST API for code indexing and intelligent search.
"""

import os
import logging
import asyncio
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field

from core.database.database import get_db
from modules.codegraph import CodeGraphService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/code-graph", tags=["code-graph"])

# Request/Response Models
class IndexGitHubRequest(BaseModel):
    """Request to index a GitHub repository"""
    project_name: str = Field(..., description="Unique project name")
    github_url: str = Field(..., description="GitHub repository URL")
    branch: str = Field(default="main", description="Branch to index")
    auth_token: Optional[str] = Field(None, description="GitHub auth token for private repos")
    exclude_patterns: Optional[List[str]] = Field(None, description="Additional exclude patterns")


class IndexResponse(BaseModel):
    """Response from indexing operation"""
    project_id: int
    project_name: str
    total_files: int
    total_symbols: int
    duration_seconds: float
    status: str


class SearchResponse(BaseModel):
    """Response from search operations"""
    project: str
    query: str
    count: int
    results: List[dict]
    execution_time_ms: float
    prompt_block: Optional[str] = None


class ProjectResponse(BaseModel):
    """Project metadata response"""
    id: int
    name: str
    source_type: str
    source_url: Optional[str]
    branch: Optional[str]
    language: Optional[str]
    total_files: int
    total_symbols: int
    total_relationships: int
    last_indexed: Optional[str]
    index_duration_seconds: Optional[float]
    status: str
    auto_reindex: bool
    created_at: Optional[str]


# Helper functions
def get_openai_key() -> str:
    """
    Lazy load OpenAI API key from credentials or environment.
    
    Note: CodeGraph currently uses OpenAI embeddings API directly.
    Future: Will support multiple embedding providers via LLM service settings.
    """
    try:
        from core.credentials.resolver import get_credential_resolver
        resolver = get_credential_resolver()
        # Try to get from LLM service settings first
        from core.llm.manager import get_provider_and_model_from_settings, get_credential_data
        
        provider, _ = get_provider_and_model_from_settings("codegraph")
        cred_data = get_credential_data(provider, service_name="codegraph")
        api_key = cred_data.get("api_key")
        
        if api_key:
            return api_key
        
        # Fallback to direct credential lookup
        return resolver.get_credential_field("development_openai", "api_key")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def get_codegraph_service(db: Session = Depends(get_db)) -> CodeGraphService:
    """
    Get CodeGraph service instance.
    
    Uses LLM service settings to determine API key source.
    Future: Will support multiple embedding providers based on codegraph.provider setting.
    """
    return CodeGraphService(db)


# Endpoints
@router.post("/index/github", response_model=IndexResponse)
async def index_github_repository(
    request: IndexGitHubRequest,
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    Index a GitHub repository
    
    **Example:**
    ```json
    {
        "project_name": "automatos-ai",
        "github_url": "https://github.com/AutomatosAI/automatos-ai.git",
        "branch": "main",
        "exclude_patterns": ["tests", "docs"]
    }
    ```
    
    **What it does:**
    - Clones the repository (shallow clone for speed)
    - Parses Python and TypeScript/JavaScript files
    - Extracts functions, classes, and symbols
    - Generates semantic embeddings
    - Stores in database for search
    """
    try:
        result = await service.index_github_project(
            project_name=request.project_name,
            github_url=request.github_url,
            branch=request.branch,
            auth_token=request.auth_token,
            exclude_patterns=request.exclude_patterns
        )
        
        return IndexResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error indexing GitHub repository: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/search/symbols", response_model=SearchResponse)
async def search_symbols(
    project: str = Query(..., description="Project name"),
    q: str = Query(..., description="Search query"),
    symbol_type: Optional[str] = Query(None, description="Filter by symbol type (function, class, etc.)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    Search for code symbols by name (exact/fuzzy matching)
    
    **Example:**
    ```
    GET /api/code-graph/search/symbols?project=automatos-ai&q=authenticate&limit=5
    ```
    
    **Search types:**
    - Exact match: `authenticate_user` finds `authenticate_user()`
    - Fuzzy match: `auth` finds `authenticate_user()`, `AuthService`, etc.
    - Qualified: `api.auth` finds symbols in `api/auth.py`
    """
    try:
        results = await service.search_symbols(
            project_name=project,
            query=q,
            symbol_type=symbol_type,
            limit=limit
        )
        
        return SearchResponse(**results, prompt_block=None)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search/semantic", response_model=SearchResponse)
async def search_semantic(
    project: str = Query(..., description="Project name"),
    q: str = Query(..., description="Semantic search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    Semantic search using vector similarity
    
    **Example:**
    ```
    GET /api/code-graph/search/semantic?project=automatos-ai&q=user authentication flow
    ```
    
    **How it works:**
    - Converts your query to a vector embedding
    - Finds code with similar semantic meaning
    - Returns ranked results with relevance scores
    
    **Use cases:**
    - "How do I validate user input?" → finds validation functions
    - "Database connection setup" → finds DB initialization code
    - "Error handling patterns" → finds try/catch blocks
    """
    try:
        results = await service.semantic_search(
            project_name=project,
            query=q,
            limit=limit
        )
        
        return SearchResponse(**results)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    List all indexed projects
    
    Returns metadata for all projects including:
    - Indexing statistics
    - File and symbol counts
    - Last indexed timestamp
    - Project status
    """
    try:
        projects = service.list_projects()
        return [ProjectResponse(**p) for p in projects]
        
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """Get project details by ID"""
    try:
        projects = service.list_projects()
        project = next((p for p in projects if p["id"] == project_id), None)
        
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        return ProjectResponse(**project)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    Delete a project and all associated data
    
    **Warning:** This permanently deletes:
    - All symbols and code snippets
    - All relationships and call graphs
    - All search history
    - File metadata
    
    This action cannot be undone.
    """
    try:
        result = service.delete_project(project_id)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")


@router.post("/projects/{project_id}/reindex")
async def reindex_project(
    project_id: int,
    service: CodeGraphService = Depends(get_codegraph_service)
):
    """
    Re-index an existing project (runs in background)
    
    **Use cases:**
    - Code has been updated
    - Need to refresh embeddings
    - Fix indexing errors
    
    **Note:** Existing data will be replaced. Returns immediately, indexing runs in background.
    """
    try:
        # Get project details
        projects = service.list_projects()
        project = next((p for p in projects if p["id"] == project_id), None)
        
        if not project:
            raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
        
        if project["source_type"] != "github":
            raise HTTPException(
                status_code=400,
                detail="Re-indexing only supported for GitHub projects"
            )
        
        # Store project details for background task
        project_name = project["name"]
        github_url = project["source_url"]
        branch = project["branch"] or "main"
        
        # Update status to indexing immediately
        service.db.execute(
            text("UPDATE codegraph_projects SET status = 'indexing', updated_at = NOW() WHERE id = :id"),
            {"id": project_id}
        )
        service.db.commit()
        
        # Run indexing in background using asyncio task
        async def run_indexing():
            db = None
            try:
                # Create new service instance with new DB session for background task
                db = next(get_db())
                bg_service = CodeGraphService(db)
                await bg_service.index_github_project(
                    project_name=project_name,
                    github_url=github_url,
                    branch=branch
                )
            except Exception as e:
                logger.error(f"Background indexing failed for project {project_id}: {e}", exc_info=True)
                # Update status to failed
                try:
                    if db:
                        db.execute(
                            text("UPDATE codegraph_projects SET status = 'failed', updated_at = NOW() WHERE id = :id"),
                            {"id": project_id}
                        )
                        db.commit()
                    else:
                        # Get new session if db wasn't created
                        db = next(get_db())
                        db.execute(
                            text("UPDATE codegraph_projects SET status = 'failed', updated_at = NOW() WHERE id = :id"),
                            {"id": project_id}
                        )
                        db.commit()
                except Exception as update_error:
                    logger.error(f"Failed to update project status: {update_error}")
            finally:
                if db:
                    db.close()
        
        # Use asyncio.create_task for proper async background execution
        asyncio.create_task(run_indexing())
        
        return {
            "message": "Re-indexing started in background",
            "project_id": project_id,
            "status": "indexing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting re-indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start re-indexing: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "codegraph",
        "openai_configured": bool(get_openai_key())
    }


@router.get("/call-graph")
async def get_call_graph_api(
    project: str = Query(..., description="Project name"),
    symbol: str = Query(..., description="Qualified name of the root symbol"),
    depth: int = Query(1, ge=1, le=5, description="Depth of the call graph traversal"),
    direction: str = Query("outgoing", description="Traversal direction: 'outgoing', 'incoming', 'both'"),
    db: Session = Depends(get_db)
):
    """
    Retrieve the call graph or dependency tree for a given symbol.
    Returns nodes and edges for visualization of code relationships.
    
    Example: /api/code-graph/call-graph?project=Automatos-ai&symbol=AgentFactory&depth=2
    """
    try:
        codegraph_service = CodeGraphService(db)
        call_graph = await codegraph_service.get_call_graph(project, symbol, depth, direction)
        return call_graph
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Call graph error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve call graph: {e}")


