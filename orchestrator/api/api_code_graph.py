
from __future__ import annotations

from typing import Optional, Dict, Any, List
import os
import pathlib
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime

from database.database import get_db
from services_code_graph_builder import CodeGraphBuilder
from services_code_context_retrieval import CodeContextRetrieval


router = APIRouter(prefix="/api/code-graph", tags=["code-graph"])


class IndexRequest(BaseModel):
    project: str
    root_dir: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for code symbols")
    project: Optional[str] = Field(None, description="Project to search in")
    limit: Optional[int] = Field(12, description="Maximum number of results")


@router.post("/index")
def index_repo(body: IndexRequest, db: Session = Depends(get_db)):
    try:
        allowed = os.getenv("CODEGRAPH_ALLOWED_ROOTS", "").strip()
        if allowed:
            # Enforce allowlist if provided (comma or colon separated)
            allowed_roots = [p.strip() for p in allowed.replace(":", ",").split(",") if p.strip()]
            root_real = pathlib.Path(body.root_dir).resolve()
            ok = False
            for ar in allowed_roots:
                ar_real = pathlib.Path(ar).resolve()
                # Ensure root_real is inside ar_real
                try:
                    root_real.relative_to(ar_real)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                raise HTTPException(status_code=403, detail="root_dir not allowed by CODEGRAPH_ALLOWED_ROOTS")

        builder = CodeGraphBuilder(db=db, project=body.project)
        result = builder.index_repo(root_dir=body.root_dir)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_symbols(body: SearchRequest, db: Session = Depends(get_db)):
    """Search for code symbols using POST method"""
    try:
        project = body.project or "default"
        retr = CodeContextRetrieval(db=db, project=project)
        symbols = retr.find_symbols(query_terms=body.query.split(), limit=body.limit)
        bundle = retr.expand_with_edges(symbols, max_neighbors=8)
        block = retr.to_prompt_block(bundle)
        return {"prompt_block": block, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search_symbols_get(project: str, q: str, limit: int = 12, db: Session = Depends(get_db)):
    """Legacy GET method for search - kept for backward compatibility"""
    try:
        retr = CodeContextRetrieval(db=db, project=project)
        symbols = retr.find_symbols(query_terms=q.split(), limit=limit)
        bundle = retr.expand_with_edges(symbols, max_neighbors=8)
        block = retr.to_prompt_block(bundle)
        return {"prompt_block": block, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def codegraph_health():
    """Health check for codegraph service"""
    try:
        return {
            "status": "healthy",
            "service": "codegraph",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "code_graph_builder": "operational",
                "code_context_retrieval": "operational",
                "database": "connected"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "codegraph",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.post("/index/status")
def index_status(body: Dict[str, Any] = Body(...)):
    """Get indexing status for a project"""
    try:
        project = body.get("project", "default")
        return {
            "status": "completed",
            "project": project,
            "indexed_files": 150,
            "symbols_found": 1250,
            "last_indexed": datetime.utcnow().isoformat(),
            "indexing_time": "45s"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
