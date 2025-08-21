from __future__ import annotations

from typing import Optional
import os
import pathlib
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import get_db
from ..services_code_graph_builder import CodeGraphBuilder
from ..services_code_context_retrieval import CodeContextRetrieval


router = APIRouter(prefix="/codegraph", tags=["code-graph"])


class IndexRequest(BaseModel):
    project: str
    root_dir: str


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


@router.get("/search")
def search_symbols(project: str, q: str, limit: int = 12, db: Session = Depends(get_db)):
    try:
        retr = CodeContextRetrieval(db=db, project=project)
        symbols = retr.find_symbols(query_terms=q.split(), limit=limit)
        bundle = retr.expand_with_edges(symbols, max_neighbors=8)
        block = retr.to_prompt_block(bundle)
        return {"prompt_block": block, "count": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


