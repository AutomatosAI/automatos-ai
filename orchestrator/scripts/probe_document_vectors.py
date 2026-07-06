#!/usr/bin/env python3
"""PRD-185 S8: read-only production document-vector probe.

The Phase-2 review (finding P2-03) flagged that the committed prod config may not
be able to construct the F005-guarded S3 Vectors backend — which would leave the
document-vector plane *dark since W2*, so every agent answer over workspace
documents would be ungrounded. This script answers that, **without writing or
mutating anything**:

  1. Which vector backend does the committed config select?  (``S3_VECTORS_ENABLED``)
  2. Can that backend be CONSTRUCTED from the committed config?  (surfaces the
     F005 / missing-``{workspace_id}`` / missing-credentials failures)
  3. Is the document index POPULATED, and at what DIMENSION?  (read-only
     ``get_index`` + ``list_vectors``, or a pgvector row count)

It then prints a written finding and a recommendation. It performs NO destructive
action and NO re-embed — the fix-or-fold decision (relight the plane vs fold into
the Wave-3 Qdrant consolidation, P2-16) is **Gerard's call** per the PRD-185 §S8
gate; this probe only produces the evidence for it.

Run against a prod-configured environment (needs the same env + AWS creds the
API uses; DB reachable for the populated check)::

    python -m scripts.probe_document_vectors                 # workspaces with docs (up to --limit)
    python -m scripts.probe_document_vectors --workspace <uuid>
    python -m scripts.probe_document_vectors --all
    python -m scripts.probe_document_vectors --json          # machine-readable

This file is import-safe (no work at import time) so its pure decision logic —
``classify_plane`` / ``recommend`` — can be unit-tested with no DB / network.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pure decision logic (unit-tested, no I/O)
# ---------------------------------------------------------------------------

VERDICT_LIVE = "live"            # active plane is populated and usable
VERDICT_DEGRADED = "degraded"    # constructs + has data, but a dimension mismatch
VERDICT_DARK = "dark"            # active plane cannot be built, or is empty
VERDICT_UNKNOWN = "unknown"      # probe could not reach the backend to decide


def classify_plane(
    active_backend: str,
    constructable: bool,
    configured_dimension: int,
    probes: List[Dict[str, Any]],
) -> str:
    """Classify the document-vector plane from probe results. Pure.

    Args:
        active_backend: "s3_vectors" or "pgvector" — what the committed config selects.
        constructable: could the active backend be built from the committed config?
        configured_dimension: the embedding dimension the config expects.
        probes: per-workspace results, each a dict with keys:
            ``index_exists`` (bool), ``populated`` (bool), ``dimension`` (int|None),
            ``error`` (str|None).

    Returns one of VERDICT_*.
    """
    if not constructable:
        # The config can't even build the backend — the plane is dark by construction.
        return VERDICT_DARK

    reachable = [p for p in probes if not p.get("error")]
    if not reachable:
        return VERDICT_UNKNOWN

    populated = [p for p in reachable if p.get("populated")]
    if not populated:
        # Backend builds and indexes are reachable, but nothing is stored → dark.
        return VERDICT_DARK

    # Populated. Flag a dimension mismatch (embeddings unusable against a
    # differently-dimensioned index) as degraded, not healthy.
    for p in populated:
        dim = p.get("dimension")
        if dim and configured_dimension and dim != configured_dimension:
            return VERDICT_DEGRADED
    return VERDICT_LIVE


def recommend(verdict: str, active_backend: str) -> str:
    """Map a verdict to the PRD-185 §S8 gate recommendation. Pure."""
    if verdict == VERDICT_LIVE:
        return ("Plane is LIVE — grounding is real. Unblocks Wave-1 P2-07 (RAG "
                "quality). No relight needed.")
    if verdict == VERDICT_DEGRADED:
        return ("Plane has data but a DIMENSION MISMATCH — stored embeddings do "
                "not match the configured index dimension, so retrieval is "
                "silently wrong. Re-embed at the correct dimension before any "
                "RAG-quality work.")
    if verdict == VERDICT_DARK:
        if active_backend == "s3_vectors":
            return ("Plane is DARK — the S3 Vectors backend cannot be built from "
                    "the committed config, or its index is empty. Decision (§S8, "
                    "Gerard's call): fix env (S3_VECTORS_BUCKET must carry the "
                    "'{workspace_id}' placeholder + valid AWS creds) and re-embed "
                    "as a fast follow, OR fold into the Wave-3 Qdrant "
                    "consolidation (P2-16). RAG-quality work (P2-07) is gated on "
                    "this either way.")
        return ("Plane is DARK — pgvector holds no embedded documents. Ingestion "
                "has not populated the vector store; re-embed before relying on "
                "document grounding.")
    return ("Plane state UNKNOWN — the probe could not reach the backend. Re-run "
            "with valid credentials / DB access from a prod-configured shell.")


# ---------------------------------------------------------------------------
# I/O: config snapshot + read-only backend inspection
# ---------------------------------------------------------------------------

def _config_snapshot() -> Dict[str, Any]:
    """Read the committed config — secrets reported only as present/absent."""
    from config import config
    return {
        "active_backend": "s3_vectors" if config.S3_VECTORS_ENABLED else "pgvector",
        "S3_VECTORS_ENABLED": bool(config.S3_VECTORS_ENABLED),
        "S3_VECTORS_BUCKET": config.S3_VECTORS_BUCKET or "(unset)",
        "S3_VECTORS_BUCKET_has_placeholder": "{workspace_id}" in (config.S3_VECTORS_BUCKET or ""),
        "S3_VECTORS_INDEX_NAME": config.S3_VECTORS_INDEX_NAME,
        "S3_VECTORS_DIMENSION": config.S3_VECTORS_DIMENSION,
        "S3_VECTORS_METRIC": config.S3_VECTORS_METRIC,
        "AWS_REGION": config.AWS_REGION,
        "AWS_ACCESS_KEY_ID_present": bool(config.AWS_ACCESS_KEY_ID),
        "AWS_SECRET_ACCESS_KEY_present": bool(config.AWS_SECRET_ACCESS_KEY),
        "pgvector_dimension": config.VECTOR_STORE_DIMENSIONS,
    }


def _workspaces_with_documents(db, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Read-only: workspaces that have uploaded documents, with chunk totals."""
    from sqlalchemy import func
    from core.models.core import Document

    q = (
        db.query(
            Document.workspace_id,
            func.count(Document.id),
            func.coalesce(func.sum(Document.chunk_count), 0),
        )
        .group_by(Document.workspace_id)
        .order_by(func.count(Document.id).desc())
    )
    if limit:
        q = q.limit(limit)
    return [
        {"workspace_id": str(ws), "documents": int(docs), "chunks": int(chunks)}
        for ws, docs, chunks in q.all()
    ]


def _probe_s3_workspace(workspace_id: str) -> Dict[str, Any]:
    """Read-only S3 Vectors inspection for one workspace: get_index + list_vectors.

    Never calls initialize()/_ensure_setup() — that would CREATE buckets/indexes.
    """
    result: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "index_exists": False,
        "populated": False,
        "dimension": None,
        "vector_sample_count": 0,
        "error": None,
    }
    try:
        from botocore.exceptions import ClientError
        from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend

        backend = S3VectorsBackend(workspace_id=workspace_id)  # __init__ validates config + F005
        client = backend.client
        try:
            index = client.get_index(
                vectorBucketName=backend.bucket_name,
                indexName=backend.index_name,
            )
            result["index_exists"] = True
            result["dimension"] = index.get("dimension") or (index.get("index", {}) or {}).get("dimension")
        except ClientError as e:
            result["error"] = f"get_index: {e.response.get('Error', {}).get('Code', str(e))}"
            return result

        try:
            listed = client.list_vectors(
                vectorBucketName=backend.bucket_name,
                indexName=backend.index_name,
            )
            sample = listed.get("vectors", []) or []
            result["vector_sample_count"] = len(sample)
            result["populated"] = len(sample) > 0
        except ClientError as e:
            result["error"] = f"list_vectors: {e.response.get('Error', {}).get('Code', str(e))}"
    except Exception as e:  # construction / import / auth failure
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _probe_pgvector(db, workspaces: List[Dict[str, Any]], configured_dim: int) -> List[Dict[str, Any]]:
    """Read-only pgvector inspection: does each workspace have embedded chunks?

    ``documents.chunk_count`` is the ingestion-side truth for "was this embedded".
    """
    probes = []
    for w in workspaces:
        embedded = w["chunks"]
        probes.append({
            "workspace_id": w["workspace_id"],
            "index_exists": True,          # pgvector table always exists
            "populated": embedded > 0,
            "dimension": configured_dim,   # pgvector column dimension == configured
            "vector_sample_count": embedded,
            "error": None,
        })
    return probes


def _construct_check(active_backend: str) -> Dict[str, Any]:
    """Can the active backend be built from the committed config? Read-only."""
    if active_backend == "pgvector":
        return {"constructable": True, "error": None}
    try:
        from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
        # A throwaway probe workspace: __init__ runs the F005 / placeholder /
        # credential validation without touching AWS state.
        S3VectorsBackend(workspace_id="00000000-0000-0000-0000-000000000000")
        return {"constructable": True, "error": None}
    except Exception as e:
        return {"constructable": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_probe(workspace: Optional[str], probe_all: bool, limit: int) -> Dict[str, Any]:
    """Orchestrate the read-only probe and return a structured finding."""
    from core.database.database import SessionLocal

    cfg = _config_snapshot()
    active = cfg["active_backend"]
    construct = _construct_check(active)

    db = SessionLocal()
    try:
        if workspace:
            workspaces = [{"workspace_id": workspace, "documents": None, "chunks": None}]
        else:
            workspaces = _workspaces_with_documents(db, None if probe_all else limit)

        if active == "s3_vectors":
            probes = [_probe_s3_workspace(w["workspace_id"]) for w in workspaces] if construct["constructable"] else []
        else:
            # pgvector needs real doc/chunk counts; refetch if a single ws was named
            if workspace:
                workspaces = _workspaces_with_documents(db, None) or workspaces
                workspaces = [w for w in workspaces if w["workspace_id"] == workspace] or workspaces
            probes = _probe_pgvector(db, workspaces, cfg["pgvector_dimension"])
    finally:
        db.close()

    verdict = classify_plane(
        active_backend=active,
        constructable=construct["constructable"],
        configured_dimension=cfg["S3_VECTORS_DIMENSION"] if active == "s3_vectors" else cfg["pgvector_dimension"],
        probes=probes,
    )
    return {
        "config": cfg,
        "construct_check": construct,
        "workspaces_probed": len(probes),
        "probes": probes,
        "verdict": verdict,
        "recommendation": recommend(verdict, active),
    }


def _print_report(finding: Dict[str, Any]) -> None:
    cfg = finding["config"]
    print("=" * 72)
    print("PRD-185 S8 — Document-Vector Plane Probe (read-only)")
    print("=" * 72)
    print(f"\nActive backend (committed config): {cfg['active_backend']}")
    print("\nConfig snapshot:")
    for k, v in cfg.items():
        print(f"  {k:38} = {v}")

    c = finding["construct_check"]
    print(f"\nCan the committed config construct the backend? {c['constructable']}")
    if c["error"]:
        print(f"  construction error: {c['error']}")

    print(f"\nWorkspaces probed: {finding['workspaces_probed']}")
    for p in finding["probes"]:
        line = (f"  ws={p['workspace_id']}  index_exists={p['index_exists']}  "
                f"populated={p['populated']}  dimension={p['dimension']}  "
                f"sample={p['vector_sample_count']}")
        if p.get("error"):
            line += f"  ERROR={p['error']}"
        print(line)

    print(f"\n{'-' * 72}")
    print(f"VERDICT: {finding['verdict'].upper()}")
    print(f"RECOMMENDATION: {finding['recommendation']}")
    print(f"{'-' * 72}")
    print("\nThis probe is analysis-only. Relight-vs-fold is Gerard's call (PRD-185 §S8).")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PRD-185 S8 read-only document-vector probe")
    parser.add_argument("--workspace", help="Probe a single workspace UUID")
    parser.add_argument("--all", action="store_true", help="Probe every workspace with documents")
    parser.add_argument("--limit", type=int, default=5, help="Max workspaces to probe (default 5)")
    parser.add_argument("--json", action="store_true", help="Emit the structured finding as JSON")
    args = parser.parse_args(argv)

    finding = run_probe(workspace=args.workspace, probe_all=args.all, limit=args.limit)

    if args.json:
        import json
        print(json.dumps(finding, indent=2, default=str))
    else:
        _print_report(finding)

    # Exit non-zero when the plane is not healthy, so a CI/ops caller can gate on it.
    return 0 if finding["verdict"] == VERDICT_LIVE else 1


if __name__ == "__main__":
    sys.exit(main())
