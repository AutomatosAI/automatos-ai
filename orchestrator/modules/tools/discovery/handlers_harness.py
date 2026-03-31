"""HARNESS handler functions for PlatformActionExecutor (PRD-121)."""

import json
import logging
import os
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def harness_status(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return current HARNESS state."""
    from services.harness_service import get_harness_service

    try:
        status = get_harness_service().get_status(workspace_id)
        return {"success": True, "data": status}
    except Exception as exc:
        logger.error("[HARNESS] harness_status failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def harness_trigger(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Manually trigger a HARNESS run."""
    from services.harness_service import get_harness_service

    try:
        await get_harness_service().trigger_now(workspace_id)
        return {
            "success": True,
            "data": {
                "message": "HARNESS optimization run triggered. Results will appear in the Reports tab.",
            },
        }
    except Exception as exc:
        logger.error("[HARNESS] harness_trigger failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def harness_history(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """List past HARNESS runs from baseline archive files."""
    limit = params.get("limit", 10)

    try:
        runs: List[Dict[str, Any]] = []

        # Read baseline files from workspace storage
        from config import config

        baselines_dir = os.path.join(
            config.WORKSPACE_VOLUME_PATH,
            str(workspace_id),
            "harness",
            "baselines",
        )

        if os.path.isdir(baselines_dir):
            files = sorted(os.listdir(baselines_dir), reverse=True)[:limit]
            for fname in files:
                fpath = os.path.join(baselines_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        baseline = json.load(f)
                    conv = baseline.get("convergence", {})
                    runs.append({
                        "run_date": baseline.get("created_at", fname.replace(".json", "")),
                        "iteration": baseline.get("iteration", 0),
                        "prescription_count": len(baseline.get("applied_changes", []))
                            + len(baseline.get("queued_changes", [])),
                        "applied_count": len(baseline.get("applied_changes", [])),
                        "queued_count": len(baseline.get("queued_changes", [])),
                        "convergence_status": conv.get("status", "unknown"),
                        "total_delta_magnitude": conv.get("total_delta_magnitude", 0),
                    })
                except Exception as exc:
                    logger.warning("[HARNESS] Failed to parse baseline %s: %s", fname, exc)
                    continue

        return {"success": True, "data": {"runs": runs, "total": len(runs)}}

    except Exception as exc:
        logger.error("[HARNESS] harness_history failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
