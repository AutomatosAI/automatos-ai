"""
PRD-58: FutureAGI Service (Orchestrator Side)
==============================================

Thin HTTP client that routes ALL FutureAGI operations to the isolated
futureagi-worker service. No SDK dependency in the orchestrator.

The worker handles: assess, safety, optimize, live scoring.
This module handles: DB reads/writes, fire-and-forget dispatch, dataset collection.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

WORKER_URL = os.getenv("AGENT_OPT_WORKER_URL", "http://agent-opt-worker.railway.internal:8080")
WORKER_TIMEOUT = 120  # seconds for assess/safety
OPTIMIZE_TIMEOUT = 300  # seconds for optimization


def _extract_text(parts) -> str:
    """Extract plain text from chat message parts (JSON list of {type, text})."""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        texts = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                texts.append(p.get("text", ""))
            elif isinstance(p, str):
                texts.append(p)
        return " ".join(texts).strip()
    return ""


class FutureAGIService:
    """
    Routes all FutureAGI calls to the isolated worker service.
    Keeps DB operations (eval run storage, dataset collection) in the orchestrator.
    """

    _instance: Optional["FutureAGIService"] = None

    def __init__(self) -> None:
        self._available = False
        self._init()

    @classmethod
    def get_instance(cls) -> "FutureAGIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init(self) -> None:
        # Check if worker URL is configured (keys live on the worker now)
        self._available = bool(os.getenv("AGENT_OPT_WORKER_URL") or os.getenv("FUTUREAGI_API_KEY"))
        if self._available:
            logger.info(f"FutureAGI service ready (worker at {WORKER_URL})")
        else:
            logger.info("FutureAGI service disabled (no worker URL or API keys)")

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Worker HTTP calls
    # ------------------------------------------------------------------

    async def _call_worker(self, path: str, payload: Dict[str, Any], timeout: int = WORKER_TIMEOUT) -> Dict[str, Any]:
        """POST to the worker service and return JSON response."""
        url = f"{WORKER_URL}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                error_text = resp.text[:300]
                logger.warning(f"[worker] {path} error: {resp.status_code} {error_text}")
                return {"error": f"Worker error ({resp.status_code}): {error_text}"}
            return resp.json()
        except httpx.ConnectError:
            logger.warning(f"[worker] {path} not reachable at {WORKER_URL}")
            return {"error": "FutureAGI worker not available"}
        except httpx.TimeoutException:
            logger.warning(f"[worker] {path} timed out after {timeout}s")
            return {"error": f"Worker timed out after {timeout}s"}
        except Exception as e:
            logger.error(f"[worker] {path} failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Assess
    # ------------------------------------------------------------------

    async def assess_prompt(
        self,
        prompt_content: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available:
            return {"error": "FutureAGI not configured", "scores": {}}

        payload: Dict[str, Any] = {"prompt_content": prompt_content}
        if metrics:
            payload["metrics"] = metrics
        if test_cases:
            payload["test_input"] = test_cases[0].get("input")
            payload["test_output"] = test_cases[0].get("output", test_cases[0].get("expected_output"))

        return await self._call_worker("/assess", payload)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    async def safety_check(self, prompt_content: str) -> Dict[str, Any]:
        if not self.is_available:
            return {"error": "FutureAGI not configured", "safe": None}

        return await self._call_worker("/safety", {"prompt_content": prompt_content})

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "meta_prompt",
        target_metric: str = "is_helpful",
        num_iterations: int = 3,
    ) -> Dict[str, Any]:
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        dataset = await self._collect_optimization_dataset()
        if not dataset:
            return {"error": "No live traffic data yet. Enable FutureAGI scoring and chat first to build a dataset."}

        payload = {
            "prompt_content": prompt_content,
            "dataset": dataset,
            "scoring_template": target_metric,
            "algorithm": algorithm,
            "num_rounds": num_iterations,
            "teacher_model": "gpt-4o-mini",
        }

        result = await self._call_worker("/optimize", payload, timeout=OPTIMIZE_TIMEOUT)

        # Normalize response keys for frontend
        if "optimized_prompt" in result:
            result["best_score"] = result.pop("final_score", None)
            result["rounds"] = result.pop("rounds_completed", None)
            result["duration"] = result.pop("duration_seconds", None)

        return result

    # ------------------------------------------------------------------
    # Live traffic scoring (fire-and-forget from chat pipeline)
    # ------------------------------------------------------------------

    LIVE_METRICS = ["completeness", "is_helpful", "is_concise"]

    async def eval_live_traffic(
        self,
        input_text: str,
        output_text: str,
        context_text: Optional[str] = None,
    ) -> None:
        """
        Fire-and-forget after each chat response.
        Finds all prompts with eval enabled, scores via worker, stores results.
        """
        if not self.is_available:
            return

        from core.database.database import SessionLocal
        from core.models.system_prompts import SystemPrompt, SystemPromptEvalRun, SystemPromptVersion

        db = SessionLocal()
        try:
            enabled_prompts = db.query(SystemPrompt).filter(
                SystemPrompt.futureagi_eval_enabled == True  # noqa: E712
            ).all()
            if not enabled_prompts:
                return

            # Call worker to score the exchange
            payload: Dict[str, Any] = {
                "input_text": input_text,
                "output_text": output_text,
                "metrics": self.LIVE_METRICS,
            }
            if context_text:
                payload["context_text"] = context_text

            result = await self._call_worker("/score", payload)

            if "error" in result:
                logger.warning(f"[live] worker scoring failed: {result['error']}")
                return

            scores = result.get("scores", {})

            # Store a run for each enabled prompt
            for prompt in enabled_prompts:
                version = db.query(SystemPromptVersion).filter(
                    SystemPromptVersion.prompt_id == prompt.id,
                    SystemPromptVersion.status == "active",
                ).first()
                if not version:
                    continue

                run = SystemPromptEvalRun(
                    prompt_id=prompt.id,
                    version_id=version.id,
                    run_type="live",
                    status="completed",
                    scores={"scores": scores, "metrics_run": len(scores), "source": "live_traffic"},
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                )
                db.add(run)
                db.commit()
                score_summary = ", ".join(f"{k}={v.get('score')}" for k, v in scores.items())
                logger.info(f"[live] Scored {prompt.slug}: {score_summary}")

        except Exception as e:
            logger.warning(f"[live] eval_live_traffic failed: {e}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Dataset collection for optimization
    # ------------------------------------------------------------------

    async def _collect_optimization_dataset(self, limit: int = 20) -> List[Dict[str, str]]:
        """Collect recent chat input/output pairs from the messages table."""
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            from sqlalchemy import text
            rows = db.execute(text("""
                SELECT m1.parts, m2.parts
                FROM messages m1
                JOIN messages m2
                  ON m1.chat_id = m2.chat_id
                  AND m2.role = 'assistant'
                  AND m2.created_at = (
                      SELECT MIN(created_at) FROM messages
                      WHERE chat_id = m1.chat_id AND role = 'assistant'
                      AND created_at > m1.created_at
                  )
                WHERE m1.role = 'user'
                ORDER BY m1.created_at DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()

            dataset = []
            for user_parts, assistant_parts in rows:
                user_text = _extract_text(user_parts)
                assistant_text = _extract_text(assistant_parts)
                if user_text and assistant_text:
                    dataset.append({"input": user_text, "output": assistant_text})

            logger.info(f"[optimize] collected {len(dataset)} I/O pairs for dataset")
            return dataset
        except Exception as e:
            logger.warning(f"[optimize] dataset collection failed: {e}")
            return []
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Run orchestrator (called by admin_prompts.py)
    # ------------------------------------------------------------------

    async def run_assessment(self, run_id: str) -> None:
        """
        Process a SystemPromptEvalRun by its ID.
        Loads the run from DB, calls the worker, saves results.
        """
        from core.database.database import SessionLocal
        from core.models.system_prompts import SystemPromptEvalRun, SystemPromptVersion

        db = SessionLocal()
        try:
            run = db.query(SystemPromptEvalRun).filter(
                SystemPromptEvalRun.id == run_id
            ).first()
            if not run:
                logger.error(f"Assessment run {run_id} not found")
                return

            run.status = "running"
            run.started_at = datetime.utcnow()
            db.commit()

            version = db.query(SystemPromptVersion).filter(
                SystemPromptVersion.id == run.version_id
            ).first()
            if not version:
                run.status = "failed"
                run.error_message = "Version not found"
                run.completed_at = datetime.utcnow()
                db.commit()
                return

            prompt_content = version.content
            config = run.metadata_ or {}

            if run.run_type == "assess":
                result = await self.assess_prompt(
                    prompt_content,
                    test_cases=config.get("test_cases"),
                    metrics=config.get("metrics"),
                )
            elif run.run_type == "optimize":
                result = await self.optimize_prompt(
                    prompt_content,
                    algorithm=config.get("algorithm", "meta_prompt"),
                    target_metric=config.get("target_metric", "is_helpful"),
                    num_iterations=config.get("num_iterations", 3),
                )
            elif run.run_type == "safety":
                result = await self.safety_check(prompt_content)
            else:
                result = await self.assess_prompt(prompt_content)

            if "error" in result and not result.get("scores") and not result.get("status"):
                run.status = "failed"
                run.error_message = result["error"]
            else:
                run.status = "completed"
                run.scores = result

                if run.run_type == "assess" and result.get("scores"):
                    version.eval_scores = result["scores"]

            run.completed_at = datetime.utcnow()
            logger.info(f"Assessment run {run_id} -> {run.status}")
            db.commit()

        except Exception as e:
            logger.error(f"Assessment run {run_id} failed: {e}")
            try:
                run.status = "failed"
                run.error_message = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()
            except Exception:
                pass
        finally:
            db.close()


# Module-level singleton
futureagi_service = FutureAGIService.get_instance()
