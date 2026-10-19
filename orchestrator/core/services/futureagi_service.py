"""
PRD-58: FutureAGI Service (Orchestrator Side)
==============================================

Thin HTTP client that routes ALL FutureAGI operations to the isolated
futureagi-worker service. No SDK dependency in the orchestrator.

The worker handles: assess, safety, optimize, live scoring.
This module handles: DB reads/writes, fire-and-forget dispatch, dataset collection.

PRD-262 US-006: every worker call carries X-Internal-Token (AGENT_OPT_WORKER_TOKEN).
No database session is checked out while the worker is awaited: the per-turn hook
reads a cached "any prompt has scoring on" flag, and the database work runs in
threads, each in its own short session (``get_db_session``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from config import config

logger = logging.getLogger(__name__)

WORKER_URL = config.AGENT_OPT_WORKER_URL
WORKER_TIMEOUT = 120  # seconds for assess/safety
OPTIMIZE_TIMEOUT = 300  # seconds for optimization
TOKEN_HEADER = "X-Internal-Token"
# How long a chat turn trusts the cached "does any prompt have live scoring on?"
# answer. Each API process re-reads it at most this often, in a thread; a toggle
# on this process clears it at once (admin_prompts.toggle_futureagi).
SCORING_FLAG_TTL_SECONDS = 60.0


def _worker_headers() -> Dict[str, str]:
    """X-Internal-Token when a token is configured: the worker answers 401 without it."""
    token = config.AGENT_OPT_WORKER_TOKEN
    return {TOKEN_HEADER: token} if token else {}


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


@dataclass(frozen=True)
class _ScoringFlag:
    """The cached answer to "does any prompt have live scoring on?" and when it was read."""

    enabled: bool
    read_at: float

    def fresh(self, now: float) -> bool:
        return now - self.read_at < SCORING_FLAG_TTL_SECONDS


@dataclass(frozen=True)
class _RunInputs:
    """What run_assessment needs from the database, read before the worker is called."""

    run_type: str
    prompt_content: str
    version_id: Any
    settings: Dict[str, Any]


def _any_prompt_scoring_enabled() -> bool:
    """Is live scoring on for any prompt? One short session; runs in a thread."""
    from core.database.database import get_db_session
    from core.models.system_prompts import SystemPrompt

    with get_db_session() as db:
        first = db.query(SystemPrompt).filter(SystemPrompt.futureagi_eval_enabled.is_(True)).first()
        return first is not None


def _store_live_runs(scores: Dict[str, Any]) -> int:
    """Store a live run for every prompt with scoring on, against its active version.

    One short session, opened after the worker answered; runs in a thread.
    """
    from core.database.database import get_db_session
    from core.models.system_prompts import SystemPrompt, SystemPromptEvalRun, SystemPromptVersion

    summary = ", ".join(f"{k}={v.get('score')}" for k, v in scores.items())
    stored = 0
    with get_db_session() as db:
        for prompt in db.query(SystemPrompt).filter(SystemPrompt.futureagi_eval_enabled.is_(True)).all():
            version = db.query(SystemPromptVersion).filter(
                SystemPromptVersion.prompt_id == prompt.id,
                SystemPromptVersion.status == "active",
            ).first()
            if not version:
                continue
            now = datetime.utcnow()
            db.add(SystemPromptEvalRun(
                prompt_id=prompt.id,
                version_id=version.id,
                run_type="live",
                status="completed",
                scores={"scores": scores, "metrics_run": len(scores), "source": "live_traffic"},
                started_at=now,
                completed_at=now,
            ))
            stored += 1
            logger.info("[live] Scored %s: %s", prompt.slug, summary)
    return stored


def _mark_failed(run: Any, message: str) -> None:
    run.status = "failed"
    run.error_message = message
    run.completed_at = datetime.utcnow()


def _start_run(run_id: str) -> Optional[_RunInputs]:
    """Mark the run running and read what the worker needs. The session commits and
    closes before the worker is called; runs in a thread."""
    from core.database.database import get_db_session
    from core.models.system_prompts import SystemPromptEvalRun, SystemPromptVersion

    with get_db_session() as db:
        run = db.query(SystemPromptEvalRun).filter(SystemPromptEvalRun.id == run_id).first()
        if not run:
            logger.error("Assessment run %s not found", run_id)
            return None
        run.status = "running"
        run.started_at = datetime.utcnow()
        version = db.query(SystemPromptVersion).filter(SystemPromptVersion.id == run.version_id).first()
        if not version:
            _mark_failed(run, "Version not found")
            return None
        return _RunInputs(
            run_type=run.run_type,
            prompt_content=version.content,
            version_id=version.id,
            settings=dict(run.metadata_ or {}),
        )


def _save_version_scores(db: Any, version_id: Any, scores: Dict[str, Any]) -> None:
    from core.models.system_prompts import SystemPromptVersion

    version = db.query(SystemPromptVersion).filter(SystemPromptVersion.id == version_id).first()
    if version:
        version.eval_scores = scores


def _finish_run(run_id: str, inputs: _RunInputs, result: Dict[str, Any]) -> None:
    """Write the worker's result in a new session; runs in a thread."""
    from core.database.database import get_db_session
    from core.models.system_prompts import SystemPromptEvalRun

    with get_db_session() as db:
        run = db.query(SystemPromptEvalRun).filter(SystemPromptEvalRun.id == run_id).first()
        if not run:
            logger.error("Assessment run %s vanished before its result was saved", run_id)
            return
        if result.get("status") == "failed" or ("error" in result and not result.get("scores")):
            _mark_failed(run, result.get("error", "Unknown worker error"))
        else:
            run.status = "completed"
            run.scores = result
            run.completed_at = datetime.utcnow()
            if inputs.run_type == "assess" and result.get("scores"):
                _save_version_scores(db, inputs.version_id, result["scores"])
        logger.info("Assessment run %s -> %s", run_id, run.status)


def _fail_run(run_id: str, message: str) -> None:
    """Mark the run failed in its own session; runs in a thread."""
    from core.database.database import get_db_session
    from core.models.system_prompts import SystemPromptEvalRun

    with get_db_session() as db:
        run = db.query(SystemPromptEvalRun).filter(SystemPromptEvalRun.id == run_id).first()
        if run:
            _mark_failed(run, message)


class FutureAGIService:
    """
    Routes all FutureAGI calls to the isolated worker service.
    Keeps DB operations (eval run storage, dataset collection) in the orchestrator.
    """

    _instance: Optional["FutureAGIService"] = None

    def __init__(self) -> None:
        self._available = False
        self._scoring_flag: Optional[_ScoringFlag] = None
        # The scheduled live-scoring tasks, held so none is collected before it ends.
        self._live_tasks: set = set()
        self._init()

    @classmethod
    def get_instance(cls) -> "FutureAGIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init(self) -> None:
        # Check if worker URL is configured (keys live on the worker now)
        self._available = bool(config.AGENT_OPT_WORKER_URL or config.FUTUREAGI_API_KEY)
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
            async with httpx.AsyncClient(timeout=timeout, headers=_worker_headers()) as client:
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

    async def _call_worker_get(self, path: str, timeout: int = 15) -> Dict[str, Any]:
        """GET from the worker service and return JSON response."""
        url = f"{WORKER_URL}{path}"
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=_worker_headers()) as client:
                resp = await client.get(url)
            if resp.status_code != 200:
                error_text = resp.text[:300]
                return {"error": f"Worker error ({resp.status_code}): {error_text}"}
            return resp.json()
        except Exception as e:
            logger.warning(f"[worker] GET {path} failed: {e}")
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

        # Use provided test cases, or pull real I/O from live traffic
        if test_cases:
            payload["test_input"] = test_cases[0].get("input")
            payload["test_output"] = test_cases[0].get("output", test_cases[0].get("expected_output"))
        else:
            # Pull a real chat exchange so scoring is meaningful
            dataset = await self._collect_optimization_dataset(limit=1)
            if dataset:
                payload["test_input"] = dataset[0]["input"]
                payload["test_output"] = dataset[0]["output"]
                logger.info(f"[assess] using real I/O: input={dataset[0]['input'][:80]}...")
            else:
                logger.warning("[assess] no live traffic data, scores will be based on synthetic output")

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
        num_iterations: int = 2,
    ) -> Dict[str, Any]:
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        dataset = await self._collect_optimization_dataset(limit=10)
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

        # Start async optimization job on worker
        start_result = await self._call_worker("/optimize", payload, timeout=30)
        job_id = start_result.get("job_id")
        if not job_id:
            return start_result  # error or unexpected response

        logger.info(f"[optimize] job started: {job_id}, {len(dataset)} examples, {num_iterations} rounds")

        # Poll for completion (up to 25 minutes)
        max_wait = 1500
        poll_interval = 10
        elapsed = 0
        consecutive_errors = 0
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            status_result = await self._call_worker_get(f"/optimize/{job_id}", timeout=15)
            status = status_result.get("status", "unknown")

            if status == "completed":
                logger.info(f"[optimize] job {job_id} completed in {elapsed}s")
                result = status_result
                # Normalize response keys for frontend
                if "optimized_prompt" in result:
                    result["best_score"] = result.pop("final_score", None)
                    result["rounds"] = result.pop("rounds_completed", None)
                    result["duration"] = result.pop("duration_seconds", None)
                return result
            elif status == "failed":
                logger.warning(f"[optimize] job {job_id} failed: {status_result.get('error')}")
                return {"error": status_result.get("error", "Optimization failed"), "status": "failed"}
            elif "error" in status_result:
                consecutive_errors += 1
                logger.warning(f"[optimize] poll error #{consecutive_errors}: {status_result.get('error')}")
                # If worker lost the job (404 / restart), fail fast
                if consecutive_errors >= 5:
                    return {"error": "Lost connection to optimization job (worker may have restarted)", "status": "failed"}
            else:
                consecutive_errors = 0
                if elapsed % 60 == 0:
                    logger.info(f"[optimize] job {job_id} still {status} ({elapsed}s elapsed)")

        return {"error": f"Optimization timed out after {max_wait}s", "status": "failed"}

    # ------------------------------------------------------------------
    # Live traffic scoring (fire-and-forget from chat pipeline)
    # ------------------------------------------------------------------

    LIVE_METRICS = ["completeness", "is_helpful", "is_concise"]

    def schedule_live_eval(self, input_text: str, output_text: str, context_text: Optional[str] = None) -> None:
        """The chat pipeline's per-turn hook: score the exchange in the background.

        PRD-262 US-006: the turn reads only the cached flag, never the database. While
        the flag is fresh and says no prompt has scoring on, nothing is scheduled.
        """
        if not (input_text and output_text) or not self.is_available:
            return
        flag = self._scoring_flag
        if flag is not None and flag.fresh(time.monotonic()) and not flag.enabled:
            return
        task = asyncio.create_task(self.eval_live_traffic(input_text, output_text, context_text))
        self._live_tasks.add(task)
        task.add_done_callback(self._live_tasks.discard)

    def forget_scoring_flag(self) -> None:
        """Drop the cached flag (an admin toggled scoring), so the next turn re-reads it."""
        self._scoring_flag = None

    async def _scoring_enabled(self) -> bool:
        """The cached flag, re-read in a thread once it is older than SCORING_FLAG_TTL_SECONDS."""
        flag = self._scoring_flag
        if flag is None or not flag.fresh(time.monotonic()):
            enabled = await asyncio.to_thread(_any_prompt_scoring_enabled)
            flag = _ScoringFlag(enabled=enabled, read_at=time.monotonic())
            self._scoring_flag = flag
        return flag.enabled

    async def eval_live_traffic(
        self,
        input_text: str,
        output_text: str,
        context_text: Optional[str] = None,
    ) -> None:
        """Score one chat exchange through the worker and store a live run per scored prompt.

        Scheduled by schedule_live_eval. The flag read and the writes run in threads,
        each in its own short session, so no pooled connection waits on the worker
        (up to WORKER_TIMEOUT seconds).
        """
        if not self.is_available:
            return
        try:
            if not await self._scoring_enabled():
                return
            payload: Dict[str, Any] = {
                "input_text": input_text,
                "output_text": output_text,
                "metrics": self.LIVE_METRICS,
            }
            if context_text:
                payload["context_text"] = context_text
            result = await self._call_worker("/score", payload)
            if "error" in result:
                logger.warning("[live] worker scoring failed: %s", result["error"])
                return
            await asyncio.to_thread(_store_live_runs, result.get("scores", {}))
        except Exception:
            logger.exception("[live] eval_live_traffic failed")

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
        """Process one SystemPromptEvalRun by its ID (admin_prompts.py, in a background thread).

        PRD-262 US-006: no session is checked out while the worker is awaited; an
        optimisation polls it for up to 25 minutes. The run is read and marked running
        in one short session that commits and closes, and the result is written in a
        new one. Before, one pooled connection and one threadpool token were held for
        the whole wait.
        """
        try:
            inputs = await asyncio.to_thread(_start_run, run_id)
            if inputs is None:
                return
            result = await self._run_on_worker(inputs)
            await asyncio.to_thread(_finish_run, run_id, inputs, result)
        except Exception as exc:
            logger.exception("Assessment run %s failed", run_id)
            await asyncio.to_thread(_fail_run, run_id, str(exc) or type(exc).__name__)

    async def _run_on_worker(self, inputs: _RunInputs) -> Dict[str, Any]:
        """Send the run to the worker by its type. No database session is open here."""
        settings = inputs.settings
        if inputs.run_type == "optimize":
            return await self.optimize_prompt(
                inputs.prompt_content,
                algorithm=settings.get("algorithm", "meta_prompt"),
                target_metric=settings.get("target_metric", "is_helpful"),
                num_iterations=settings.get("num_iterations", 2),
            )
        if inputs.run_type == "safety":
            return await self.safety_check(inputs.prompt_content)
        if inputs.run_type == "assess":
            return await self.assess_prompt(
                inputs.prompt_content,
                test_cases=settings.get("test_cases"),
                metrics=settings.get("metrics"),
            )
        return await self.assess_prompt(inputs.prompt_content)


# Module-level singleton
futureagi_service = FutureAGIService.get_instance()
