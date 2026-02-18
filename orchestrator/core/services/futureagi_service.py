"""
PRD-58 Phase 1B: FutureAGI Integration Service
================================================

Direct HTTP integration with FutureAGI API (no SDK dependency).

API: POST https://api.futureagi.com/sdk/api/v1/new-eval/
Auth: X-Api-Key + X-Secret-Key headers

Each template has different required input keys and valid models.
See TEMPLATE_CONFIG below for the full mapping.

Requires:
  FUTUREAGI_API_KEY   - API key from FutureAGI dashboard
  FUTUREAGI_SECRET_KEY - Secret key from FutureAGI dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FUTUREAGI_BASE_URL = "https://api.futureagi.com"
ASSESSMENT_ENDPOINT = f"{FUTUREAGI_BASE_URL}/sdk/api/v1/new-eval/"
TIMEOUT = 90  # seconds per request (some templates are slow)

# Template config: required input keys + best model per template
# Sourced from GET /sdk/api/v1/get-evals/ on 2026-02-18
TEMPLATE_CONFIG = {
    # --- Quality assessment ---
    "completeness":      {"keys": ["input", "output"], "model": "turing_large"},
    "prompt_adherence":  {"keys": ["input", "output"], "model": "turing_large"},
    "groundedness":      {"keys": ["input", "output", "context"], "model": "turing_large"},
    "factual_accuracy":  {"keys": ["input", "output"], "model": "turing_large"},
    "summary_quality":   {"keys": ["input", "output"], "model": "turing_large"},
    "is_concise":        {"keys": ["output"], "model": "turing_large"},
    "is_helpful":        {"keys": ["input", "output"], "model": "turing_large"},
    # --- Safety ---
    "toxicity":          {"keys": ["output"], "model": "protect"},
    "bias_detection":    {"keys": ["output"], "model": "protect_flash"},
    "prompt_injection":  {"keys": ["input"], "model": "protect"},
    "content_moderation": {"keys": ["output"], "model": "protect"},
}


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
    Singleton service calling FutureAGI REST API directly via httpx.
    No SDK dependency — just HTTP POST with JSON payloads.
    """

    _instance: Optional["FutureAGIService"] = None

    def __init__(self) -> None:
        self._api_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._available = False
        self._init()

    @classmethod
    def get_instance(cls) -> "FutureAGIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init(self) -> None:
        self._api_key = os.getenv("FUTUREAGI_API_KEY")
        self._secret_key = os.getenv("FUTUREAGI_SECRET_KEY")
        if self._api_key and self._secret_key:
            self._available = True
            logger.info("FutureAGI service ready (direct HTTP)")
        else:
            logger.info("FutureAGI keys not configured, service disabled")

    @property
    def is_available(self) -> bool:
        return self._available

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Api-Key": self._api_key or "",
            "X-Secret-Key": self._secret_key or "",
            "Content-Type": "application/json",
        }

    def _build_inputs(
        self, template_name: str, input_text: str, output_text: str, context_text: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Build the inputs dict with only the keys this template accepts."""
        config = TEMPLATE_CONFIG.get(template_name, {"keys": ["input", "output"]})
        required = config["keys"]
        inputs: Dict[str, List[str]] = {}
        if "input" in required:
            inputs["input"] = [input_text]
        if "output" in required:
            inputs["output"] = [output_text]
        if "context" in required:
            inputs["context"] = [context_text or input_text]
        return inputs

    def _get_model(self, template_name: str) -> str:
        config = TEMPLATE_CONFIG.get(template_name, {"model": "turing_large"})
        return config["model"]

    async def _call_template(
        self, template_name: str, input_text: str, output_text: str, context_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Single assessment call to FutureAGI API. Returns parsed result."""
        payload = {
            "eval_name": template_name,
            "inputs": self._build_inputs(template_name, input_text, output_text, context_text),
            "model": self._get_model(template_name),
        }

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(ASSESSMENT_ENDPOINT, json=payload, headers=self._headers())
        except httpx.TimeoutException:
            logger.warning(f"[{template_name}] timed out after {TIMEOUT}s")
            return {"error": f"Timed out after {TIMEOUT}s"}

        logger.info(f"[{template_name}] status={resp.status_code}")

        if resp.status_code != 200:
            error_text = resp.text[:300]
            logger.warning(f"[{template_name}] error: {resp.status_code} {error_text}")
            return {"error": f"HTTP {resp.status_code}: {error_text}"}

        data = resp.json()

        # Parse response: {"result": [{"evaluations": [{"failure": bool, "reason": str, "metrics": [{"id": str, "value": float}]}]}]}
        try:
            results = data.get("result", [])
            if not results:
                logger.warning(f"[{template_name}] empty result: {data}")
                return {"error": "Empty result from API"}

            evals_list = results[0].get("evaluations", [])
            if not evals_list:
                logger.warning(f"[{template_name}] no evaluations: {results[0]}")
                return {"error": "No evaluations in result"}

            item = evals_list[0]
            metrics = item.get("metrics", [])
            score_val = metrics[0]["value"] if metrics else None

            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # Determine passed/failed: use explicit 'failure' field, fallback to score
            if "failure" in item:
                passed = not item["failure"]
            elif score_val is not None:
                passed = float(score_val) >= 0.5
            else:
                passed = True  # No signal → assume pass

            logger.debug(f"[{template_name}] raw eval: failure={item.get('failure')}, score={score_val}, passed={passed}")

            return {
                "score": float(score_val) if score_val is not None else None,
                "passed": passed,
                "reason": item.get("reason", ""),
                "explanation": metadata.get("explanation", {}) if isinstance(metadata, dict) else {},
            }
        except Exception as e:
            logger.warning(f"[{template_name}] parse error: {e}")
            return {"error": f"Parse error: {e}"}

    # ------------------------------------------------------------------
    # Live traffic evaluation (fire-and-forget from chat pipeline)
    # ------------------------------------------------------------------

    LIVE_METRICS = ["completeness", "is_helpful", "is_concise"]

    async def eval_live_traffic(
        self,
        input_text: str,
        output_text: str,
        context_text: Optional[str] = None,
    ) -> None:
        """
        Called fire-and-forget after each chat response.
        Finds ALL prompts with futureagi_eval_enabled=True, scores each.
        """
        if not self.is_available:
            return

        from core.database.database import SessionLocal
        from core.models.system_prompts import SystemPrompt, SystemPromptEvalRun, SystemPromptVersion

        db = SessionLocal()
        try:
            # Find all prompts with live eval enabled
            enabled_prompts = db.query(SystemPrompt).filter(
                SystemPrompt.futureagi_eval_enabled == True  # noqa: E712
            ).all()
            if not enabled_prompts:
                return

            for prompt in enabled_prompts:
                version = db.query(SystemPromptVersion).filter(
                    SystemPromptVersion.prompt_id == prompt.id,
                    SystemPromptVersion.status == "active",
                ).first()
                if not version:
                    continue

                # Run quality metrics concurrently
                tasks = {
                    name: self._call_template(name, input_text, output_text, context_text=context_text)
                    for name in self.LIVE_METRICS
                }
                raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

                scores = {}
                for name, raw in zip(tasks.keys(), raw_results):
                    if isinstance(raw, Exception):
                        scores[name] = {"score": None, "error": str(raw)}
                    elif "error" in raw:
                        scores[name] = {"score": None, "error": raw["error"]}
                    else:
                        scores[name] = {
                            "score": raw["score"],
                            "passed": raw["passed"],
                            "reason": raw.get("reason", ""),
                        }

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
    # Assessment (runs templates concurrently)
    # ------------------------------------------------------------------

    async def assess_prompt(
        self,
        prompt_content: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score a system prompt using FutureAGI assessment templates."""
        if not self.is_available:
            return {"error": "FutureAGI not configured", "scores": {}}

        default_metrics = ["completeness", "is_helpful", "is_concise"]
        selected_metrics = metrics or default_metrics

        if test_cases:
            input_text = test_cases[0].get("input", prompt_content)
            output_text = test_cases[0].get("output", test_cases[0].get("expected_output", ""))
        else:
            input_text = prompt_content
            output_text = "System prompt assessed successfully."

        # Run all metrics concurrently
        tasks = {
            name: self._call_template(name, input_text, output_text, context_text=prompt_content)
            for name in selected_metrics
        }
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        results = {}
        for name, raw in zip(tasks.keys(), raw_results):
            if isinstance(raw, Exception):
                results[name] = {"score": None, "error": str(raw)}
            elif "error" in raw:
                results[name] = {"score": None, "error": raw["error"]}
            else:
                results[name] = {
                    "score": raw["score"],
                    "passed": raw["passed"],
                    "reason": raw["reason"],
                }

        return {"scores": results, "metrics_run": len(results)}

    # ------------------------------------------------------------------
    # Safety Check (runs templates concurrently)
    # ------------------------------------------------------------------

    async def safety_check(self, prompt_content: str) -> Dict[str, Any]:
        """Run safety scanning on a prompt."""
        if not self.is_available:
            return {"error": "FutureAGI not configured", "safe": None}

        safety_templates = ["toxicity", "bias_detection", "prompt_injection", "content_moderation"]

        # Run all safety checks concurrently
        tasks = {
            name: self._call_template(name, prompt_content, prompt_content)
            for name in safety_templates
        }
        raw_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        checks = {}
        for name, raw in zip(tasks.keys(), raw_results):
            if isinstance(raw, Exception):
                checks[name] = {"score": None, "safe": None, "error": str(raw)}
            elif "error" in raw:
                checks[name] = {"score": None, "safe": None, "error": raw["error"]}
            else:
                checks[name] = {
                    "score": raw["score"],
                    "safe": raw["passed"],
                    "reason": raw["reason"],
                }

        all_safe = all(
            c.get("safe", True)
            for c in checks.values()
            if "safe" in c and c["safe"] is not None
        )

        return {"safe": all_safe, "checks": checks}

    # ------------------------------------------------------------------
    # Optimization (via agent-opt worker service)
    # ------------------------------------------------------------------

    AGENT_OPT_URL = os.getenv("AGENT_OPT_WORKER_URL", "http://agent-opt-worker.railway.internal:8080")

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "meta_prompt",
        target_metric: str = "is_helpful",
        num_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Prompt optimization via agent-opt worker service.

        Collects live traffic input/output pairs as the dataset,
        sends to the isolated agent-opt worker for synchronous optimization.
        Returns the improved prompt directly.
        """
        # Gather dataset from recent live eval traffic
        dataset = await self._collect_optimization_dataset(prompt_content)
        if not dataset:
            return {"error": "No live traffic data yet. Enable FutureAGI scoring and chat first to build a dataset."}

        payload = {
            "prompt_content": prompt_content,
            "dataset": dataset,
            "eval_template": target_metric,
            "algorithm": algorithm,
            "num_rounds": num_iterations,
            "teacher_model": "gpt-4o-mini",
            "task_description": f"Optimize this AI system prompt for better {target_metric.replace('_', ' ')}.",
        }

        try:
            url = f"{self.AGENT_OPT_URL}/optimize"
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(url, json=payload)

            logger.info(f"[optimize] worker status={resp.status_code}")

            if resp.status_code != 200:
                error_text = resp.text[:300]
                logger.warning(f"[optimize] worker error: {error_text}")
                return {"error": f"Worker error: {error_text}"}

            data = resp.json()
            return {
                "optimized_prompt": data.get("optimized_prompt"),
                "best_score": data.get("final_score"),
                "initial_score": data.get("initial_score"),
                "algorithm": data.get("algorithm"),
                "rounds": data.get("rounds_completed"),
                "duration": data.get("duration_seconds"),
                "history": data.get("history", []),
                "status": "completed",
            }
        except httpx.ConnectError:
            logger.warning("[optimize] agent-opt worker not reachable")
            return {"error": "Optimization worker not available. Is the agent-opt-worker service running?"}
        except httpx.TimeoutException:
            logger.warning("[optimize] worker timed out after 300s")
            return {"error": "Optimization timed out (>5min). Try fewer rounds or smaller dataset."}
        except Exception as e:
            logger.error(f"[optimize] failed: {e}")
            return {"error": str(e)}

    async def _collect_optimization_dataset(
        self, prompt_content: str, limit: int = 20
    ) -> List[Dict[str, str]]:
        """Collect recent chat input/output pairs from the database for optimization."""
        from core.database.database import SessionLocal

        db = SessionLocal()
        try:
            # Pull recent chat messages (input/output pairs)
            from sqlalchemy import text
            rows = db.execute(text("""
                SELECT m1.parts, m2.parts
                FROM chat_messages m1
                JOIN chat_messages m2
                  ON m1.chat_id = m2.chat_id
                  AND m2.role = 'assistant'
                  AND m2.created_at = (
                      SELECT MIN(created_at) FROM chat_messages
                      WHERE chat_id = m1.chat_id AND role = 'assistant'
                      AND created_at > m1.created_at
                  )
                WHERE m1.role = 'user'
                ORDER BY m1.created_at DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()

            dataset = []
            for user_parts, assistant_parts in rows:
                # Extract text from parts JSON
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
        Loads the run from DB, performs the requested operation, saves results.
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
                    algorithm=config.get("algorithm", "bayesian"),
                    target_metric=config.get("target_metric", "prompt_adherence"),
                    num_iterations=config.get("num_iterations", 10),
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
            logger.info(f"Assessment run {run_id} saved to DB")

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
