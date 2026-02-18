"""
PRD-58 Phase 1B: FutureAGI Integration Service
================================================

Direct HTTP integration with FutureAGI API (no SDK dependency).

API: POST https://api.futureagi.com/sdk/api/v1/new-eval/
Auth: X-Api-Key + X-Secret-Key headers
Payload: {"eval_name": "<template>", "inputs": {"input": [...], "output": [...]}, "model": "gpt-4"}

Requires:
  FUTUREAGI_API_KEY   - API key from FutureAGI dashboard
  FUTUREAGI_SECRET_KEY - Secret key from FutureAGI dashboard
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FUTUREAGI_BASE_URL = "https://api.futureagi.com"
ASSESSMENT_ENDPOINT = f"{FUTUREAGI_BASE_URL}/sdk/api/v1/new-eval/"
TIMEOUT = 60  # seconds per request


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

    async def _run_single_assessment(
        self, assessment_name: str, input_text: str, output_text: str, model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Single assessment call to FutureAGI API.
        Returns parsed result dict with score, failure, reason, metrics.
        """
        payload = {
            "eval_name": assessment_name,
            "inputs": {
                "input": [input_text],
                "output": [output_text],
            },
            "model": model,
        }

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(ASSESSMENT_ENDPOINT, json=payload, headers=self._headers())

        logger.info(f"[{assessment_name}] status={resp.status_code}")

        if resp.status_code != 200:
            logger.warning(f"[{assessment_name}] error: {resp.status_code} {resp.text[:500]}")
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        logger.info(f"[{assessment_name}] raw response keys={list(data.keys())}")

        # Parse: {"result": [{"evaluations": [{"failure": bool, "reason": str, "metrics": [{"id": str, "value": float}]}]}]}
        try:
            results = data.get("result", [])
            if not results:
                return {"error": "Empty result from API", "raw": data}

            assessments = results[0].get("evaluations", [])
            if not assessments:
                return {"error": "No evaluations in result", "raw": data}

            item = assessments[0]
            metrics = item.get("metrics", [])
            score_val = metrics[0]["value"] if metrics else None
            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            return {
                "score": float(score_val) if score_val is not None else None,
                "passed": not item.get("failure", True),
                "reason": item.get("reason", ""),
                "explanation": metadata.get("explanation", {}) if isinstance(metadata, dict) else {},
            }
        except Exception as e:
            logger.warning(f"[{assessment_name}] parse error: {e}, raw={data}")
            return {"error": f"Parse error: {e}", "raw": data}

    # ------------------------------------------------------------------
    # Assessment
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

        default_metrics = ["prompt_adherence", "completeness", "groundedness"]
        selected_metrics = metrics or default_metrics

        if test_cases:
            input_text = test_cases[0].get("input", prompt_content)
            output_text = test_cases[0].get("output", test_cases[0].get("expected_output", ""))
        else:
            input_text = prompt_content
            output_text = "System prompt assessed successfully."

        results = {}
        for metric_name in selected_metrics:
            result = await self._run_single_assessment(metric_name, input_text, output_text)
            if "error" in result:
                results[metric_name] = {"score": None, "error": result["error"]}
            else:
                results[metric_name] = {
                    "score": result["score"],
                    "passed": result["passed"],
                    "reason": result["reason"],
                }

        return {"scores": results, "metrics_run": len(results)}

    # ------------------------------------------------------------------
    # Safety Check
    # ------------------------------------------------------------------

    async def safety_check(self, prompt_content: str) -> Dict[str, Any]:
        """Run safety scanning on a prompt."""
        if not self.is_available:
            return {"error": "FutureAGI not configured", "safe": None}

        safety_checks = ["toxicity", "bias_detection", "prompt_injection", "content_moderation"]

        checks = {}
        for check_name in safety_checks:
            result = await self._run_single_assessment(check_name, prompt_content, prompt_content)
            if "error" in result:
                checks[check_name] = {"score": None, "safe": None, "error": result["error"]}
            else:
                checks[check_name] = {
                    "score": result["score"],
                    "safe": result["passed"],
                    "reason": result["reason"],
                }

        all_safe = all(
            c.get("safe", True)
            for c in checks.values()
            if "safe" in c and c["safe"] is not None
        )

        return {"safe": all_safe, "checks": checks}

    # ------------------------------------------------------------------
    # Optimization (via FutureAGI improve-prompt endpoint)
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "bayesian",
        target_metric: str = "prompt_adherence",
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Prompt optimization via FutureAGI improve-prompt API."""
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        url = f"{FUTUREAGI_BASE_URL}/model-hub/prompt-templates/improve-prompt/"
        payload = {
            "prompt": prompt_content,
            "algorithm": algorithm,
            "target_metric": target_metric,
            "num_iterations": num_iterations,
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=self._headers())

            logger.info(f"[optimize] status={resp.status_code}")

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            result = data.get("result", data)

            return {
                "optimized_prompt": result.get("improved_prompt", result.get("prompt", str(result))),
                "best_score": result.get("score"),
                "algorithm": algorithm,
                "iterations": num_iterations,
            }
        except Exception as e:
            logger.error(f"FutureAGI optimization failed: {e}")
            return {"error": str(e)}

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

            if "error" in result and not result.get("scores"):
                run.status = "failed"
                run.error_message = result["error"]
            else:
                run.status = "completed"
                run.scores = result

                if run.run_type in ("assess",) and result.get("scores"):
                    version.eval_scores = result["scores"]

            run.completed_at = datetime.utcnow()
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
