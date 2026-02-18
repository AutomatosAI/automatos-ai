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

            return {
                "score": float(score_val) if score_val is not None else None,
                "passed": not item.get("failure", True),
                "reason": item.get("reason", ""),
                "explanation": metadata.get("explanation", {}) if isinstance(metadata, dict) else {},
            }
        except Exception as e:
            logger.warning(f"[{template_name}] parse error: {e}")
            return {"error": f"Parse error: {e}"}

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
    # Optimization (via FutureAGI improve-prompt endpoint)
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "bayesian",
        target_metric: str = "prompt_adherence",
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Prompt optimization via FutureAGI improve-prompt API.

        The API is async — it returns an improveId immediately.
        We submit the job and return the job ID. The improved prompt
        can be retrieved later via the FutureAGI dashboard.
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        url = f"{FUTUREAGI_BASE_URL}/model-hub/prompt-templates/improve-prompt/"
        payload = {
            "existing_prompt": prompt_content,
            "improvement_requirements": (
                f"Improve this system prompt for better {target_metric.replace('_', ' ')}. "
                f"Make it clearer, more specific, and more effective."
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=payload, headers=self._headers())

            logger.info(f"[optimize] status={resp.status_code} body={resp.text[:300]}")

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            if not data.get("status"):
                return {"error": data.get("result", "Unknown error")}

            result = data.get("result", {})
            improve_id = result.get("improveId") if isinstance(result, dict) else None

            if improve_id:
                return {
                    "optimized_prompt": None,
                    "improve_id": improve_id,
                    "status": "submitted",
                    "message": "Optimization job submitted to FutureAGI. Results available in the FutureAGI dashboard.",
                    "algorithm": algorithm,
                }

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
