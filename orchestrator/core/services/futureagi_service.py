"""
PRD-58 Phase 1B: FutureAGI Integration Service
================================================

Wrapper around the FutureAGI SDK (pip install futureagi) for:
- Prompt quality scoring (instruction_adherence, task_completion, etc.)
- Prompt optimization (Bayesian, ProTeGi, GEPA, etc.)
- Safety scanning (toxicity, bias, jailbreak detection)

Requires:
  FUTUREAGI_API_KEY   - API key from FutureAGI dashboard
  FUTUREAGI_SECRET_KEY - Secret key from FutureAGI dashboard

pip install futureagi
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FutureAGIService:
    """
    Singleton service wrapping the FutureAGI SDK.

    All methods are async-safe and handle SDK unavailability gracefully.
    """

    _instance: Optional["FutureAGIService"] = None

    def __init__(self) -> None:
        self._client = None
        self._available = False
        self._init_client()

    @classmethod
    def get_instance(cls) -> "FutureAGIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_client(self) -> None:
        """Initialize the FutureAGI SDK client."""
        api_key = os.getenv("FUTUREAGI_API_KEY")
        secret_key = os.getenv("FUTUREAGI_SECRET_KEY")

        if not api_key or not secret_key:
            logger.info("FutureAGI keys not configured, service disabled")
            return

        try:
            from fi.client import FIClient
            self._client = FIClient(api_key=api_key, secret_key=secret_key)
            self._available = True
            logger.info("FutureAGI client initialized successfully")
        except ImportError as ie:
            logger.warning(f"futureagi import failed: {ie}")
        except Exception as e:
            logger.warning(f"FutureAGI client init failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    async def assess_prompt(
        self,
        prompt_content: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Score a system prompt using FutureAGI metrics.

        Args:
            prompt_content: The system prompt text to assess
            test_cases: Optional list of {"input": ..., "expected_output": ...}
            metrics: Which metrics to run (defaults to all)

        Returns:
            Dict with metric scores (0.0-1.0) and details
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured", "scores": {}}

        default_metrics = [
            "instruction_adherence",
            "task_completion",
            "factual_accuracy",
            "conciseness",
            "groundedness",
        ]
        selected_metrics = metrics or default_metrics

        try:
            from fi.evals import (
                InstructionAdherence,
                Groundedness,
                Conciseness,
            )

            # Build metric instances
            metric_map = {
                "instruction_adherence": InstructionAdherence,
                "groundedness": Groundedness,
                "conciseness": Conciseness,
            }

            results = {}
            for metric_name in selected_metrics:
                metric_cls = metric_map.get(metric_name)
                if not metric_cls:
                    # For metrics not directly available, use a general scoring approach
                    results[metric_name] = {"score": None, "note": "Metric not yet mapped"}
                    continue

                try:
                    metric_instance = metric_cls()
                    # Use the client to run scoring
                    score = self._client.run_metric(
                        metric=metric_instance,
                        prompt=prompt_content,
                        test_cases=test_cases or [],
                    )
                    results[metric_name] = {
                        "score": float(score) if score is not None else None,
                    }
                except Exception as metric_err:
                    logger.warning(f"Metric {metric_name} failed: {metric_err}")
                    results[metric_name] = {"score": None, "error": str(metric_err)}

            return {"scores": results, "metrics_run": len(results)}

        except Exception as e:
            logger.error(f"FutureAGI assessment failed: {e}")
            return {"error": str(e), "scores": {}}

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "bayesian",
        target_metric: str = "instruction_adherence",
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Run prompt optimization using FutureAGI's optimization algorithms.

        Algorithms: bayesian, protegi, meta_prompt, gepa, random, prompt_wizard
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        try:
            # Map algorithm names to SDK classes
            from fi.optim import BayesianSearch, ProTeGi, MetaPrompt, GEPA, RandomSearch

            algo_map = {
                "bayesian": BayesianSearch,
                "protegi": ProTeGi,
                "meta_prompt": MetaPrompt,
                "gepa": GEPA,
                "random": RandomSearch,
            }

            algo_cls = algo_map.get(algorithm)
            if not algo_cls:
                return {"error": f"Unknown algorithm: {algorithm}"}

            optimizer = algo_cls(
                target_metric=target_metric,
                num_iterations=num_iterations,
            )

            result = self._client.optimize(
                optimizer=optimizer,
                initial_prompt=prompt_content,
            )

            return {
                "optimized_prompt": result.best_prompt if hasattr(result, "best_prompt") else str(result),
                "best_score": float(result.best_score) if hasattr(result, "best_score") else None,
                "algorithm": algorithm,
                "iterations": num_iterations,
                "history": result.history if hasattr(result, "history") else [],
            }

        except ImportError as ie:
            logger.warning(f"FutureAGI optimization import failed: {ie}")
            return {"error": f"Optimization module not available: {ie}"}
        except Exception as e:
            logger.error(f"FutureAGI optimization failed: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Safety Check
    # ------------------------------------------------------------------

    async def safety_check(self, prompt_content: str) -> Dict[str, Any]:
        """
        Run safety scanning on a prompt (toxicity, bias, jailbreak vectors).
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured", "safe": None}

        try:
            from fi.evals import Toxicity, Bias

            checks = {}

            # Toxicity check
            try:
                toxicity = Toxicity()
                tox_score = self._client.run_metric(
                    metric=toxicity,
                    prompt=prompt_content,
                    test_cases=[],
                )
                checks["toxicity"] = {
                    "score": float(tox_score) if tox_score is not None else None,
                    "safe": float(tox_score) < 0.3 if tox_score is not None else None,
                }
            except Exception as e:
                checks["toxicity"] = {"error": str(e)}

            # Bias check
            try:
                bias = Bias()
                bias_score = self._client.run_metric(
                    metric=bias,
                    prompt=prompt_content,
                    test_cases=[],
                )
                checks["bias"] = {
                    "score": float(bias_score) if bias_score is not None else None,
                    "safe": float(bias_score) < 0.3 if bias_score is not None else None,
                }
            except Exception as e:
                checks["bias"] = {"error": str(e)}

            # Overall safety
            all_safe = all(
                c.get("safe", True)
                for c in checks.values()
                if "safe" in c and c["safe"] is not None
            )

            return {
                "safe": all_safe,
                "checks": checks,
            }

        except ImportError:
            return {"error": "Safety modules not available", "safe": None}
        except Exception as e:
            logger.error(f"FutureAGI safety check failed: {e}")
            return {"error": str(e), "safe": None}

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

            # Mark as running
            run.status = "running"
            run.started_at = datetime.utcnow()
            db.commit()

            # Load the version content
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

            # Dispatch based on run_type
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
                    target_metric=config.get("target_metric", "instruction_adherence"),
                    num_iterations=config.get("num_iterations", 10),
                )
            elif run.run_type == "safety":
                result = await self.safety_check(prompt_content)
            else:
                # Default to assessment
                result = await self.assess_prompt(prompt_content)

            # Save results
            if "error" in result and not result.get("scores"):
                run.status = "failed"
                run.error_message = result["error"]
            else:
                run.status = "completed"
                run.scores = result

                # Also snapshot scores on the version if this is a quality assessment
                if run.run_type in ("assess", "evaluate") and result.get("scores"):
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
