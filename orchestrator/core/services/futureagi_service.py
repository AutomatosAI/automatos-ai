"""
PRD-58 Phase 1B: FutureAGI Integration Service
================================================

Wrapper around the FutureAGI SDK (pip install futureagi) for:
- Prompt quality scoring (groundedness, toxicity, coherence, etc.)
- Safety scanning (toxicity, bias, prompt injection detection)

futureagi 0.6.0 API:
  - fi.client.Client(fi_api_key, fi_secret_key)
  - fi.evals.Evaluator(fi_api_key, fi_secret_key)
  - fi.evals.templates.* (Groundedness, Toxicity, PromptAdherence, etc.)
  - fi.testcases.TestCase(input=..., output=..., prompt=...)

Requires:
  FUTUREAGI_API_KEY   - API key from FutureAGI dashboard
  FUTUREAGI_SECRET_KEY - Secret key from FutureAGI dashboard
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
        self._evaluator = None
        self._available = False
        self._api_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._init_client()

    @classmethod
    def get_instance(cls) -> "FutureAGIService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_client(self) -> None:
        """Initialize the FutureAGI SDK evaluator."""
        self._api_key = os.getenv("FUTUREAGI_API_KEY")
        self._secret_key = os.getenv("FUTUREAGI_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            logger.info("FutureAGI keys not configured, service disabled")
            return

        try:
            from fi.evals import Evaluator
            self._evaluator = Evaluator(
                fi_api_key=self._api_key,
                fi_secret_key=self._secret_key,
            )
            self._available = True
            logger.info("FutureAGI evaluator initialized successfully")
        except ImportError as ie:
            logger.warning(f"futureagi import failed: {ie}")
        except Exception as e:
            logger.warning(f"FutureAGI init failed: {e}")

    @property
    def is_available(self) -> bool:
        return self._available and self._evaluator is not None

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
        Score a system prompt using FutureAGI eval templates.

        Available templates: groundedness, prompt_adherence, completeness,
        context_adherence, factual_accuracy, summary_quality, coherence
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured", "scores": {}}

        default_metrics = [
            "prompt_adherence",
            "completeness",
            "groundedness",
        ]
        selected_metrics = metrics or default_metrics

        try:
            from fi.evals.templates import (
                PromptAdherence,
                Completeness,
                Groundedness,
                FactualAccuracy,
                SummaryQuality,
                ContextAdherence,
                ConversationCoherence,
            )

            template_map = {
                "prompt_adherence": PromptAdherence,
                "completeness": Completeness,
                "groundedness": Groundedness,
                "factual_accuracy": FactualAccuracy,
                "summary_quality": SummaryQuality,
                "context_adherence": ContextAdherence,
                "coherence": ConversationCoherence,
            }

            # FutureAGI API requires: only input/output keys (no prompt/context),
            # and model_name for system evals
            eval_model = "gpt-4"

            results = {}
            for metric_name in selected_metrics:
                template_cls = template_map.get(metric_name)
                if not template_cls:
                    results[metric_name] = {"score": None, "note": "Template not mapped"}
                    continue

                try:
                    template = template_cls()

                    # Build input as dict with only accepted keys (input, output)
                    if test_cases:
                        tc = {
                            "input": test_cases[0].get("input", prompt_content),
                            "output": test_cases[0].get("output", test_cases[0].get("expected_output", "")),
                        }
                    else:
                        tc = {
                            "input": prompt_content,
                            "output": "System prompt evaluated successfully.",
                        }

                    batch_result = self._evaluator.evaluate(
                        eval_templates=[template],
                        inputs=[tc],
                        model_name=eval_model,
                    )

                    # Extract score from BatchRunResult
                    if batch_result and hasattr(batch_result, "eval_results") and batch_result.eval_results:
                        first = batch_result.eval_results[0]
                        score_val = None
                        if first.metrics:
                            score_val = first.metrics[0].value
                        results[metric_name] = {
                            "score": float(score_val) if score_val is not None else None,
                            "passed": not first.failure,
                            "reason": first.reason or None,
                        }
                    else:
                        results[metric_name] = {"score": None, "note": "No result returned"}

                except Exception as metric_err:
                    logger.warning(f"Metric {metric_name} failed: {metric_err}")
                    results[metric_name] = {"score": None, "error": str(metric_err)}

            return {"scores": results, "metrics_run": len(results)}

        except ImportError as ie:
            logger.error(f"FutureAGI eval import failed: {ie}")
            return {"error": str(ie), "scores": {}}
        except Exception as e:
            logger.error(f"FutureAGI assessment failed: {e}")
            return {"error": str(e), "scores": {}}

    # ------------------------------------------------------------------
    # Safety Check
    # ------------------------------------------------------------------

    async def safety_check(self, prompt_content: str) -> Dict[str, Any]:
        """
        Run safety scanning on a prompt (toxicity, bias, prompt injection).
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured", "safe": None}

        try:
            from fi.evals.templates import (
                Toxicity,
                BiasDetection,
                PromptInjection,
                ContentModeration,
            )

            safety_templates = {
                "toxicity": Toxicity,
                "bias": BiasDetection,
                "prompt_injection": PromptInjection,
                "content_moderation": ContentModeration,
            }

            checks = {}
            for check_name, template_cls in safety_templates.items():
                try:
                    template = template_cls()
                    tc = {
                        "input": prompt_content,
                        "output": prompt_content,
                    }
                    batch_result = self._evaluator.evaluate(
                        eval_templates=[template],
                        inputs=[tc],
                        model_name="gpt-4",
                    )

                    if batch_result and batch_result.eval_results:
                        first = batch_result.eval_results[0]
                        score_val = None
                        if first.metrics:
                            score_val = first.metrics[0].value
                        checks[check_name] = {
                            "score": float(score_val) if score_val is not None else None,
                            "safe": not first.failure,
                            "reason": first.reason or None,
                        }
                    else:
                        checks[check_name] = {"score": None, "safe": None}

                except Exception as e:
                    logger.warning(f"Safety check {check_name} failed: {e}")
                    checks[check_name] = {"error": str(e)}

            # Overall safety: all checks must pass
            all_safe = all(
                c.get("safe", True)
                for c in checks.values()
                if "safe" in c and c["safe"] is not None
            )

            return {
                "safe": all_safe,
                "checks": checks,
            }

        except ImportError as ie:
            logger.warning(f"Safety module import failed: {ie}")
            return {"error": str(ie), "safe": None}
        except Exception as e:
            logger.error(f"FutureAGI safety check failed: {e}")
            return {"error": str(e), "safe": None}

    # ------------------------------------------------------------------
    # Optimization (placeholder — fi.optim may not exist in 0.6.0)
    # ------------------------------------------------------------------

    async def optimize_prompt(
        self,
        prompt_content: str,
        algorithm: str = "bayesian",
        target_metric: str = "prompt_adherence",
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Prompt optimization. Returns error if fi.optim is not available.
        """
        if not self.is_available:
            return {"error": "FutureAGI not configured"}

        try:
            from fi.prompt import PromptOptimizer
            optimizer = PromptOptimizer(
                fi_api_key=self._api_key,
                fi_secret_key=self._secret_key,
            )
            result = optimizer.optimize(
                prompt=prompt_content,
                algorithm=algorithm,
                target_metric=target_metric,
                num_iterations=num_iterations,
            )
            return {
                "optimized_prompt": getattr(result, "best_prompt", str(result)),
                "best_score": float(getattr(result, "best_score", 0)),
                "algorithm": algorithm,
                "iterations": num_iterations,
            }
        except ImportError:
            return {"error": "Prompt optimization not available in this SDK version"}
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
                    target_metric=config.get("target_metric", "prompt_adherence"),
                    num_iterations=config.get("num_iterations", 10),
                )
            elif run.run_type == "safety":
                result = await self.safety_check(prompt_content)
            else:
                result = await self.assess_prompt(prompt_content)

            # Save results
            if "error" in result and not result.get("scores"):
                run.status = "failed"
                run.error_message = result["error"]
            else:
                run.status = "completed"
                run.scores = result

                # Snapshot scores on the version for quality assessments
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
