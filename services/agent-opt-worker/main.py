"""
PRD-58: FutureAGI Worker Service
=================================

Single gateway for ALL FutureAGI operations. Runs in its own container
with the full SDK (agent-opt + ai-evaluation) isolated from the main
orchestrator.

Endpoints:
  POST /assess    - Score a prompt using quality metric templates
  POST /safety    - Run safety checks on a prompt
  POST /optimize  - Run prompt optimization (synchronous)
  POST /score     - Score a live chat input/output pair
  GET  /health    - Health check
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("futureagi-worker")

app = FastAPI(title="FutureAGI Worker", version="2.0.0")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_keys():
    api_key = os.getenv("FUTUREAGI_API_KEY") or os.getenv("FI_API_KEY")
    secret_key = os.getenv("FUTUREAGI_SECRET_KEY") or os.getenv("FI_SECRET_KEY")
    if not api_key or not secret_key:
        raise HTTPException(status_code=500, detail="FutureAGI keys not configured")
    # Ensure FI_* env vars are set — the agent-opt Evaluator only forwards
    # fi_api_key to the inner fi.evals.Evaluator, dropping fi_secret_key.
    # Setting env vars lets the SDK auto-detect both keys.
    os.environ.setdefault("FI_API_KEY", api_key)
    os.environ.setdefault("FI_SECRET_KEY", secret_key)
    return api_key, secret_key


def _run_single_template(template: str, inputs: Dict[str, str], model: str = "turing_flash") -> Dict[str, Any]:
    """Run a single scoring template via the SDK and return parsed result."""
    api_key, secret_key = _get_keys()

    from fi.evals import Evaluator

    evaluator = Evaluator(fi_api_key=api_key, fi_secret_key=secret_key)

    logger.info(f"[{template}] calling SDK with model={model}, keys={len(inputs)} input keys")
    try:
        result = evaluator.evaluate(
            eval_templates=template,
            inputs=inputs,
            model_name=model,
            timeout=60,
        )
    except Exception as e:
        logger.warning(f"[{template}] scoring failed: {e}")
        return {"error": str(e)}

    # Parse SDK result
    try:
        if not result or not hasattr(result, "eval_results") or not result.eval_results:
            logger.warning(f"[{template}] empty eval_results")
            return {"error": f"Empty result from SDK for {template}"}
        er = result.eval_results[0]
        er_dict = {k: v for k, v in vars(er).items() if not k.startswith("_")} if hasattr(er, "__dict__") else str(er)
        logger.info(f"[{template}] raw result: {er_dict}")

        # SDK returns: output="Passed"/"Failed", output_type="Pass/Fail",
        # reason=explanation, runtime=ms, eval_id=uuid
        output = getattr(er, "output", "")
        reason = getattr(er, "reason", "") or ""
        score = getattr(er, "score", None)
        runtime_ms = getattr(er, "runtime", None)

        # output can be: "Passed"/"Failed" (Pass/Fail type) or a float (score type)
        if isinstance(output, (int, float)):
            # Numeric score (e.g. completeness returns 0.0-1.0)
            score = float(output)
            passed = score >= 0.5
        elif isinstance(output, str):
            output_lower = output.lower().strip()
            if output_lower == "passed":
                passed = True
                score = 1.0
            elif output_lower == "failed":
                passed = False
                score = 0.0
            else:
                # Try to parse as number
                try:
                    score = float(output)
                    passed = score >= 0.5
                except (ValueError, TypeError):
                    passed = True
                    score = 1.0
        else:
            passed = True
            score = 1.0

        logger.info(f"[{template}] output={output} score={score} passed={passed}")
        return {"score": score, "passed": passed, "reason": reason}
    except (IndexError, AttributeError) as e:
        logger.warning(f"[{template}] parse error: {e}")
        return {"error": f"Parse error: {e}"}


# Template config: required input keys + best model per template
TEMPLATE_CONFIG = {
    "completeness":       {"keys": ["input", "output"], "model": "turing_large"},
    "prompt_adherence":   {"keys": ["input", "output"], "model": "turing_large"},
    "groundedness":       {"keys": ["input", "output", "context"], "model": "turing_large"},
    "factual_accuracy":   {"keys": ["input", "output"], "model": "turing_large"},
    "summary_quality":    {"keys": ["input", "output"], "model": "turing_large"},
    "is_concise":         {"keys": ["output"], "model": "turing_large"},
    "is_helpful":         {"keys": ["input", "output"], "model": "turing_large"},
    "toxicity":           {"keys": ["output"], "model": "protect"},
    "bias_detection":     {"keys": ["output"], "model": "protect_flash"},
    "prompt_injection":   {"keys": ["input"], "model": "protect"},
    "content_moderation": {"keys": ["output"], "model": "protect"},
}


def _build_inputs(template: str, input_text: str, output_text: str, context_text: Optional[str] = None) -> Dict[str, str]:
    """Build inputs dict with only the keys this template accepts."""
    config = TEMPLATE_CONFIG.get(template, {"keys": ["input", "output"]})
    required = config["keys"]
    inputs = {}
    if "input" in required:
        inputs["input"] = input_text
    if "output" in required:
        inputs["output"] = output_text
    if "context" in required:
        inputs["context"] = context_text or input_text
    return inputs


def _get_model(template: str) -> str:
    return TEMPLATE_CONFIG.get(template, {}).get("model", "turing_flash")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AssessRequest(BaseModel):
    prompt_content: str
    test_input: Optional[str] = None
    test_output: Optional[str] = None
    metrics: Optional[List[str]] = None


class SafetyRequest(BaseModel):
    prompt_content: str


class ScoreRequest(BaseModel):
    input_text: str
    output_text: str
    context_text: Optional[str] = None
    metrics: Optional[List[str]] = None


class OptimizeRequest(BaseModel):
    prompt_content: str
    dataset: List[Dict[str, str]] = Field(..., min_length=1)
    scoring_template: str = "is_helpful"
    algorithm: str = "meta_prompt"
    num_rounds: int = Field(default=3, ge=1, le=20)
    teacher_model: str = "gpt-4o-mini"
    task_description: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "futureagi-worker", "version": "2.1.0"}


@app.get("/test")
def test_sdk():
    """Quick SDK smoke test — runs is_concise + is_helpful evals."""
    results = {}
    for template, inputs, model in [
        ("is_concise", {"output": "Hello, world!"}, "turing_large"),
        ("is_helpful", {"input": "What is 2+2?", "output": "4"}, "turing_large"),
        ("toxicity", {"output": "Hello, world!"}, "protect"),
    ]:
        try:
            results[template] = _run_single_template(template, inputs, model)
        except Exception as e:
            results[template] = {"error": str(e)}
    return {"tests": results}


# ---------------------------------------------------------------------------
# Assess - score a prompt using quality metrics
# ---------------------------------------------------------------------------

@app.post("/assess")
def assess(req: AssessRequest):
    start = time.time()
    metrics = req.metrics or ["completeness", "is_helpful", "is_concise"]
    input_text = req.test_input or req.prompt_content
    output_text = req.test_output or "System prompt assessed successfully."

    # Run all templates concurrently
    results = {}
    with ThreadPoolExecutor(max_workers=len(metrics)) as pool:
        futures = {}
        for template in metrics:
            inputs = _build_inputs(template, input_text, output_text, context_text=req.prompt_content)
            model = _get_model(template)
            futures[pool.submit(_run_single_template, template, inputs, model)] = template
        for future in as_completed(futures):
            template = futures[future]
            try:
                results[template] = future.result()
            except Exception as e:
                results[template] = {"error": str(e)}

    return {"scores": results, "metrics_run": len(results), "duration": round(time.time() - start, 1)}


# ---------------------------------------------------------------------------
# Safety - run safety templates
# ---------------------------------------------------------------------------

@app.post("/safety")
def safety(req: SafetyRequest):
    start = time.time()
    safety_templates = ["toxicity", "bias_detection", "prompt_injection", "content_moderation"]

    # Run all safety templates concurrently
    raw_results = {}
    with ThreadPoolExecutor(max_workers=len(safety_templates)) as pool:
        futures = {}
        for template in safety_templates:
            inputs = _build_inputs(template, req.prompt_content, req.prompt_content)
            model = _get_model(template)
            futures[pool.submit(_run_single_template, template, inputs, model)] = template
        for future in as_completed(futures):
            template = futures[future]
            try:
                raw_results[template] = future.result()
            except Exception as e:
                raw_results[template] = {"error": str(e)}

    checks = {}
    for template, result in raw_results.items():
        if "error" in result:
            checks[template] = {"score": None, "safe": None, "error": result["error"]}
        else:
            checks[template] = {
                "score": result.get("score"),
                "safe": result.get("passed"),
                "reason": result.get("reason", ""),
            }

    all_safe = all(
        c.get("safe", True)
        for c in checks.values()
        if "safe" in c and c["safe"] is not None
    )

    return {"safe": all_safe, "checks": checks, "duration": round(time.time() - start, 1)}


# ---------------------------------------------------------------------------
# Score - score a real chat exchange (live traffic)
# ---------------------------------------------------------------------------

@app.post("/score")
def score_live(req: ScoreRequest):
    start = time.time()
    metrics = req.metrics or ["completeness", "is_helpful", "is_concise"]

    # Run all scoring templates concurrently
    scores = {}
    with ThreadPoolExecutor(max_workers=len(metrics)) as pool:
        futures = {}
        for template in metrics:
            inputs = _build_inputs(template, req.input_text, req.output_text, context_text=req.context_text)
            model = _get_model(template)
            futures[pool.submit(_run_single_template, template, inputs, model)] = template
        for future in as_completed(futures):
            template = futures[future]
            try:
                result = future.result()
                if "error" in result:
                    scores[template] = {"score": None, "error": result["error"]}
                else:
                    scores[template] = {
                        "score": result.get("score"),
                        "passed": result.get("passed"),
                        "reason": result.get("reason", ""),
                    }
            except Exception as e:
                scores[template] = {"score": None, "error": str(e)}

    return {"scores": scores, "metrics_run": len(scores), "source": "live_traffic", "duration": round(time.time() - start, 1)}


# ---------------------------------------------------------------------------
# Optimize - full prompt optimization via agent-opt
# ---------------------------------------------------------------------------

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    start = time.time()
    api_key, secret_key = _get_keys()

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        from fi.opt.base.evaluator import Evaluator
        from fi.opt.datamappers import BasicDataMapper
        from fi.opt.generators import LiteLLMGenerator
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"agent-opt not installed: {e}")

    evaluator = Evaluator(
        eval_template=req.scoring_template,
        eval_model_name="turing_flash",
        fi_api_key=api_key,
        fi_secret_key=secret_key,
    )

    data_mapper = BasicDataMapper(key_map={"input": "input", "output": "output"})
    teacher = LiteLLMGenerator(model=req.teacher_model, prompt_template="{prompt}")

    task_desc = req.task_description or (
        "This is a system prompt for an AI assistant. "
        "Optimize it to produce more helpful, concise, and complete responses."
    )

    try:
        optimizer = _create_optimizer(req.algorithm, teacher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Starting {req.algorithm}: {req.num_rounds} rounds, "
        f"{len(req.dataset)} examples, template={req.scoring_template}"
    )

    try:
        result = optimizer.optimize(
            evaluator=evaluator,
            data_mapper=data_mapper,
            dataset=req.dataset,
            initial_prompts=[req.prompt_content],
            task_description=task_desc,
            num_rounds=req.num_rounds,
            eval_subset_size=min(len(req.dataset), 10),
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    duration = time.time() - start
    best_prompt = result.best_generator.get_prompt_template()

    history = []
    if hasattr(result, "history") and result.history:
        for i, iteration in enumerate(result.history):
            history.append({
                "round": i + 1,
                "score": getattr(iteration, "average_score", None),
                "prompt_preview": getattr(iteration, "prompt", "")[:200],
            })

    logger.info(f"Optimization complete: {result.final_score:.4f} in {duration:.1f}s")

    return {
        "optimized_prompt": best_prompt,
        "final_score": result.final_score,
        "initial_score": history[0]["score"] if history else None,
        "rounds_completed": req.num_rounds,
        "algorithm": req.algorithm,
        "history": history,
        "duration_seconds": round(duration, 1),
        "status": "completed",
    }


def _create_optimizer(algorithm: str, teacher):
    if algorithm == "meta_prompt":
        from fi.opt.optimizers import MetaPromptOptimizer
        return MetaPromptOptimizer(teacher_generator=teacher)
    elif algorithm == "bayesian":
        from fi.opt.optimizers import BayesianSearchOptimizer
        return BayesianSearchOptimizer(
            inference_model_name=teacher.model,
            min_examples=2, max_examples=5,
        )
    elif algorithm == "protegi":
        from fi.opt.optimizers import ProTeGi
        return ProTeGi(
            teacher_generator=teacher, num_gradients=4, beam_size=4,
        )
    elif algorithm == "random":
        from fi.opt.optimizers import RandomSearchOptimizer
        return RandomSearchOptimizer(generator=teacher)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Options: meta_prompt, bayesian, protegi, random")
