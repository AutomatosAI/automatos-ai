"""
PRD-58: FutureAGI Worker Service
=================================

Single gateway for ALL FutureAGI operations. Runs in its own container
with the full SDK (agent-opt + ai-evaluation) isolated from the main
orchestrator.

Endpoints:
  POST /assess    - Score a prompt using quality metric templates
  POST /safety    - Run safety checks on a prompt
  POST /optimize  - Start async prompt optimization (returns job_id)
  GET  /optimize/{job_id} - Poll optimization status/result
  POST /score     - Score a live chat input/output pair
  GET  /health    - Health check
  GET  /metrics   - Prometheus metrics

PRD-262 US-006: every route but /health and /metrics requires the
X-Internal-Token the API sends (AGENT_OPT_WORKER_TOKEN, worker_auth.py), the
worker refuses to start without that token, and no API docs are served.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import AfterValidator, BaseModel, Field

from automatos_logging import setup_logging
from automatos_metrics import add_fastapi_metrics
from worker_auth import require_internal_token
from worker_config import ALLOWED_TEACHER_MODELS, MAX_DATASET_ITEMS, internal_token, max_concurrent_jobs
setup_logging(service="agent-opt-worker")
logger = logging.getLogger("agent-opt-worker")

# PRD-262 US-006: no /docs, /redoc or /openapi.json. Every route but /health and
# /metrics is declared on `router`, whose dependency requires X-Internal-Token;
# internal_token() stops the start-up when the token is unset or blank.
app = FastAPI(title="FutureAGI Worker", version="3.0.0", docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter(dependencies=[Depends(require_internal_token(internal_token()))])

# Prometheus metrics endpoint + request tracking
add_fastapi_metrics(app, service="agent-opt-worker")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_keys():
    api_key = os.getenv("FUTUREAGI_API_KEY") or os.getenv("FI_API_KEY")
    secret_key = os.getenv("FUTUREAGI_SECRET_KEY") or os.getenv("FI_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("FutureAGI keys not configured")
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
            timeout=55,
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
                    logger.warning(f"[{template}] unparseable output: {output!r}, defaulting to score=0")
                    passed = False
                    score = 0.0
        else:
            logger.warning(f"[{template}] unexpected output type {type(output)}: {output!r}, defaulting to score=0")
            passed = False
            score = 0.0

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


# PRD-262 US-006: requests name only known templates (an unknown name still cost a
# turing_flash evaluation), at most one entry per template (/assess and /score
# start one thread per entry), and only the teacher model the platform sends.
def _known_template(name: str) -> str:
    if name not in TEMPLATE_CONFIG:
        raise ValueError(f"unknown scoring template {name!r}; known: {', '.join(TEMPLATE_CONFIG)}")
    return name


def _at_most_one_per_template(names: Optional[List[str]]) -> Optional[List[str]]:
    if names is None:
        return names
    if len(names) > len(TEMPLATE_CONFIG):
        raise ValueError(f"at most {len(TEMPLATE_CONFIG)} metrics")
    if len(set(names)) != len(names):
        raise ValueError("each metric may be named once")
    return names


def _allowed_teacher(model: str) -> str:
    if model not in ALLOWED_TEACHER_MODELS:
        raise ValueError(f"teacher_model must be one of: {', '.join(sorted(ALLOWED_TEACHER_MODELS))}")
    return model


TemplateName = Annotated[str, AfterValidator(_known_template)]
MetricNames = Annotated[Optional[List[TemplateName]], AfterValidator(_at_most_one_per_template)]
TeacherModel = Annotated[str, AfterValidator(_allowed_teacher)]


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
    metrics: MetricNames = None


class SafetyRequest(BaseModel):
    prompt_content: str


class ScoreRequest(BaseModel):
    input_text: str
    output_text: str
    context_text: Optional[str] = None
    metrics: MetricNames = None


class OptimizeRequest(BaseModel):
    prompt_content: str
    dataset: List[Dict[str, str]] = Field(..., min_length=1, max_length=MAX_DATASET_ITEMS)
    scoring_template: TemplateName = "is_helpful"
    algorithm: str = "meta_prompt"
    num_rounds: int = Field(default=3, ge=1, le=20)
    teacher_model: TeacherModel = "gpt-4o-mini"
    task_description: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "futureagi-worker", "version": "3.0.0"}


# ---------------------------------------------------------------------------
# Assess - score a prompt using quality metrics
# ---------------------------------------------------------------------------

@router.post("/assess")
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

SAFETY_PREAMBLE = (
    "NOTE: The following text is a SYSTEM PROMPT (instructions for an AI assistant), "
    "not user-generated content. It may contain instructional language about handling "
    "sensitive topics. Evaluate the actual intent, not the instructional framing.\n\n"
)

@router.post("/safety")
def safety(req: SafetyRequest):
    start = time.time()
    safety_templates = ["toxicity", "prompt_injection", "content_moderation"]

    # Prefix system prompt context so the safety model understands this is
    # instructional text, not user content — reduces false positives
    contextualized = SAFETY_PREAMBLE + req.prompt_content

    # Run all safety templates concurrently
    raw_results = {}
    with ThreadPoolExecutor(max_workers=len(safety_templates)) as pool:
        futures = {}
        for template in safety_templates:
            inputs = _build_inputs(template, contextualized, contextualized)
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

    resolved = [c for c in checks.values() if c.get("safe") is not None]
    all_safe = all(c["safe"] for c in resolved) if resolved else None

    return {"safe": all_safe, "checks": checks, "duration": round(time.time() - start, 1)}


# ---------------------------------------------------------------------------
# Score - score a real chat exchange (live traffic)
# ---------------------------------------------------------------------------

@router.post("/score")
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
# Optimize - async prompt optimization via agent-opt
# ---------------------------------------------------------------------------

# In-memory job store for async optimization
_optimize_jobs: Dict[str, Dict[str, Any]] = {}
_JOB_TTL = 3600  # clean up jobs older than 1 hour
# PRD-262 US-006: at most this many jobs run at once (AGENT_OPT_MAX_CONCURRENT_JOBS);
# each runs up to 20 rounds against OpenAI and FutureAGI.
MAX_CONCURRENT_JOBS = max_concurrent_jobs()
_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
# /optimize is a plain def, so concurrent requests run in FastAPI's threadpool:
# every insert into or sweep of the job store takes this lock.
_JOBS_LOCK = threading.Lock()


def _cleanup_old_jobs():
    """Remove completed/failed jobs older than TTL to prevent memory leak."""
    now = time.time()
    with _JOBS_LOCK:
        stale = [
            jid for jid, job in list(_optimize_jobs.items())
            if job["status"] in ("completed", "failed") and now - job["created_at"] > _JOB_TTL
        ]
        for jid in stale:
            _optimize_jobs.pop(jid, None)
    if stale:
        logger.info(f"[cleanup] removed {len(stale)} stale optimization jobs")


def _escape_template_vars(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Escape {variable} placeholders so the SDK's .format() doesn't crash.
    Returns (escaped_text, list of (placeholder, original) for restoration).
    """
    replacements = []
    for match in re.finditer(r'(?<!\{)\{(\w+)\}(?!\})', text):
        original = match.group(0)
        var_name = match.group(1)
        placeholder = f"__TMPL_{var_name.upper()}__"
        replacements.append((placeholder, original))
    escaped = text
    for placeholder, original in replacements:
        escaped = escaped.replace(original, placeholder)
    return escaped, replacements


def _restore_template_vars(text: str, replacements: list[tuple[str, str]]) -> str:
    """Restore escaped template variables back to {variable} form."""
    for placeholder, original in replacements:
        text = text.replace(placeholder, original)
    return text


def _run_optimize_job(job_id: str, req: OptimizeRequest):
    """Run optimization in a background thread. Updates _optimize_jobs in place."""
    job = _optimize_jobs[job_id]
    start = time.time()

    try:
        api_key, secret_key = _get_keys()

        if not os.getenv("OPENAI_API_KEY"):
            job["status"] = "failed"
            job["error"] = "OpenAI API key not configured"
            return

        from fi.opt.base.evaluator import Evaluator
        from fi.opt.datamappers import BasicDataMapper
        from fi.opt.generators import LiteLLMGenerator

        # Escape template variables
        escaped_prompt, template_replacements = _escape_template_vars(req.prompt_content)
        if template_replacements:
            var_names = [orig for _, orig in template_replacements]
            logger.info(f"[job {job_id[:8]}] Escaped {len(template_replacements)} template vars: {var_names}")

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
            "Optimize it to produce more helpful, concise, and complete responses. "
            "IMPORTANT: Preserve any placeholder tokens like __TMPL_*__ exactly as they are."
        )

        optimizer = _create_optimizer(req.algorithm, teacher)

        logger.info(
            f"[job {job_id[:8]}] Starting {req.algorithm}: {req.num_rounds} rounds, "
            f"{len(req.dataset)} examples, template={req.scoring_template}"
        )

        job["status"] = "running"

        result = optimizer.optimize(
            evaluator=evaluator,
            data_mapper=data_mapper,
            dataset=req.dataset,
            initial_prompts=[escaped_prompt],
            task_description=task_desc,
            num_rounds=req.num_rounds,
            eval_subset_size=min(len(req.dataset), 10),
        )

        duration = time.time() - start
        best_prompt = result.best_generator.get_prompt_template()
        best_prompt = _restore_template_vars(best_prompt, template_replacements)

        history = []
        if hasattr(result, "history") and result.history:
            for i, iteration in enumerate(result.history):
                history.append({
                    "round": i + 1,
                    "score": getattr(iteration, "average_score", None),
                    "prompt_preview": getattr(iteration, "prompt", "")[:200],
                })

        logger.info(f"[job {job_id[:8]}] Complete: {result.final_score:.4f} in {duration:.1f}s")

        job["status"] = "completed"
        job["result"] = {
            "optimized_prompt": best_prompt,
            "final_score": result.final_score,
            "initial_score": history[0]["score"] if history else None,
            "rounds_completed": req.num_rounds,
            "algorithm": req.algorithm,
            "history": history,
            "duration_seconds": round(duration, 1),
            "status": "completed",
        }

    except Exception as e:
        duration = time.time() - start
        logger.exception(f"[job {job_id[:8]}] Failed after {duration:.1f}s: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["duration_seconds"] = round(duration, 1)


def _run_job_in_slot(job_id: str, req: OptimizeRequest) -> None:
    """Run one optimisation job, then free its slot however the job ends."""
    try:
        _run_optimize_job(job_id, req)
    finally:
        _JOB_SLOTS.release()


@router.post("/optimize")
def optimize(req: OptimizeRequest):
    """Start async optimization — returns job_id immediately. Poll GET /optimize/{job_id} for result."""
    # Validate upfront
    try:
        _get_keys()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    if not _JOB_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=f"At most {MAX_CONCURRENT_JOBS} optimisation jobs run at once; try again when one finishes",
        )

    # Once a slot is taken, only the started thread may free it. Anything that
    # fails before the thread starts gives the slot back, or the cap would leak
    # shut and every later /optimize would answer 429.
    try:
        _cleanup_old_jobs()
        job_id = str(uuid.uuid4())
        with _JOBS_LOCK:
            _optimize_jobs[job_id] = {
                "status": "starting",
                "created_at": time.time(),
                "result": None,
                "error": None,
            }
        thread = threading.Thread(target=_run_job_in_slot, args=(job_id, req), daemon=True)
        thread.start()
    except BaseException:
        _JOB_SLOTS.release()
        raise

    logger.info(f"[job {job_id[:8]}] Optimization job started (async)")
    return {"job_id": job_id, "status": "starting"}


@router.get("/optimize/{job_id}")
def get_optimize_status(job_id: str):
    """Poll optimization job status. Returns result when complete."""
    job = _optimize_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    elapsed = round(time.time() - job["created_at"], 1)

    if job["status"] == "completed":
        result = job["result"]
        # Clean up old job after retrieval
        return {**result, "job_id": job_id}
    elif job["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": job.get("error", "Unknown error"),
            "duration_seconds": job.get("duration_seconds", elapsed),
        }
    else:
        return {
            "job_id": job_id,
            "status": job["status"],
            "elapsed_seconds": elapsed,
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


# Last, after every route above. Each route declared on `router` carries the token
# dependency from the moment it is declared (APIRouter adds its dependencies then),
# and FastAPI 0.137+ keeps the included router as one node of app.routes instead
# of copying its routes in.
app.include_router(router)
