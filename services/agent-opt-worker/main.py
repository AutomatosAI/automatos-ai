"""
PRD-58 Phase 1B: Agent-Opt Worker Service
==========================================

Isolated microservice for FutureAGI prompt optimization.
Runs agent-opt library in its own container to avoid dependency
conflicts with the main orchestrator.

Endpoints:
  POST /optimize  — Run prompt optimization (synchronous, returns improved prompt)
  GET  /health    — Health check
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-opt-worker")

app = FastAPI(title="Agent-Opt Worker", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    prompt_content: str = Field(..., description="The current system prompt to optimize")
    dataset: List[Dict[str, str]] = Field(
        ...,
        description="List of {input, output} pairs from live traffic",
        min_length=1,
    )
    eval_template: str = Field(
        default="is_helpful",
        description="FutureAGI eval template name for scoring",
    )
    algorithm: str = Field(
        default="meta_prompt",
        description="Optimization algorithm: meta_prompt, bayesian, protegi, random",
    )
    num_rounds: int = Field(default=3, ge=1, le=20)
    teacher_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for generating improved prompts",
    )
    task_description: Optional[str] = Field(
        default=None,
        description="Description of what the prompt should accomplish",
    )


class OptimizeResponse(BaseModel):
    optimized_prompt: str
    final_score: float
    initial_score: Optional[float] = None
    rounds_completed: int
    algorithm: str
    history: List[Dict[str, Any]] = []
    duration_seconds: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-opt-worker"}


# ---------------------------------------------------------------------------
# Optimize endpoint
# ---------------------------------------------------------------------------

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """
    Run synchronous prompt optimization using agent-opt library.
    This can take 1-5 minutes depending on rounds and dataset size.
    """
    start = time.time()

    fi_api_key = os.getenv("FUTUREAGI_API_KEY")
    fi_secret_key = os.getenv("FUTUREAGI_SECRET_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not fi_api_key or not fi_secret_key:
        raise HTTPException(status_code=500, detail="FutureAGI keys not configured")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        from fi.opt.base.evaluator import Evaluator
        from fi.opt.datamappers import BasicDataMapper
        from fi.opt.generators import LiteLLMGenerator
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"agent-opt not installed correctly: {e}",
        )

    # --- Set up evaluator (FutureAGI scoring) ---
    evaluator = Evaluator(
        eval_template=req.eval_template,
        eval_model_name="turing_flash",
        fi_api_key=fi_api_key,
        fi_secret_key=fi_secret_key,
    )

    # --- Data mapper: our dataset keys → eval input keys ---
    data_mapper = BasicDataMapper(
        key_map={"input": "input", "output": "output"}
    )

    # --- Teacher model (generates improved prompts) ---
    teacher = LiteLLMGenerator(
        model=req.teacher_model,
        prompt_template="{prompt}",
    )

    # --- Task description ---
    task_desc = req.task_description or (
        "This is a system prompt for an AI assistant. "
        "Optimize it to produce more helpful, concise, and complete responses."
    )

    # --- Select optimizer ---
    try:
        optimizer = _create_optimizer(req.algorithm, teacher, req.num_rounds)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # --- Run optimization ---
    logger.info(
        f"Starting {req.algorithm} optimization: "
        f"{req.num_rounds} rounds, {len(req.dataset)} examples, "
        f"eval={req.eval_template}, teacher={req.teacher_model}"
    )

    try:
        result = optimizer.optimize(
            evaluator=evaluator,
            data_mapper=data_mapper,
            dataset=req.dataset,
            initial_prompts=[req.prompt_content],
            task_description=task_desc,
            eval_subset_size=min(len(req.dataset), 10),
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

    duration = time.time() - start

    # --- Extract results ---
    best_prompt = result.best_generator.get_prompt_template()
    final_score = result.final_score

    history = []
    if hasattr(result, "history") and result.history:
        for i, iteration in enumerate(result.history):
            history.append({
                "round": i + 1,
                "score": getattr(iteration, "average_score", None),
                "prompt_preview": getattr(iteration, "prompt", "")[:200],
            })

    initial_score = history[0]["score"] if history else None

    logger.info(
        f"Optimization complete: {final_score:.4f} "
        f"({req.num_rounds} rounds, {duration:.1f}s)"
    )

    return OptimizeResponse(
        optimized_prompt=best_prompt,
        final_score=final_score,
        initial_score=initial_score,
        rounds_completed=req.num_rounds,
        algorithm=req.algorithm,
        history=history,
        duration_seconds=round(duration, 1),
    )


def _create_optimizer(algorithm: str, teacher, num_rounds: int):
    """Factory for optimizer instances."""
    if algorithm == "meta_prompt":
        from fi.opt.optimizers import MetaPromptOptimizer
        return MetaPromptOptimizer(
            teacher_generator=teacher,
            num_rounds=num_rounds,
        )
    elif algorithm == "bayesian":
        from fi.opt.optimizers import BayesianSearchOptimizer
        return BayesianSearchOptimizer(
            inference_model_name=teacher.model,
            n_trials=num_rounds * 4,
            min_examples=2,
            max_examples=5,
        )
    elif algorithm == "protegi":
        from fi.opt.optimizers import ProTeGi
        return ProTeGi(
            teacher_generator=teacher,
            num_gradients=4,
            beam_size=4,
            num_rounds=num_rounds,
        )
    elif algorithm == "random":
        from fi.opt.optimizers import RandomSearchOptimizer
        return RandomSearchOptimizer(
            teacher_generator=teacher,
            num_candidates=num_rounds * 3,
        )
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Options: meta_prompt, bayesian, protegi, random"
        )
