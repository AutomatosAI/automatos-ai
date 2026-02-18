"""
Orchestrator Evaluation Module
===============================
PRD-59 US-007/US-008: Evaluation datasets and optimization for Stage 1-2 prompts.

Provides:
  - Evaluation datasets (input/expected-output pairs) for task decomposition and agent selection
  - FutureAGI integration for prompt quality scoring
  - Bayesian optimization loop for prompt improvement
"""

from .datasets import (
    EvalCase,
    TASK_DECOMPOSER_EVAL_DATASET,
    AGENT_SELECTOR_EVAL_DATASET,
    get_eval_dataset,
)
from .runner import PromptEvalRunner

__all__ = [
    "EvalCase",
    "TASK_DECOMPOSER_EVAL_DATASET",
    "AGENT_SELECTOR_EVAL_DATASET",
    "get_eval_dataset",
    "PromptEvalRunner",
]
