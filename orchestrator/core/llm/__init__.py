"""
LLM-Driven Orchestration Components
====================================

PRD-16: Software 3.0 Transformation - LLM-driven reasoning system
for adaptive, context-aware orchestration.

This module implements the core LLM infrastructure for:
- Function calling and execution
- Reasoning-based decision making
- Meta-cognitive monitoring
- Adaptive orchestration strategies
"""

from .orchestrator_llm import OrchestratorLLM, LLMResponse
from .function_registry import FunctionRegistry, FunctionSpec
from .function_executor import FunctionExecutor, FunctionResult
from .response_parser import ResponseParser, ParsedResponse

__all__ = [
    'OrchestratorLLM',
    'LLMResponse',
    'FunctionRegistry',
    'FunctionSpec',
    'FunctionExecutor',
    'FunctionResult',
    'ResponseParser',
    'ParsedResponse'
]
