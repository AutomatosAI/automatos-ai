"""
Orchestrator LLM - Master reasoning engine for Automatos AI
============================================================

PRD-16: Core LLM wrapper that implements Software 3.0 paradigm.
Provides reasoning capabilities with function calling support.

References:
- Karpathy's Software 3.0: Programming LLMs in English
- Context Engineering: Mathematical + LLM reasoning
- Tool-Augmented Reasoning: Function calling patterns
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Use existing LLM infrastructure
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.llm_provider import LLMManager, LLMConfig, LLMProvider as LLMProviderEnum
from config import config

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Different reasoning modes for the orchestrator"""
    ANALYTICAL = "analytical"      # Logical, step-by-step reasoning
    CREATIVE = "creative"          # Exploratory, innovative thinking
    ADAPTIVE = "adaptive"          # Context-aware, flexible reasoning
    META_COGNITIVE = "meta"        # Reasoning about reasoning


@dataclass
class LLMResponse:
    """Response from LLM with reasoning trace"""
    content: str                              # Raw response content
    reasoning: Optional[str] = None           # Extracted reasoning
    confidence: float = 0.0                   # Confidence score (0-1)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0


class OrchestratorLLM:
    """
    Master Orchestrator LLM implementing Software 3.0 paradigm.
    
    This is the brain of the orchestration system, providing:
    - Natural language reasoning
    - Function calling capabilities
    - Meta-cognitive monitoring
    - Adaptive decision making
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_mode: ReasoningMode = ReasoningMode.ADAPTIVE
    ):
        """
        Initialize the Orchestrator LLM.
        
        Args:
            model: Model to use (default from config)
            provider: Provider to use (default from config)
            temperature: Creativity level (0-1)
            max_tokens: Maximum response tokens
            reasoning_mode: Default reasoning mode
        """
        self.model = model or config.LLM_MODEL
        self.provider = provider or config.LLM_PROVIDER
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_mode = reasoning_mode
        
        # Initialize LLM manager
        self.llm_manager = self._create_llm_manager()
        
        # Function calling setup
        self.function_registry = None  # Set by orchestrator
        self.function_executor = None  # Set by orchestrator
        
        # Reasoning history for meta-cognition
        self.reasoning_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        logger.info(
            f"OrchestratorLLM initialized: provider={self.provider}, "
            f"model={self.model}, mode={reasoning_mode.value}"
        )
    
    def _create_llm_manager(self) -> LLMManager:
        """Create LLM manager with configuration"""
        provider_map = {
            "openai": LLMProviderEnum.OPENAI,
            "anthropic": LLMProviderEnum.ANTHROPIC
        }
        
        provider_enum = provider_map.get(self.provider.lower())
        if not provider_enum:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Get API key from credential resolver
        from services.credential_resolver import get_credential_resolver
        resolver = get_credential_resolver()
        
        api_key = None
        if provider_enum == LLMProviderEnum.OPENAI:
            api_key = resolver.get_credential_field("development_openai", "api_key")
        elif provider_enum == LLMProviderEnum.ANTHROPIC:
            api_key = resolver.get_credential_field("development_anthropic", "api_key")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")
        
        llm_config = LLMConfig(
            provider=provider_enum,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key
        )
        
        return LLMManager(config=llm_config)
    
    async def generate_with_reasoning(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_mode: Optional[ReasoningMode] = None,
        require_confidence: bool = True
    ) -> LLMResponse:
        """
        Generate response with explicit reasoning.
        
        Args:
            prompt: The prompt to process
            context: Additional context for reasoning
            reasoning_mode: Override default reasoning mode
            require_confidence: Whether to require confidence score
            
        Returns:
            LLMResponse with reasoning trace
        """
        start_time = asyncio.get_event_loop().time()
        mode = reasoning_mode or self.reasoning_mode
        
        # Build reasoning-enhanced prompt
        enhanced_prompt = self._build_reasoning_prompt(
            prompt, context, mode, require_confidence
        )
        
        # Generate response
        messages = [{"role": "user", "content": enhanced_prompt}]
        
        try:
            raw_response = await self.llm_manager.generate_response(messages)
            
            # Parse response for structured output
            parsed = self._parse_reasoning_response(raw_response.content)
            
            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Build response
            response = LLMResponse(
                content=parsed.get("response", raw_response.content),
                reasoning=parsed.get("reasoning", ""),
                confidence=float(parsed.get("confidence", 0.0)),
                metadata={
                    "mode": mode.value,
                    "context_provided": context is not None,
                    "model": self.model,
                    "provider": self.provider
                },
                processing_time=processing_time,
                tokens_used=raw_response.usage.get("total_tokens", 0),
                cost=self._calculate_cost(raw_response.usage)
            )
            
            # Store in history for meta-cognition
            self._update_reasoning_history(prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            raise
    
    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        function_executor: Optional[Callable] = None,
        max_function_calls: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate response with function calling support.
        
        This implements the Tool-Augmented Reasoning pattern where
        the LLM can call functions to gather information before
        making decisions.
        
        Args:
            prompt: The prompt to process
            functions: List of available function specifications
            function_executor: Async function to execute function calls
            max_function_calls: Maximum number of function calls allowed
            context: Additional context
            
        Returns:
            LLMResponse with function call history
        """
        start_time = asyncio.get_event_loop().time()
        function_calls = []
        total_tokens = 0
        
        # Build initial prompt with function descriptions
        enhanced_prompt = self._build_function_prompt(prompt, functions, context)
        
        messages = [{"role": "user", "content": enhanced_prompt}]
        function_call_count = 0
        
        while function_call_count < max_function_calls:
            # Generate response
            raw_response = await self.llm_manager.generate_response(messages)
            total_tokens += raw_response.usage.get("total_tokens", 0)
            
            # Check for function calls in response
            function_request = self._extract_function_call(raw_response.content)
            
            if not function_request:
                # No more function calls, return final response
                parsed = self._parse_function_response(raw_response.content)
                
                return LLMResponse(
                    content=parsed.get("response", raw_response.content),
                    reasoning=parsed.get("reasoning", ""),
                    confidence=float(parsed.get("confidence", 0.0)),
                    function_calls=function_calls,
                    metadata={
                        "functions_available": len(functions),
                        "function_calls_made": len(function_calls),
                        "model": self.model
                    },
                    processing_time=asyncio.get_event_loop().time() - start_time,
                    tokens_used=total_tokens,
                    cost=self._calculate_cost({"total_tokens": total_tokens})
                )
            
            # Execute function if executor provided
            if function_executor:
                try:
                    result = await function_executor(
                        function_request["name"],
                        function_request.get("parameters", {})
                    )
                    
                    # Record function call
                    function_calls.append({
                        "name": function_request["name"],
                        "parameters": function_request.get("parameters", {}),
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Add function result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": raw_response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Function result: {json.dumps(result)}"
                    })
                    
                except Exception as e:
                    logger.error(f"Function execution failed: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"Function execution failed: {str(e)}"
                    })
            
            function_call_count += 1
        
        # Max function calls reached
        logger.warning(f"Max function calls ({max_function_calls}) reached")
        
        return LLMResponse(
            content="Maximum function calls reached. Unable to complete reasoning.",
            reasoning="Exceeded function call limit",
            confidence=0.0,
            function_calls=function_calls,
            processing_time=asyncio.get_event_loop().time() - start_time,
            tokens_used=total_tokens
        )
    
    def _build_reasoning_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        mode: ReasoningMode,
        require_confidence: bool
    ) -> str:
        """Build prompt with reasoning instructions"""
        
        mode_instructions = {
            ReasoningMode.ANALYTICAL: (
                "Use step-by-step logical reasoning. "
                "Break down the problem systematically. "
                "Consider all relevant factors."
            ),
            ReasoningMode.CREATIVE: (
                "Think creatively and explore innovative solutions. "
                "Consider unconventional approaches. "
                "Look for novel connections."
            ),
            ReasoningMode.ADAPTIVE: (
                "Adapt your reasoning to the context. "
                "Consider the specific situation and constraints. "
                "Be flexible in your approach."
            ),
            ReasoningMode.META_COGNITIVE: (
                "Reflect on the reasoning process itself. "
                "Evaluate the quality of your thinking. "
                "Consider alternative reasoning strategies."
            )
        }
        
        # Build response structure conditionally
        response_structure = '"reasoning": "Your step-by-step reasoning process",\n    "response": "Your final answer/decision",'
        if require_confidence:
            response_structure += '\n    "confidence": 0.0 to 1.0,'
        response_structure += '\n    "key_factors": ["List of key factors considered"]'
        
        enhanced = f"""
{mode_instructions[mode]}

TASK: {prompt}

{f"CONTEXT: {json.dumps(context, indent=2)}" if context else ""}

Please provide your response in the following JSON structure:
{{
    {response_structure}
}}

Think carefully and explain your reasoning clearly.
"""
        return enhanced
    
    def _build_function_prompt(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt with function calling instructions"""
        
        function_descriptions = []
        for func in functions:
            desc = f"- {func['name']}: {func['description']}"
            if func.get("parameters"):
                desc += f"\n  Parameters: {json.dumps(func['parameters'], indent=4)}"
            function_descriptions.append(desc)
        
        enhanced = f"""
You have access to the following functions to help with your task:

{chr(10).join(function_descriptions)}

To call a function, use this exact format:
FUNCTION_CALL: {{
    "name": "function_name",
    "parameters": {{...}}
}}

After receiving function results, continue your reasoning and either:
1. Call more functions if needed
2. Provide your final response

TASK: {prompt}

{f"CONTEXT: {json.dumps(context, indent=2)}" if context else ""}

When you have gathered enough information and are ready to provide your final answer, 
respond with:

FINAL_RESPONSE: {{
    "reasoning": "Your complete reasoning process",
    "response": "Your final answer/decision",
    "confidence": 0.0 to 1.0,
    "function_calls_made": ["List of functions you called"]
}}

Begin by analyzing what information you need to complete this task.
"""
        return enhanced
    
    def _parse_reasoning_response(self, content: str) -> Dict[str, Any]:
        """Parse structured reasoning response"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to basic parsing
        return {
            "response": content,
            "reasoning": "",
            "confidence": 0.5
        }
    
    def _parse_function_response(self, content: str) -> Dict[str, Any]:
        """Parse response with function calls"""
        try:
            # Look for FINAL_RESPONSE marker
            if "FINAL_RESPONSE:" in content:
                json_str = content.split("FINAL_RESPONSE:")[1].strip()
                return json.loads(json_str)
        except:
            pass
        
        return self._parse_reasoning_response(content)
    
    def _extract_function_call(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract function call from response"""
        try:
            if "FUNCTION_CALL:" in content:
                json_str = content.split("FUNCTION_CALL:")[1].strip()
                # Find the JSON object
                import re
                json_match = re.search(r'\{.*?\}', json_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to extract function call: {e}")
        
        return None
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage"""
        # Pricing per 1K tokens (approximate)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4o": {"input": 0.0025, "output": 0.01},  # GPT-4o pricing
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015}
        }
        
        model_pricing = pricing.get(self.model, {"input": 0.01, "output": 0.03})
        
        # Estimate input/output split (rough approximation)
        total_tokens = usage.get("total_tokens", 0)
        input_tokens = usage.get("prompt_tokens", total_tokens * 0.7)
        output_tokens = usage.get("completion_tokens", total_tokens * 0.3)
        
        cost = (input_tokens / 1000 * model_pricing["input"] +
                output_tokens / 1000 * model_pricing["output"])
        
        return round(cost, 6)
    
    def _update_reasoning_history(self, prompt: str, response: LLMResponse):
        """Update reasoning history for meta-cognition"""
        self.reasoning_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:200],  # Truncate for storage
            "reasoning_mode": response.metadata.get("mode"),
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost
        })
        
        # Trim history if needed
        if len(self.reasoning_history) > self.max_history:
            self.reasoning_history = self.reasoning_history[-self.max_history:]
    
    async def reflect_on_reasoning(self) -> Dict[str, Any]:
        """
        Meta-cognitive reflection on recent reasoning.
        Analyzes patterns, identifies issues, suggests improvements.
        """
        if len(self.reasoning_history) < 5:
            return {
                "status": "insufficient_data",
                "message": "Not enough reasoning history for reflection"
            }
        
        # Analyze recent reasoning patterns
        recent = self.reasoning_history[-10:]
        
        avg_confidence = sum(h["confidence"] for h in recent) / len(recent)
        avg_time = sum(h["processing_time"] for h in recent) / len(recent)
        total_cost = sum(h["cost"] for h in recent)
        
        # Generate reflection prompt
        reflection_prompt = f"""
Analyze the following reasoning performance metrics:

- Average confidence: {avg_confidence:.2f}
- Average processing time: {avg_time:.2f}s
- Total cost: ${total_cost:.4f}
- Number of decisions: {len(recent)}

Recent reasoning modes used: {[h["reasoning_mode"] for h in recent]}

Provide a meta-cognitive assessment:
1. Are the confidence levels appropriate?
2. Is the reasoning efficiency acceptable?
3. Are there patterns that suggest issues?
4. What improvements would you recommend?

Respond with a JSON structure containing your assessment.
"""
        
        response = await self.generate_with_reasoning(
            reflection_prompt,
            reasoning_mode=ReasoningMode.META_COGNITIVE
        )
        
        return {
            "status": "completed",
            "metrics": {
                "avg_confidence": avg_confidence,
                "avg_processing_time": avg_time,
                "total_cost": total_cost
            },
            "assessment": response.reasoning,
            "recommendations": response.content
        }

