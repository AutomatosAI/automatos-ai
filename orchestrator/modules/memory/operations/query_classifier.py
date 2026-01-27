from enum import Enum
from typing import Optional, Dict, Any, List
import logging
import json
import re
from pydantic import BaseModel, Field

# Try to import LLMManager, handle potential circular imports or location changes
try:
    from modules.llm.manager import LLMManager
except ImportError:
    # Fallback or placeholder until correct path is found
    # In orchestrator/modules/llm/manager.py usually
    pass

logger = logging.getLogger(__name__)

class QueryIntent(str, Enum):
    GREETING = "greeting"
    FACTUAL_QUERY = "factual_query"     # General knowledge, no memory needed
    MEMORY_RETRIEVAL = "memory_retrieval" # Explicitly asks for something remembered
    TASK_EXECUTION = "task_execution"   # "Create a file..."
    CLARIFICATION = "clarification"     # Following up on previous message
    FEEDBACK = "feedback"               # "Good job", "That's wrong"
    SENSITIVE = "sensitive"             # Passwords, keys (should definitely not store/retrieve carelessly)

class ClassificationResult(BaseModel):
    intent: QueryIntent
    confidence: float
    reasoning: str
    extract_entities: List[str] = Field(default_factory=list)
    requires_memory: bool

class QueryClassifier:
    """Classifies queries to determine if memory retrieval is needed."""
    
    def __init__(self, llm_manager: Any): # using Any to avoid strict type dependency if imports fail
        self.llm_manager = llm_manager
        
    async def classify(self, query: str, recent_history: List[Dict[str, str]] = None) -> ClassificationResult:
        """
        Classifies the user query to determine intent and memory needs.
        """
        if not query:
            return ClassificationResult(
                intent=QueryIntent.GREETING, 
                confidence=1.0, 
                reasoning="Empty query", 
                requires_memory=False
            )
            
        system_prompt = """You are an intent classifier for an AI agent's memory system.
        Your goal is to determine if the user's query requires retrieving information from long-term memory.
        
        Intents:
        - GREETING: Hello, hi, how are you. (Memory: NO)
        - FACTUAL_QUERY: General knowledge questions (e.g., "What is the capital of France?"). (Memory: NO, unless asking about user specific facts)
        - MEMORY_RETRIEVAL: Asking about past interactions, preferences, stored files, or specific project details previously argued. (Memory: YES)
        - TASK_EXECUTION: Commands to do something. (Memory: MAYBE, if context is needed)
        - CLARIFICATION: Follow-up to immediately preceding conversation. (Memory: NO, widely handled by context window)
        - FEEDBACK: User providing feedback. (Memory: YES, should be stored potentially)
        - SENSITIVE: Queries involving secrets. (Memory: CAREFUL - retrieval ok, storage careful)
        
        Return a JSON object:
        {
            "intent": "intent_enum_value",
            "confidence": 0.9,
            "reasoning": "Explain why",
            "requires_memory": true/false
        }
        """
        
        # Construct simplified history for context
        history_context = ""
        if recent_history:
            history_context = "\nRecent History:\n" + "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in recent_history[-3:]])
        
        prompt = f"""{history_context}
        
        User Query: "{query}"
        
        Classify this query."""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = await self.llm_manager.generate_response(messages=messages, tools=[])

            content = response.content if hasattr(response, "content") else (
                str(response) if response is not None else ""
            )
            data = None
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
                if match:
                    try:
                        data = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        pass
                if not data:
                    start = content.find("{")
                    if start != -1:
                        depth, end = 0, start
                        for i, c in enumerate(content[start:], start):
                            if c == "{":
                                depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0:
                                    end = i
                                    break
                        try:
                            data = json.loads(content[start : end + 1])
                        except json.JSONDecodeError:
                            pass
            if not isinstance(data, dict):
                raise ValueError("Could not parse JSON from classifier response")

            return ClassificationResult(**data)

        except Exception as e:
            logger.error("Query classification failed: %s", e)
            # Fallback: Assume memory is needed if it's not a short greeting
            requires_memory = len(query) > 10
            return ClassificationResult(
                intent=QueryIntent.MEMORY_RETRIEVAL if requires_memory else QueryIntent.GREETING,
                confidence=0.0,
                reasoning=f"Fallback due to error: {e}",
                requires_memory=requires_memory
            )
