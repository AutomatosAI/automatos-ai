"""
Response Parser - Parses structured LLM responses
==================================================

PRD-16: Handles parsing of LLM responses containing:
- Reasoning traces
- Function calls
- Confidence scores
- Structured decisions
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Type of LLM response"""
    REASONING = "reasoning"          # Pure reasoning response
    FUNCTION_CALL = "function_call"  # Contains function calls
    DECISION = "decision"            # Final decision
    ERROR = "error"                  # Error response


@dataclass
class ParsedResponse:
    """Parsed LLM response with extracted components"""
    type: ResponseType
    content: str                     # Main response content
    reasoning: Optional[str] = None  # Extracted reasoning
    confidence: float = 0.0          # Confidence score
    decision: Optional[str] = None   # Extracted decision
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""           # Original unparsed response
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.type.value,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "decision": self.decision,
            "function_calls": self.function_calls,
            "metadata": self.metadata
        }


class ResponseParser:
    """
    Parses LLM responses to extract structured information.
    
    Handles various response formats:
    - JSON structures
    - Markdown code blocks
    - Tagged sections (REASONING:, DECISION:, etc.)
    - Function call formats
    """
    
    def __init__(self):
        # Patterns for different response formats
        self.patterns = {
            "json_block": r"```json\s*(.*?)\s*```",
            "json_inline": r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            "function_call": r"FUNCTION_CALL:\s*(\{.*?\})",
            "final_response": r"FINAL_RESPONSE:\s*(\{.*?\})",
            "reasoning_section": r"REASONING:\s*(.*?)(?=DECISION:|CONFIDENCE:|$)",
            "decision_section": r"DECISION:\s*(.*?)(?=REASONING:|CONFIDENCE:|$)",
            "confidence_section": r"CONFIDENCE:\s*([\d.]+)",
            "markdown_sections": r"##\s*(.*?)\n(.*?)(?=##|\Z)"
        }
        
        logger.info("ResponseParser initialized")
    
    def parse(self, response: str) -> ParsedResponse:
        """
        Parse an LLM response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            ParsedResponse with extracted components
        """
        if not response:
            return ParsedResponse(
                type=ResponseType.ERROR,
                content="Empty response",
                raw_response=response
            )
        
        # Try different parsing strategies
        parsed = None
        
        # 1. Try to parse as function call response
        parsed = self._parse_function_response(response)
        if parsed:
            return parsed
        
        # 2. Try to parse as JSON response
        parsed = self._parse_json_response(response)
        if parsed:
            return parsed
        
        # 3. Try to parse as tagged response
        parsed = self._parse_tagged_response(response)
        if parsed:
            return parsed
        
        # 4. Try to parse as markdown response
        parsed = self._parse_markdown_response(response)
        if parsed:
            return parsed
        
        # 5. Fallback: treat as plain text
        return ParsedResponse(
            type=ResponseType.REASONING,
            content=response.strip(),
            raw_response=response
        )
    
    def _parse_function_response(self, response: str) -> Optional[ParsedResponse]:
        """Parse response containing function calls"""
        function_calls = []
        
        # Look for FUNCTION_CALL markers
        function_matches = re.finditer(self.patterns["function_call"], response, re.DOTALL)
        for match in function_matches:
            try:
                func_data = json.loads(match.group(1))
                function_calls.append(func_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function call: {match.group(1)}")
        
        # Look for FINAL_RESPONSE marker
        final_match = re.search(self.patterns["final_response"], response, re.DOTALL)
        if final_match:
            try:
                final_data = json.loads(final_match.group(1))
                return ParsedResponse(
                    type=ResponseType.FUNCTION_CALL if function_calls else ResponseType.DECISION,
                    content=final_data.get("response", ""),
                    reasoning=final_data.get("reasoning"),
                    confidence=float(final_data.get("confidence", 0.0)),
                    decision=final_data.get("response"),
                    function_calls=function_calls,
                    metadata={"function_calls_made": final_data.get("function_calls_made", [])},
                    raw_response=response
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse final response: {e}")
        
        if function_calls:
            # Has function calls but no final response yet
            return ParsedResponse(
                type=ResponseType.FUNCTION_CALL,
                content="Function calls pending",
                function_calls=function_calls,
                raw_response=response
            )
        
        return None
    
    def _parse_json_response(self, response: str) -> Optional[ParsedResponse]:
        """Parse JSON-formatted response"""
        # Try JSON code block first
        json_match = re.search(self.patterns["json_block"], response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try inline JSON
            json_match = re.search(self.patterns["json_inline"], response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        try:
            data = json.loads(json_str)
            
            # Determine response type based on content
            response_type = ResponseType.REASONING
            if "decision" in data:
                response_type = ResponseType.DECISION
            elif "function_calls" in data:
                response_type = ResponseType.FUNCTION_CALL
            
            return ParsedResponse(
                type=response_type,
                content=data.get("response", data.get("content", "")),
                reasoning=data.get("reasoning"),
                confidence=float(data.get("confidence", 0.0)),
                decision=data.get("decision"),
                function_calls=data.get("function_calls", []),
                metadata={
                    k: v for k, v in data.items()
                    if k not in ["response", "content", "reasoning", "confidence", "decision", "function_calls"]
                },
                raw_response=response
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None
    
    def _parse_tagged_response(self, response: str) -> Optional[ParsedResponse]:
        """Parse response with tagged sections"""
        reasoning = None
        decision = None
        confidence = 0.0
        
        # Extract reasoning section
        reasoning_match = re.search(self.patterns["reasoning_section"], response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # Extract decision section
        decision_match = re.search(self.patterns["decision_section"], response, re.DOTALL | re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).strip()
        
        # Extract confidence
        confidence_match = re.search(self.patterns["confidence_section"], response, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                pass
        
        if reasoning or decision or confidence > 0:
            return ParsedResponse(
                type=ResponseType.DECISION if decision else ResponseType.REASONING,
                content=decision or reasoning or response,
                reasoning=reasoning,
                confidence=confidence,
                decision=decision,
                raw_response=response
            )
        
        return None
    
    def _parse_markdown_response(self, response: str) -> Optional[ParsedResponse]:
        """Parse markdown-formatted response"""
        sections = {}
        
        # Find all markdown sections
        section_matches = re.finditer(self.patterns["markdown_sections"], response, re.DOTALL)
        for match in section_matches:
            section_name = match.group(1).strip().lower()
            section_content = match.group(2).strip()
            sections[section_name] = section_content
        
        if not sections:
            return None
        
        # Map sections to ParsedResponse fields
        reasoning = sections.get("reasoning") or sections.get("analysis")
        decision = sections.get("decision") or sections.get("conclusion")
        confidence_str = sections.get("confidence")
        
        confidence = 0.0
        if confidence_str:
            # Try to extract number from confidence section
            conf_match = re.search(r"([\d.]+)", confidence_str)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    if confidence > 1:
                        confidence = confidence / 100  # Convert percentage
                except ValueError:
                    pass
        
        return ParsedResponse(
            type=ResponseType.DECISION if decision else ResponseType.REASONING,
            content=decision or reasoning or response,
            reasoning=reasoning,
            confidence=confidence,
            decision=decision,
            metadata={"sections": list(sections.keys())},
            raw_response=response
        )
    
    def extract_confidence(self, text: str) -> float:
        """
        Extract confidence score from text.
        
        Handles various formats:
        - "0.85", "85%", "high confidence (0.9)"
        - "very confident", "somewhat confident", etc.
        """
        # Try numeric extraction
        numeric_patterns = [
            r"(\d+\.?\d*)\s*%",           # Percentage
            r"confidence[:\s]+(\d+\.?\d*)", # Confidence: 0.8
            r"\((\d+\.?\d*)\)",            # (0.8)
            r"^(\d+\.?\d*)$"               # Just a number
        ]
        
        for pattern in numeric_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 1:
                        value = value / 100  # Convert percentage
                    return min(1.0, max(0.0, value))
                except ValueError:
                    continue
        
        # Try qualitative extraction
        qualitative_map = {
            "very high": 0.95,
            "high": 0.85,
            "moderate": 0.65,
            "medium": 0.65,
            "low": 0.35,
            "very low": 0.15,
            "certain": 0.95,
            "confident": 0.85,
            "somewhat confident": 0.65,
            "uncertain": 0.35,
            "not confident": 0.15
        }
        
        text_lower = text.lower()
        for phrase, score in qualitative_map.items():
            if phrase in text_lower:
                return score
        
        return 0.5  # Default to medium confidence
    
    def validate_function_call(self, func_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a function call structure.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(func_data, dict):
            return False, "Function call must be a dictionary"
        
        if "name" not in func_data:
            return False, "Function call missing 'name' field"
        
        if not isinstance(func_data["name"], str):
            return False, "Function name must be a string"
        
        if "parameters" in func_data and not isinstance(func_data["parameters"], dict):
            return False, "Function parameters must be a dictionary"
        
        return True, None
