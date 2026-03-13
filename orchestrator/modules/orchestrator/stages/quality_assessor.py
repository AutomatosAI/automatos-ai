"""
Output Quality Assessor
=======================

Phase 4: Comprehensive Output Quality Assessment Framework

Evaluates output quality across multiple dimensions to ensure analyst-grade deliverables:
- Completeness: All requirements addressed
- Coherence: Logical flow and structure
- Accuracy: Correctness of information
- Professionalism: Formatting, tone, presentation
- Clarity: Ease of understanding

Features:
- Multi-dimensional quality scoring (0.0-1.0 scale)
- Detailed feedback with actionable suggestions
- LLM-driven intelligent assessment
- Heuristic fallbacks for reliability
- Support for different output types (reports, code, operations)
- Integration with OrchestratorConfig quality thresholds

Author: Automatos AI - Phase 4 Implementation
"""

import logging
import re
import asyncio
from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OutputType(Enum):
    """Types of outputs to assess"""
    REPORT = "report"  # Research/analysis reports
    CODE = "code"  # Code implementations
    OPERATIONS = "operations"  # Deployment/ops tasks
    DOCUMENTATION = "documentation"  # Technical docs
    ANALYSIS = "analysis"  # Data analysis
    GENERAL = "general"  # General purpose


@dataclass
class QualityDimension:
    """Individual quality dimension score and feedback"""
    name: str
    score: float  # 0.0-1.0
    weight: float  # Importance weight
    feedback: str
    suggestions: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityAssessment:
    """Complete quality assessment result"""
    overall_score: float  # 0.0-1.0, weighted average
    dimensions: Dict[str, QualityDimension]
    passes_threshold: bool
    threshold_used: float
    output_type: OutputType
    assessment_method: str  # "llm", "heuristic", "hybrid"
    
    # Detailed feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "passes_threshold": self.passes_threshold,
            "threshold_used": self.threshold_used,
            "output_type": self.output_type.value,
            "assessment_method": self.assessment_method,
            "dimensions": {
                name: {
                    "score": dim.score,
                    "weight": dim.weight,
                    "feedback": dim.feedback,
                    "suggestions": dim.suggestions
                }
                for name, dim in self.dimensions.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "improvement_suggestions": self.improvement_suggestions,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used
        }


class OutputQualityAssessor:
    """
    Comprehensive quality assessment framework
    
    Evaluates outputs across multiple dimensions to ensure professional,
    analyst-grade deliverables that meet user requirements.
    
    Assessment Dimensions:
    1. Completeness (25%): All requirements addressed
    2. Coherence (20%): Logical flow, structure, organization
    3. Accuracy (25%): Correctness, evidence-based claims
    4. Professionalism (15%): Formatting, tone, presentation
    5. Clarity (15%): Readability, ease of understanding
    
    Methods:
    - LLM-driven: Uses LLM for intelligent, context-aware assessment
    - Heuristic: Fast, rule-based assessment (fallback)
    - Hybrid: Combines LLM insights with heuristic validation
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_llm: bool = True
    ):
        """
        Initialize output quality assessor
        
        Args:
            llm_client: LLM client for intelligent assessment (optional)
            use_llm: Enable LLM-based assessment (default: True)
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        
        from config import orchestrator_config
        self.config = orchestrator_config
        
        # Quality dimension weights by output type
        self.dimension_weights = {
            OutputType.REPORT: {
                "completeness": 0.25,
                "coherence": 0.20,
                "accuracy": 0.25,
                "professionalism": 0.15,
                "clarity": 0.15
            },
            OutputType.CODE: {
                "completeness": 0.30,
                "coherence": 0.15,
                "accuracy": 0.35,
                "professionalism": 0.10,
                "clarity": 0.10
            },
            OutputType.OPERATIONS: {
                "completeness": 0.35,
                "coherence": 0.15,
                "accuracy": 0.30,
                "professionalism": 0.10,
                "clarity": 0.10
            },
            OutputType.DOCUMENTATION: {
                "completeness": 0.20,
                "coherence": 0.20,
                "accuracy": 0.20,
                "professionalism": 0.15,
                "clarity": 0.25
            },
            OutputType.ANALYSIS: {
                "completeness": 0.25,
                "coherence": 0.20,
                "accuracy": 0.30,
                "professionalism": 0.10,
                "clarity": 0.15
            },
            OutputType.GENERAL: {
                "completeness": 0.25,
                "coherence": 0.20,
                "accuracy": 0.25,
                "professionalism": 0.15,
                "clarity": 0.15
            }
        }
        
        logger.info(f"OutputQualityAssessor initialized (LLM: {self.use_llm})")
    
    async def assess_quality(
        self,
        output: str,
        requirements: str,
        output_type: OutputType = OutputType.GENERAL,
        quality_threshold: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """
        Assess output quality across multiple dimensions
        
        Args:
            output: The output to assess
            requirements: Original requirements/task description
            output_type: Type of output (affects dimension weights)
            quality_threshold: Minimum acceptable quality (uses config default if None)
            context: Additional context for assessment
        
        Returns:
            QualityAssessment with scores and detailed feedback
        """
        if quality_threshold is None:
            quality_threshold = self.config.QUALITY_THRESHOLD
        
        logger.info(f"📊 Assessing output quality (type: {output_type.value}, "
                   f"threshold: {quality_threshold})")
        
        # Choose assessment method
        if self.use_llm:
            try:
                assessment = await self._assess_with_llm(
                    output, requirements, output_type, quality_threshold, context
                )
                assessment.assessment_method = "llm"
            except Exception as e:
                logger.warning(f"LLM assessment failed: {e}, falling back to heuristics")
                assessment = self._assess_with_heuristics(
                    output, requirements, output_type, quality_threshold, context
                )
                assessment.assessment_method = "heuristic_fallback"
        else:
            assessment = self._assess_with_heuristics(
                output, requirements, output_type, quality_threshold, context
            )
            assessment.assessment_method = "heuristic"
        
        # Log results
        logger.info(f"✅ Quality Assessment Complete:")
        logger.info(f"   Overall Score: {assessment.overall_score:.2%}")
        logger.info(f"   Passes Threshold: {'✅' if assessment.passes_threshold else '❌'}")
        logger.info(f"   Method: {assessment.assessment_method}")
        
        for name, dim in assessment.dimensions.items():
            logger.debug(f"   {name.capitalize()}: {dim.score:.2%} (weight: {dim.weight:.0%})")
        
        return assessment
    
    async def _assess_with_llm(
        self,
        output: str,
        requirements: str,
        output_type: OutputType,
        quality_threshold: float,
        context: Optional[Dict[str, Any]]
    ) -> QualityAssessment:
        """LLM-driven intelligent assessment"""
        
        # Build assessment prompt
        prompt = self._build_llm_assessment_prompt(
            output, requirements, output_type, quality_threshold, context
        )
        
        # Build system prompt via ContextService (PRD-80 US-019)
        from modules.context import ContextService, ContextMode

        ctx_result = await ContextService().build_context(
            mode=ContextMode.ORCHESTRATOR_STAGE,
            agent=SimpleNamespace(
                id=None,
                name="Quality Assessor",
                role="Expert output quality assessor. Always return valid JSON.",
                description=None,
            ),
            workspace_id="system",
            task_description=requirements,
        )

        # Get LLM assessment - support multiple LLM client interfaces
        if hasattr(self.llm_client, 'generate_response'):
            # LLMManager interface: generate_response(messages)
            messages = [
                {"role": "system", "content": ctx_result.system_prompt},
                {"role": "user", "content": prompt},
            ]
            llm_response = await self.llm_client.generate_response(messages)
            # Extract text from LLMResponse object
            response = getattr(llm_response, 'content', None) or getattr(llm_response, 'text', None) or str(llm_response)
        elif hasattr(self.llm_client, 'generate'):
            response = await self.llm_client.generate(prompt)
        else:
            # Synchronous fallback
            response = self.llm_client(prompt) if callable(self.llm_client) else None
        
        if not response:
            raise Exception("LLM returned no response")
        
        # Parse LLM response
        assessment = self._parse_llm_response(
            response, output_type, quality_threshold
        )
        
        # Validate with heuristics (hybrid approach)
        heuristic_scores = self._calculate_heuristic_scores(
            output, requirements, output_type
        )
        
        # Adjust LLM scores if they deviate significantly from heuristics
        for dim_name, heuristic_score in heuristic_scores.items():
            if dim_name in assessment.dimensions:
                llm_score = assessment.dimensions[dim_name].score
                deviation = abs(llm_score - heuristic_score)
                
                # If deviation > 0.3, blend scores
                if deviation > 0.3:
                    blended = (llm_score * 0.7) + (heuristic_score * 0.3)
                    assessment.dimensions[dim_name].score = blended
                    logger.debug(f"Blended {dim_name}: LLM={llm_score:.2f}, "
                               f"Heuristic={heuristic_score:.2f}, "
                               f"Final={blended:.2f}")
        
        # Recalculate overall score
        assessment.overall_score = self._calculate_overall_score(
            assessment.dimensions
        )
        assessment.passes_threshold = assessment.overall_score >= quality_threshold
        
        return assessment
    
    def _build_llm_assessment_prompt(
        self,
        output: str,
        requirements: str,
        output_type: OutputType,
        quality_threshold: float,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM quality assessment"""
        
        weights = self.dimension_weights[output_type]
        
        prompt = f"""You are an expert quality assessor evaluating the quality of an AI-generated output.

**OUTPUT TYPE:** {output_type.value}

**ORIGINAL REQUIREMENTS:**
{requirements}

**OUTPUT TO ASSESS:**
{output[:3000]}{'...' if len(output) > 3000 else ''}

**YOUR TASK:**
Assess the output quality across these 5 dimensions (score each 0.0-1.0):

1. **Completeness** (weight: {weights['completeness']:.0%})
   - Are all requirements addressed?
   - Are there gaps or missing sections?
   - Is the scope of the task fully covered?

2. **Coherence** (weight: {weights['coherence']:.0%})
   - Is there logical flow and structure?
   - Do sections connect well?
   - Is the organization clear?

3. **Accuracy** (weight: {weights['accuracy']:.0%})
   - Is information correct and verifiable?
   - Are claims evidence-based?
   - Are there errors or inconsistencies?

4. **Professionalism** (weight: {weights['professionalism']:.0%})
   - Is formatting clean and appropriate?
   - Is the tone professional?
   - Is presentation polished?

5. **Clarity** (weight: {weights['clarity']:.0%})
   - Is it easy to understand?
   - Is language clear and concise?
   - Are explanations adequate?

**ASSESSMENT CRITERIA:**
- Excellent (0.9-1.0): Exceeds expectations, analyst-grade quality
- Good (0.7-0.89): Meets expectations with minor improvements possible
- Acceptable (0.5-0.69): Meets minimum requirements but needs improvement
- Poor (0.3-0.49): Significant issues, major revision needed
- Unacceptable (0.0-0.29): Does not meet requirements, complete rework needed

**QUALITY THRESHOLD:** {quality_threshold:.0%}

**OUTPUT FORMAT:**
Return your assessment as JSON:
{{
    "completeness": {{
        "score": 0.0-1.0,
        "feedback": "Brief explanation",
        "suggestions": ["specific improvement 1", "specific improvement 2"]
    }},
    "coherence": {{
        "score": 0.0-1.0,
        "feedback": "Brief explanation",
        "suggestions": ["specific improvement 1"]
    }},
    "accuracy": {{
        "score": 0.0-1.0,
        "feedback": "Brief explanation",
        "suggestions": []
    }},
    "professionalism": {{
        "score": 0.0-1.0,
        "feedback": "Brief explanation",
        "suggestions": []
    }},
    "clarity": {{
        "score": 0.0-1.0,
        "feedback": "Brief explanation",
        "suggestions": []
    }},
    "strengths": ["key strength 1", "key strength 2"],
    "weaknesses": ["key weakness 1", "key weakness 2"],
    "overall_feedback": "Summary of quality assessment"
}}

Be specific in your feedback and suggestions. Focus on actionable improvements."""

        return prompt
    
    def _parse_llm_response(
        self,
        response: Any,
        output_type: OutputType,
        quality_threshold: float
    ) -> QualityAssessment:
        """Parse LLM assessment response"""
        
        # Extract response text
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Try to parse JSON
        import json
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Create basic assessment
            data = {
                "completeness": {"score": 0.7, "feedback": "Auto-generated", "suggestions": []},
                "coherence": {"score": 0.7, "feedback": "Auto-generated", "suggestions": []},
                "accuracy": {"score": 0.7, "feedback": "Auto-generated", "suggestions": []},
                "professionalism": {"score": 0.7, "feedback": "Auto-generated", "suggestions": []},
                "clarity": {"score": 0.7, "feedback": "Auto-generated", "suggestions": []},
                "strengths": [],
                "weaknesses": [],
                "overall_feedback": response_text[:200]
            }
        
        # Get dimension weights
        weights = self.dimension_weights[output_type]
        
        # Build dimension objects
        dimensions = {}
        for dim_name in ['completeness', 'coherence', 'accuracy', 'professionalism', 'clarity']:
            dim_data = data.get(dim_name, {})
            dimensions[dim_name] = QualityDimension(
                name=dim_name,
                score=float(dim_data.get('score', 0.7)),
                weight=weights[dim_name],
                feedback=dim_data.get('feedback', ''),
                suggestions=dim_data.get('suggestions', [])
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimensions)
        
        # Build assessment
        assessment = QualityAssessment(
            overall_score=overall_score,
            dimensions=dimensions,
            passes_threshold=overall_score >= quality_threshold,
            threshold_used=quality_threshold,
            output_type=output_type,
            assessment_method="llm",
            strengths=data.get('strengths', []),
            weaknesses=data.get('weaknesses', []),
            improvement_suggestions=self._extract_all_suggestions(dimensions),
            confidence=0.85,
            tokens_used=getattr(response, 'tokens_used', 0)
        )
        
        return assessment
    
    def _assess_with_heuristics(
        self,
        output: str,
        requirements: str,
        output_type: OutputType,
        quality_threshold: float,
        context: Optional[Dict[str, Any]]
    ) -> QualityAssessment:
        """Heuristic-based quality assessment (fast, rule-based)"""
        
        # Calculate heuristic scores
        heuristic_scores = self._calculate_heuristic_scores(
            output, requirements, output_type
        )
        
        # Get dimension weights
        weights = self.dimension_weights[output_type]
        
        # Build dimensions
        dimensions = {}
        for dim_name, score in heuristic_scores.items():
            feedback, suggestions = self._generate_heuristic_feedback(
                dim_name, score, output, requirements
            )
            
            dimensions[dim_name] = QualityDimension(
                name=dim_name,
                score=score,
                weight=weights[dim_name],
                feedback=feedback,
                suggestions=suggestions
            )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimensions)
        
        # Generate strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(dimensions)
        
        # Build assessment
        assessment = QualityAssessment(
            overall_score=overall_score,
            dimensions=dimensions,
            passes_threshold=overall_score >= quality_threshold,
            threshold_used=quality_threshold,
            output_type=output_type,
            assessment_method="heuristic",
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=self._extract_all_suggestions(dimensions),
            confidence=0.70  # Lower confidence for heuristics
        )
        
        return assessment
    
    def _calculate_heuristic_scores(
        self,
        output: str,
        requirements: str,
        output_type: OutputType
    ) -> Dict[str, float]:
        """Calculate quality scores using heuristics"""
        
        scores = {}
        
        # 1. Completeness score
        scores['completeness'] = self._heuristic_completeness(
            output, requirements, output_type
        )
        
        # 2. Coherence score
        scores['coherence'] = self._heuristic_coherence(
            output, output_type
        )
        
        # 3. Accuracy score
        scores['accuracy'] = self._heuristic_accuracy(
            output, output_type
        )
        
        # 4. Professionalism score
        scores['professionalism'] = self._heuristic_professionalism(
            output, output_type
        )
        
        # 5. Clarity score
        scores['clarity'] = self._heuristic_clarity(
            output, output_type
        )
        
        return scores
    
    def _heuristic_completeness(
        self,
        output: str,
        requirements: str,
        output_type: OutputType
    ) -> float:
        """Heuristic completeness assessment"""
        
        score = 0.5  # Base score
        
        # Check length (longer = more likely complete)
        if len(output) > 2000:
            score += 0.2
        elif len(output) > 1000:
            score += 0.1
        elif len(output) < 200:
            score -= 0.2
        
        # Check for forbidden patterns (incomplete output)
        forbidden_patterns = [
            r'to be continued',
            r'\.\.\.(?:\s*$|\s*\n)',  # Trailing ellipsis
            r'please let me know',
            r'feel free to',
            r'unfortunately.*without access',
            r'I will write',
            r'I have started'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                score -= 0.3
                break
        
        # Check for key requirement keywords
        req_words = set(re.findall(r'\b\w{4,}\b', requirements.lower()))
        output_words = set(re.findall(r'\b\w{4,}\b', output.lower()))
        
        if req_words:
            overlap = len(req_words.intersection(output_words)) / len(req_words)
            score += overlap * 0.3
        
        # Check for structural completeness (sections, headers)
        if output_type in [OutputType.REPORT, OutputType.DOCUMENTATION]:
            header_count = len(re.findall(r'^#{1,3}\s+', output, re.MULTILINE))
            if header_count >= 3:
                score += 0.1
            elif header_count >= 5:
                score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def _heuristic_coherence(
        self,
        output: str,
        output_type: OutputType
    ) -> float:
        """Heuristic coherence assessment"""
        
        score = 0.6  # Base score
        
        # Check for structure (headers, sections)
        if output_type in [OutputType.REPORT, OutputType.DOCUMENTATION]:
            has_headers = bool(re.search(r'^#{1,3}\s+', output, re.MULTILINE))
            if has_headers:
                score += 0.2
        
        # Check for paragraphs (good structure)
        paragraphs = output.split('\n\n')
        if len(paragraphs) >= 3:
            score += 0.1
        
        # Check for transition words (indicates flow)
        transition_words = [
            'however', 'therefore', 'additionally', 'furthermore',
            'consequently', 'moreover', 'nevertheless', 'thus'
        ]
        transition_count = sum(
            1 for word in transition_words
            if word in output.lower()
        )
        if transition_count >= 2:
            score += 0.1
        
        # Penalize very long paragraphs (poor structure)
        avg_paragraph_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        if avg_paragraph_length > 1000:
            score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _heuristic_accuracy(
        self,
        output: str,
        output_type: OutputType
    ) -> float:
        """Heuristic accuracy assessment"""
        
        score = 0.7  # Base score (assume accurate unless proven otherwise)
        
        # Check for code snippets (evidence of specificity)
        if output_type in [OutputType.CODE, OutputType.REPORT, OutputType.DOCUMENTATION]:
            code_blocks = len(re.findall(r'```[\s\S]*?```', output))
            if code_blocks > 0:
                score += 0.1
            if code_blocks > 2:
                score += 0.1
        
        # Check for specific references (file paths, line numbers)
        has_file_refs = bool(re.search(r'\w+\.(py|js|ts|java|go|rs):\d+', output))
        if has_file_refs:
            score += 0.1
        
        # Check for weak language (reduces confidence in accuracy)
        weak_patterns = [
            r'\bappears?\b', r'\bseems?\b', r'\bmight\b',
            r'\bpossibly\b', r'\bmaybe\b', r'\bperhaps\b'
        ]
        weak_count = sum(
            len(re.findall(pattern, output, re.IGNORECASE))
            for pattern in weak_patterns
        )
        if weak_count > 5:
            score -= 0.2
        elif weak_count > 10:
            score -= 0.3
        
        return min(1.0, max(0.0, score))
    
    def _heuristic_professionalism(
        self,
        output: str,
        output_type: OutputType
    ) -> float:
        """Heuristic professionalism assessment"""
        
        score = 0.7  # Base score
        
        # Check for markdown formatting
        has_formatting = bool(
            re.search(r'[*_`#]', output) or
            re.search(r'^\s*[-*]\s+', output, re.MULTILINE)
        )
        if has_formatting:
            score += 0.1
        
        # Penalize casual language
        casual_patterns = [
            r'\bokay\b', r'\byeah\b', r'\bnope\b',
            r'\bgonna\b', r'\bwanna\b'
        ]
        casual_count = sum(
            len(re.findall(pattern, output, re.IGNORECASE))
            for pattern in casual_patterns
        )
        if casual_count > 0:
            score -= 0.1
        
        # Penalize meta-commentary
        meta_patterns = [
            r'please let me know',
            r'feel free to',
            r'hope this helps',
            r'let me know if you',
            r'is there anything else'
        ]
        has_meta = any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in meta_patterns
        )
        if has_meta:
            score -= 0.2
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+\s+', output)
        capitalized = sum(
            1 for s in sentences
            if s and s[0].isupper()
        )
        if len(sentences) > 0:
            cap_ratio = capitalized / len(sentences)
            if cap_ratio < 0.7:
                score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _heuristic_clarity(
        self,
        output: str,
        output_type: OutputType
    ) -> float:
        """Heuristic clarity assessment"""
        
        score = 0.7  # Base score
        
        # Check average sentence length (too long = less clear)
        sentences = re.split(r'[.!?]+', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 15:  # Concise
                score += 0.1
            elif avg_sentence_length > 30:  # Too long
                score -= 0.2
        
        # Check for jargon overload (too many long words)
        words = re.findall(r'\b\w+\b', output)
        if words:
            long_words = [w for w in words if len(w) > 12]
            long_word_ratio = len(long_words) / len(words)
            if long_word_ratio > 0.1:
                score -= 0.1
        
        # Check for examples (increases clarity)
        example_indicators = [
            r'for example', r'e\.g\.', r'such as',
            r'for instance', r'like'
        ]
        has_examples = any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in example_indicators
        )
        if has_examples:
            score += 0.1
        
        # Check for code comments (for code outputs)
        if output_type == OutputType.CODE:
            has_comments = bool(re.search(r'#.*|//.*|/\*[\s\S]*?\*/', output))
            if has_comments:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _generate_heuristic_feedback(
        self,
        dimension: str,
        score: float,
        output: str,
        requirements: str
    ) -> Tuple[str, List[str]]:
        """Generate feedback and suggestions for a dimension"""
        
        feedback_templates = {
            'completeness': {
                'high': "Output comprehensively addresses requirements",
                'medium': "Output covers most requirements with some gaps",
                'low': "Output has significant gaps in requirement coverage"
            },
            'coherence': {
                'high': "Output has excellent logical flow and structure",
                'medium': "Output has adequate structure with some organizational issues",
                'low': "Output lacks clear structure and logical flow"
            },
            'accuracy': {
                'high': "Output appears accurate with specific evidence",
                'medium': "Output is generally accurate but could be more specific",
                'low': "Output has questionable accuracy or lacks evidence"
            },
            'professionalism': {
                'high': "Output is professionally formatted and presented",
                'medium': "Output is adequate but could improve presentation",
                'low': "Output lacks professional formatting and tone"
            },
            'clarity': {
                'high': "Output is clear and easy to understand",
                'medium': "Output is understandable but could be clearer",
                'low': "Output is difficult to understand or confusing"
            }
        }
        
        suggestion_templates = {
            'completeness': {
                'low': [
                    "Ensure all aspects of the requirements are addressed",
                    "Add missing sections or information",
                    "Expand on incomplete areas"
                ]
            },
            'coherence': {
                'low': [
                    "Add clear section headers and organization",
                    "Improve transitions between sections",
                    "Ensure logical flow of information"
                ]
            },
            'accuracy': {
                'low': [
                    "Add specific evidence and examples",
                    "Include code snippets or data",
                    "Reduce vague or uncertain language"
                ]
            },
            'professionalism': {
                'low': [
                    "Improve formatting with markdown",
                    "Remove casual or meta-commentary",
                    "Ensure professional tone throughout"
                ]
            },
            'clarity': {
                'low': [
                    "Break down complex sentences",
                    "Add examples and illustrations",
                    "Improve paragraph structure"
                ]
            }
        }
        
        # Determine level
        if score >= 0.8:
            level = 'high'
        elif score >= 0.6:
            level = 'medium'
        else:
            level = 'low'
        
        feedback = feedback_templates[dimension][level]
        suggestions = suggestion_templates[dimension].get(level, [])
        
        return feedback, suggestions
    
    def _calculate_overall_score(
        self,
        dimensions: Dict[str, QualityDimension]
    ) -> float:
        """Calculate weighted overall score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for dim in dimensions.values():
            total_score += dim.score * dim.weight
            total_weight += dim.weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    def _extract_all_suggestions(
        self,
        dimensions: Dict[str, QualityDimension]
    ) -> List[str]:
        """Extract all suggestions from dimensions"""
        
        all_suggestions = []
        for dim in dimensions.values():
            all_suggestions.extend(dim.suggestions)
        
        return all_suggestions
    
    def _identify_strengths_weaknesses(
        self,
        dimensions: Dict[str, QualityDimension]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses from dimensions"""
        
        strengths = []
        weaknesses = []
        
        for name, dim in dimensions.items():
            if dim.score >= 0.8:
                strengths.append(f"Strong {name}: {dim.feedback}")
            elif dim.score < 0.6:
                weaknesses.append(f"Weak {name}: {dim.feedback}")
        
        return strengths, weaknesses
    
    def suggest_improvements(
        self,
        assessment: QualityAssessment,
        output: str
    ) -> Dict[str, Any]:
        """
        Generate detailed improvement suggestions based on assessment
        
        Args:
            assessment: Quality assessment result
            output: Original output
        
        Returns:
            Dictionary with improvement suggestions and priorities
        """
        improvements = {
            "priority": [],
            "recommended": [],
            "optional": []
        }
        
        # Categorize improvements by dimension score
        for name, dim in assessment.dimensions.items():
            if dim.score < 0.5:
                # Critical improvements
                improvements["priority"].extend([
                    f"{name.capitalize()}: {sug}" for sug in dim.suggestions
                ])
            elif dim.score < 0.7:
                # Recommended improvements
                improvements["recommended"].extend([
                    f"{name.capitalize()}: {sug}" for sug in dim.suggestions
                ])
            elif dim.suggestions:
                # Optional improvements
                improvements["optional"].extend([
                    f"{name.capitalize()}: {sug}" for sug in dim.suggestions
                ])
        
        return improvements


# Export
__all__ = [
    'OutputQualityAssessor',
    'QualityAssessment',
    'QualityDimension',
    'OutputType'
]
