
"""
Tool Output Processing Module
Advanced output validation, transformation, and integration with semantic alignment
"""
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import re
import logging
import hashlib
from collections import defaultdict, deque
import numpy as np

# Try to import advanced ML libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class OutputType(Enum):
    """Types of tool outputs"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    DATABASE_RESULT = "database_result"
    API_RESPONSE = "api_response"
    ERROR = "error"

class ValidationStatus(Enum):
    """Output validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"
    FAILED = "failed"

@dataclass
class OutputSchema:
    """Schema definition for expected output"""
    output_type: OutputType
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    validation_rules: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    semantic_requirements: Optional[str] = None
    quality_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of output processing"""
    original_output: Any
    processed_output: Any
    validation_status: ValidationStatus
    quality_scores: Dict[str, float]
    transformations_applied: List[str]
    semantic_alignment: float
    integration_metadata: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    accuracy: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    consistency: float = 0.0
    semantic_coherence: float = 0.0
    format_compliance: float = 0.0
    
    def overall_quality(self) -> float:
        """Calculate overall quality score"""
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.20,
            "consistency": 0.15,
            "semantic_coherence": 0.10,
            "format_compliance": 0.10
        }
        
        return sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )

class OutputProcessor:
    """Advanced output processing with validation and semantic alignment"""
    
    def __init__(self):
        # Initialize semantic model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.semantic_enabled = True
                logger.info("Semantic model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None
                self.semantic_enabled = False
        else:
            self.semantic_model = None
            self.semantic_enabled = False
            logger.warning("Sentence transformers not available - semantic features disabled")
        
        # Processing history and statistics
        self.processing_history: deque = deque(maxlen=10000)
        self.quality_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Registered processors for different output types
        self.type_processors: Dict[OutputType, Callable] = {
            OutputType.TEXT: self._process_text_output,
            OutputType.JSON: self._process_json_output,
            OutputType.API_RESPONSE: self._process_api_response,
            OutputType.DATABASE_RESULT: self._process_database_result,
            OutputType.ERROR: self._process_error_output
        }
        
        # Output transformers
        self.transformers: Dict[str, Callable[[Any], Any]] = {}
        
        # Validation cache
        self.validation_cache: Dict[str, ProcessingResult] = {}
        
        logger.info("Initialized OutputProcessor")
    
    async def process_output(self, tool_output: Any, context: Dict[str, Any],
                           expected_schema: Optional[OutputSchema] = None,
                           quality_thresholds: Optional[Dict[str, float]] = None) -> ProcessingResult:
        """Process and validate tool output with semantic alignment"""
        try:
            start_time = datetime.utcnow()
            
            # Detect output type
            output_type = await self._detect_output_type(tool_output)
            
            # Initialize processing result
            result = ProcessingResult(
                original_output=tool_output,
                processed_output=tool_output,
                validation_status=ValidationStatus.PENDING,
                quality_scores={},
                transformations_applied=[],
                semantic_alignment=0.0,
                integration_metadata={},
                warnings=[],
                errors=[]
            )
            
            # Check cache
            cache_key = self._generate_cache_key(tool_output, context)
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                logger.debug("Using cached processing result")
                return cached_result
            
            # Apply type-specific processing
            if output_type in self.type_processors:
                processed_output = await self.type_processors[output_type](
                    tool_output, context, expected_schema
                )
                result.processed_output = processed_output
                result.transformations_applied.append(f"type_processing_{output_type.value}")
            
            # Validate against schema
            if expected_schema:
                validation_result = await self._validate_against_schema(
                    result.processed_output, expected_schema
                )
                result.validation_status = validation_result["status"]
                result.errors.extend(validation_result.get("errors", []))
                result.warnings.extend(validation_result.get("warnings", []))
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(
                result.processed_output, context, expected_schema
            )
            result.quality_scores = {
                "accuracy": quality_metrics.accuracy,
                "completeness": quality_metrics.completeness,
                "relevance": quality_metrics.relevance,
                "consistency": quality_metrics.consistency,
                "semantic_coherence": quality_metrics.semantic_coherence,
                "format_compliance": quality_metrics.format_compliance,
                "overall_quality": quality_metrics.overall_quality()
            }
            
            # Calculate semantic alignment
            if self.semantic_enabled and context.get("expected_context"):
                result.semantic_alignment = await self._calculate_semantic_alignment(
                    result.processed_output, context["expected_context"]
                )
            
            # Apply quality thresholds
            if quality_thresholds:
                await self._apply_quality_thresholds(result, quality_thresholds)
            
            # Generate integration metadata
            result.integration_metadata = await self._generate_integration_metadata(
                result.processed_output, context, output_type
            )
            
            # Set final validation status
            if not result.errors and result.quality_scores.get("overall_quality", 0) > 0.5:
                if result.warnings:
                    result.validation_status = ValidationStatus.WARNING
                else:
                    result.validation_status = ValidationStatus.VALID
            else:
                result.validation_status = ValidationStatus.INVALID
            
            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time = (end_time - start_time).total_seconds()
            
            # Cache result
            self.validation_cache[cache_key] = result
            
            # Update history and trends
            await self._update_processing_history(result, output_type)
            
            logger.debug(f"Output processing completed: {result.validation_status.value}, "
                        f"quality: {result.quality_scores.get('overall_quality', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing output: {e}")
            return ProcessingResult(
                original_output=tool_output,
                processed_output=tool_output,
                validation_status=ValidationStatus.FAILED,
                quality_scores={},
                transformations_applied=[],
                semantic_alignment=0.0,
                integration_metadata={},
                warnings=[],
                errors=[f"Processing failed: {str(e)}"],
                processing_time=0.0
            )
    
    async def _detect_output_type(self, output: Any) -> OutputType:
        """Detect the type of output"""
        try:
            if isinstance(output, str):
                # Check if it's JSON
                try:
                    json.loads(output)
                    return OutputType.JSON
                except:
                    pass
                
                # Check for common error patterns
                error_patterns = ["error", "exception", "failed", "traceback"]
                if any(pattern in output.lower() for pattern in error_patterns):
                    return OutputType.ERROR
                
                # Check for file paths
                if output.startswith(('/', './', '../')) or '.' in output:
                    return OutputType.FILE
                
                return OutputType.TEXT
            
            elif isinstance(output, dict):
                # Check for API response patterns
                if any(key in output for key in ["status", "data", "response", "result"]):
                    return OutputType.API_RESPONSE
                
                # Check for database result patterns
                if any(key in output for key in ["rows", "count", "query", "records"]):
                    return OutputType.DATABASE_RESULT
                
                return OutputType.JSON
            
            elif isinstance(output, list):
                return OutputType.JSON
            
            elif isinstance(output, bytes):
                return OutputType.BINARY
            
            else:
                return OutputType.TEXT  # Default fallback
            
        except Exception as e:
            logger.warning(f"Error detecting output type: {e}")
            return OutputType.TEXT
    
    async def _process_text_output(self, output: str, context: Dict[str, Any], 
                                 schema: Optional[OutputSchema] = None) -> str:
        """Process text output"""
        try:
            processed = output
            
            # Clean whitespace
            processed = ' '.join(processed.split())
            
            # Apply text transformations based on context
            if context.get("format") == "lowercase":
                processed = processed.lower()
            elif context.get("format") == "uppercase":
                processed = processed.upper()
            
            # Remove sensitive information if specified
            if context.get("sanitize"):
                processed = await self._sanitize_text(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing text output: {e}")
            return output
    
    async def _process_json_output(self, output: Union[str, dict, list], 
                                 context: Dict[str, Any],
                                 schema: Optional[OutputSchema] = None) -> Dict[str, Any]:
        """Process JSON output"""
        try:
            # Parse JSON if it's a string
            if isinstance(output, str):
                parsed = json.loads(output)
            else:
                parsed = output
            
            # Normalize structure
            if isinstance(parsed, list):
                processed = {"items": parsed, "count": len(parsed)}
            elif isinstance(parsed, dict):
                processed = parsed
            else:
                processed = {"value": parsed}
            
            # Apply field transformations
            if schema and schema.field_types:
                processed = await self._apply_type_conversions(processed, schema.field_types)
            
            # Add metadata
            processed["_metadata"] = {
                "processed_at": datetime.utcnow().isoformat(),
                "original_type": type(output).__name__
            }
            
            return processed
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON output: {e}")
            return {"error": "Invalid JSON", "original": str(output)}
        except Exception as e:
            logger.error(f"Error processing JSON output: {e}")
            return {"error": str(e), "original": output}
    
    async def _process_api_response(self, output: dict, context: Dict[str, Any],
                                  schema: Optional[OutputSchema] = None) -> Dict[str, Any]:
        """Process API response output"""
        try:
            processed = output.copy()
            
            # Standardize response format
            if "status" not in processed and "data" in processed:
                processed["status"] = "success"
            
            # Extract and validate status codes
            if "status_code" in processed:
                status_code = processed["status_code"]
                if isinstance(status_code, int):
                    if 200 <= status_code < 300:
                        processed["status"] = "success"
                    elif 400 <= status_code < 500:
                        processed["status"] = "client_error"
                    elif 500 <= status_code < 600:
                        processed["status"] = "server_error"
            
            # Extract data payload
            if "data" in processed:
                processed["payload"] = processed["data"]
            elif "result" in processed:
                processed["payload"] = processed["result"]
            
            # Add response metadata
            processed["_metadata"] = {
                "processed_at": datetime.utcnow().isoformat(),
                "response_type": "api_response"
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing API response: {e}")
            return output
    
    async def _process_database_result(self, output: dict, context: Dict[str, Any],
                                     schema: Optional[OutputSchema] = None) -> Dict[str, Any]:
        """Process database result output"""
        try:
            processed = output.copy()
            
            # Standardize database result format
            if "rows" in processed:
                processed["data"] = processed["rows"]
                if "count" not in processed:
                    processed["count"] = len(processed["rows"]) if isinstance(processed["rows"], list) else 0
            
            # Add query metadata if available
            if "query" in context:
                processed["_metadata"] = {
                    "query": context["query"],
                    "processed_at": datetime.utcnow().isoformat(),
                    "result_type": "database_query"
                }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing database result: {e}")
            return output
    
    async def _process_error_output(self, output: Any, context: Dict[str, Any],
                                  schema: Optional[OutputSchema] = None) -> Dict[str, Any]:
        """Process error output"""
        try:
            processed = {
                "error": True,
                "message": str(output),
                "timestamp": datetime.utcnow().isoformat(),
                "context": context.get("tool_name", "unknown")
            }
            
            # Extract error details if available
            if isinstance(output, dict) and "error" in output:
                processed.update(output)
            elif isinstance(output, str):
                # Try to extract structured error information
                if "traceback" in output.lower():
                    lines = output.split('\n')
                    processed["traceback"] = lines
                    # Extract error type and message
                    for line in lines:
                        if "Error:" in line or "Exception:" in line:
                            processed["error_type"] = line.split(':')[0].strip()
                            processed["message"] = ':'.join(line.split(':')[1:]).strip()
                            break
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing error output: {e}")
            return {"error": True, "message": str(output), "processing_error": str(e)}
    
    async def _validate_against_schema(self, output: Any, schema: OutputSchema) -> Dict[str, Any]:
        """Validate output against expected schema"""
        try:
            result = {
                "status": ValidationStatus.VALID,
                "errors": [],
                "warnings": []
            }
            
            # Type validation
            if isinstance(output, dict):
                # Check required fields
                for field in schema.required_fields:
                    if field not in output:
                        result["errors"].append(f"Missing required field: {field}")
                
                # Check field types
                for field, expected_type in schema.field_types.items():
                    if field in output:
                        if not isinstance(output[field], expected_type):
                            result["warnings"].append(
                                f"Field '{field}' type mismatch: expected {expected_type.__name__}, "
                                f"got {type(output[field]).__name__}"
                            )
                
                # Apply validation rules
                for field, validator in schema.validation_rules.items():
                    if field in output:
                        try:
                            if not validator(output[field]):
                                result["errors"].append(f"Validation failed for field: {field}")
                        except Exception as e:
                            result["warnings"].append(f"Validation error for field '{field}': {e}")
            
            # Set final status
            if result["errors"]:
                result["status"] = ValidationStatus.INVALID
            elif result["warnings"]:
                result["status"] = ValidationStatus.WARNING
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating against schema: {e}")
            return {
                "status": ValidationStatus.FAILED,
                "errors": [f"Schema validation failed: {e}"],
                "warnings": []
            }
    
    async def _calculate_quality_metrics(self, output: Any, context: Dict[str, Any],
                                       schema: Optional[OutputSchema] = None) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        try:
            metrics = QualityMetrics()
            
            # Accuracy: Based on data type correctness and validation results
            if isinstance(output, dict):
                # Check for error indicators
                if output.get("error", False):
                    metrics.accuracy = 0.0
                else:
                    metrics.accuracy = 0.8  # Base score for non-error output
                    
                    # Bonus for structured data
                    if len(output) > 1:
                        metrics.accuracy += 0.1
            else:
                metrics.accuracy = 0.6  # Lower for unstructured output
            
            # Completeness: Based on required fields presence
            if schema and schema.required_fields:
                if isinstance(output, dict):
                    present_fields = sum(1 for field in schema.required_fields if field in output)
                    metrics.completeness = present_fields / len(schema.required_fields)
                else:
                    metrics.completeness = 0.5  # Partial for non-dict output
            else:
                metrics.completeness = 1.0  # No requirements means complete
            
            # Relevance: Based on context matching
            if context.get("expected_output_type"):
                expected_type = context["expected_output_type"]
                actual_type = type(output).__name__.lower()
                if expected_type.lower() in actual_type:
                    metrics.relevance = 0.9
                else:
                    metrics.relevance = 0.6
            else:
                metrics.relevance = 0.7  # Default
            
            # Consistency: Based on data structure consistency
            if isinstance(output, dict):
                # Check for consistent data types in lists
                consistency_score = 0.8
                for key, value in output.items():
                    if isinstance(value, list) and value:
                        first_type = type(value[0])
                        if not all(isinstance(item, first_type) for item in value):
                            consistency_score -= 0.1
                metrics.consistency = max(0.0, consistency_score)
            else:
                metrics.consistency = 0.7
            
            # Format compliance: Based on expected schema
            if schema:
                # Check if output follows expected format
                format_score = 0.5  # Base score
                if schema.output_type.value in str(type(output)).lower():
                    format_score += 0.3
                if isinstance(output, dict) and schema.required_fields:
                    field_compliance = sum(1 for field in schema.required_fields if field in output)
                    field_compliance /= len(schema.required_fields)
                    format_score += field_compliance * 0.2
                metrics.format_compliance = min(1.0, format_score)
            else:
                metrics.format_compliance = 0.8
            
            # Semantic coherence: Requires semantic analysis
            if self.semantic_enabled and isinstance(output, (str, dict)):
                try:
                    text_content = self._extract_text_for_analysis(output)
                    if text_content and len(text_content) > 10:
                        # Simple coherence check based on text length and structure
                        coherence_score = min(1.0, len(text_content.split()) / 50.0)
                        metrics.semantic_coherence = coherence_score
                    else:
                        metrics.semantic_coherence = 0.5
                except Exception:
                    metrics.semantic_coherence = 0.5
            else:
                metrics.semantic_coherence = 0.5
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return QualityMetrics()  # Return default metrics
    
    async def _calculate_semantic_alignment(self, output: Any, expected_context: str) -> float:
        """Calculate semantic alignment between output and expected context"""
        try:
            if not self.semantic_enabled:
                return 0.5  # Default when semantic analysis not available
            
            # Extract text from output
            output_text = self._extract_text_for_analysis(output)
            
            if not output_text or len(output_text.strip()) < 5:
                return 0.0
            
            # Generate embeddings
            output_embedding = self.semantic_model.encode([output_text])
            context_embedding = self.semantic_model.encode([expected_context])
            
            # Calculate cosine similarity
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity(output_embedding, context_embedding)[0][0]
            else:
                # Fallback to numpy
                output_norm = np.linalg.norm(output_embedding[0])
                context_norm = np.linalg.norm(context_embedding[0])
                
                if output_norm == 0 or context_norm == 0:
                    return 0.0
                
                similarity = np.dot(output_embedding[0], context_embedding[0]) / (output_norm * context_norm)
            
            # Convert to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"Error calculating semantic alignment: {e}")
            return 0.0
    
    def _extract_text_for_analysis(self, output: Any) -> str:
        """Extract text content from output for analysis"""
        try:
            if isinstance(output, str):
                return output
            elif isinstance(output, dict):
                # Extract text from common fields
                text_parts = []
                for key, value in output.items():
                    if isinstance(value, str) and not key.startswith('_'):
                        text_parts.append(value)
                return ' '.join(text_parts)
            elif isinstance(output, list):
                text_parts = []
                for item in output[:10]:  # Limit to first 10 items
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        item_text = self._extract_text_for_analysis(item)
                        if item_text:
                            text_parts.append(item_text)
                return ' '.join(text_parts)
            else:
                return str(output)
        except Exception:
            return ""
    
    async def _apply_type_conversions(self, data: dict, type_mapping: Dict[str, type]) -> dict:
        """Apply type conversions to data fields"""
        try:
            converted = data.copy()
            
            for field, target_type in type_mapping.items():
                if field in converted:
                    try:
                        if target_type == int:
                            converted[field] = int(float(converted[field]))  # Handle string numbers
                        elif target_type == float:
                            converted[field] = float(converted[field])
                        elif target_type == str:
                            converted[field] = str(converted[field])
                        elif target_type == bool:
                            if isinstance(converted[field], str):
                                converted[field] = converted[field].lower() in ['true', '1', 'yes']
                            else:
                                converted[field] = bool(converted[field])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Type conversion failed for field '{field}': {e}")
            
            return converted
            
        except Exception as e:
            logger.error(f"Error applying type conversions: {e}")
            return data
    
    async def _sanitize_text(self, text: str) -> str:
        """Remove sensitive information from text"""
        try:
            sanitized = text
            
            # Remove email addresses
            sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', sanitized)
            
            # Remove phone numbers
            sanitized = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', sanitized)
            sanitized = re.sub(r'\b\(\d{3}\)\s?\d{3}-\d{4}\b', '[PHONE]', sanitized)
            
            # Remove credit card-like numbers
            sanitized = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', sanitized)
            
            # Remove IP addresses
            sanitized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', sanitized)
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing text: {e}")
            return text
    
    async def _apply_quality_thresholds(self, result: ProcessingResult, 
                                      thresholds: Dict[str, float]):
        """Apply quality thresholds and update validation status"""
        try:
            for metric, threshold in thresholds.items():
                if metric in result.quality_scores:
                    score = result.quality_scores[metric]
                    if score < threshold:
                        result.warnings.append(
                            f"Quality metric '{metric}' ({score:.3f}) below threshold ({threshold})"
                        )
            
            # Check overall quality threshold
            if "overall_quality" in thresholds:
                overall_score = result.quality_scores.get("overall_quality", 0.0)
                if overall_score < thresholds["overall_quality"]:
                    result.validation_status = ValidationStatus.INVALID
                    result.errors.append(
                        f"Overall quality ({overall_score:.3f}) below required threshold "
                        f"({thresholds['overall_quality']})"
                    )
        
        except Exception as e:
            logger.error(f"Error applying quality thresholds: {e}")
    
    async def _generate_integration_metadata(self, output: Any, context: Dict[str, Any], 
                                           output_type: OutputType) -> Dict[str, Any]:
        """Generate metadata for context integration"""
        try:
            metadata = {
                "output_type": output_type.value,
                "size_estimate": len(str(output)),
                "structure_complexity": self._calculate_structure_complexity(output),
                "integration_recommendations": [],
                "context_compatibility": {}
            }
            
            # Add type-specific recommendations
            if output_type == OutputType.JSON and isinstance(output, dict):
                if "data" in output:
                    metadata["integration_recommendations"].append("Extract 'data' field for downstream use")
                if "error" in output and output.get("error"):
                    metadata["integration_recommendations"].append("Handle error condition in workflow")
            
            # Context compatibility analysis
            if "expected_format" in context:
                expected = context["expected_format"]
                if expected == "json" and isinstance(output, dict):
                    metadata["context_compatibility"]["format"] = "high"
                elif expected == "text" and isinstance(output, str):
                    metadata["context_compatibility"]["format"] = "high"
                else:
                    metadata["context_compatibility"]["format"] = "medium"
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating integration metadata: {e}")
            return {}
    
    def _calculate_structure_complexity(self, output: Any) -> float:
        """Calculate structural complexity of output"""
        try:
            if isinstance(output, dict):
                complexity = len(output) * 0.1
                for value in output.values():
                    if isinstance(value, (dict, list)):
                        complexity += self._calculate_structure_complexity(value)
                return min(1.0, complexity)
            elif isinstance(output, list):
                complexity = len(output) * 0.05
                for item in output[:10]:  # Limit analysis to first 10 items
                    if isinstance(item, (dict, list)):
                        complexity += self._calculate_structure_complexity(item)
                return min(1.0, complexity)
            else:
                return 0.1  # Simple scalar value
        except Exception:
            return 0.5
    
    def _generate_cache_key(self, output: Any, context: Dict[str, Any]) -> str:
        """Generate cache key for processing results"""
        try:
            # Create a hash of the output and relevant context
            output_str = json.dumps(output, default=str, sort_keys=True)
            context_str = json.dumps({
                k: v for k, v in context.items() 
                if k in ["expected_context", "expected_format", "tool_name"]
            }, default=str, sort_keys=True)
            
            combined = f"{output_str}:{context_str}"
            return hashlib.md5(combined.encode()).hexdigest()
            
        except Exception:
            return str(hash(str(output)[:1000]))  # Fallback
    
    async def _update_processing_history(self, result: ProcessingResult, output_type: OutputType):
        """Update processing history and quality trends"""
        try:
            # Add to history
            history_entry = {
                "timestamp": result.timestamp.isoformat(),
                "output_type": output_type.value,
                "validation_status": result.validation_status.value,
                "overall_quality": result.quality_scores.get("overall_quality", 0.0),
                "processing_time": result.processing_time,
                "transformations_count": len(result.transformations_applied)
            }
            
            self.processing_history.append(history_entry)
            
            # Update quality trends
            for metric, score in result.quality_scores.items():
                self.quality_trends[metric].append(score)
                # Keep only recent scores
                if len(self.quality_trends[metric]) > 1000:
                    self.quality_trends[metric] = self.quality_trends[metric][-1000:]
        
        except Exception as e:
            logger.error(f"Error updating processing history: {e}")
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        try:
            if not self.processing_history:
                return {"message": "No processing history available"}
            
            total_processed = len(self.processing_history)
            
            # Status distribution
            status_counts = defaultdict(int)
            type_counts = defaultdict(int)
            
            processing_times = []
            quality_scores = []
            
            for entry in self.processing_history:
                status_counts[entry["validation_status"]] += 1
                type_counts[entry["output_type"]] += 1
                processing_times.append(entry["processing_time"])
                quality_scores.append(entry["overall_quality"])
            
            # Calculate averages
            avg_processing_time = sum(processing_times) / len(processing_times)
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # Quality trends
            recent_quality_trend = []
            if len(quality_scores) >= 20:
                recent = quality_scores[-20:]
                for i in range(10, 20):
                    first_half = sum(recent[:i]) / i
                    second_half = sum(recent[i:]) / (20 - i)
                    recent_quality_trend.append(second_half - first_half)
            
            trend_direction = "stable"
            if recent_quality_trend:
                avg_trend = sum(recent_quality_trend) / len(recent_quality_trend)
                if avg_trend > 0.05:
                    trend_direction = "improving"
                elif avg_trend < -0.05:
                    trend_direction = "declining"
            
            return {
                "total_processed": total_processed,
                "status_distribution": dict(status_counts),
                "type_distribution": dict(type_counts),
                "average_processing_time": avg_processing_time,
                "average_quality_score": avg_quality,
                "quality_trend": trend_direction,
                "cache_size": len(self.validation_cache),
                "semantic_processing_enabled": self.semantic_enabled
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {"error": str(e)}
