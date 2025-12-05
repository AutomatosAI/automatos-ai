"""
Model Registry Service (PRD-15)
================================

Service for managing and querying available LLM models from different providers.
Provides model discovery, recommendations, and cost estimation.

Author: Automatos AI
Date: 2025-10-06
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session
from core.models import LLMModel

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Model information with capabilities and metadata.
    Provides a clean interface for working with model data.
    """
    id: int
    provider: str
    model_id: str
    display_name: str
    model_family: str
    context_window: int
    max_output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: Dict[str, str]
    recommended_for: List[str]
    supports_functions: bool
    supports_vision: bool
    supports_streaming: bool
    status: str
    description: str = ""
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return round(input_cost + output_cost, 6)
    
    def get_capability_score(self, capability: str) -> float:
        """
        Get numeric score for a capability.
        
        Args:
            capability: Capability name (e.g., 'reasoning', 'coding')
            
        Returns:
            Score from 0-1
        """
        if capability not in self.capabilities:
            return 0.0
        
        rating = self.capabilities[capability].lower()
        rating_map = {
            'excellent': 1.0,
            'very good': 0.9,
            'good': 0.7,
            'moderate': 0.5,
            'fair': 0.3,
            'low': 0.1
        }
        
        return rating_map.get(rating, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'provider': self.provider,
            'model_id': self.model_id,
            'display_name': self.display_name,
            'model_family': self.model_family,
            'context_window': self.context_window,
            'max_output_tokens': self.max_output_tokens,
            'input_cost_per_1k': self.input_cost_per_1k,
            'output_cost_per_1k': self.output_cost_per_1k,
            'capabilities': self.capabilities,
            'recommended_for': self.recommended_for,
            'supports_functions': self.supports_functions,
            'supports_vision': self.supports_vision,
            'supports_streaming': self.supports_streaming,
            'status': self.status,
            'description': self.description
        }


class ModelRegistry:
    """
    Registry for managing available LLM models.
    
    Features:
    - Query models by provider, status, or capabilities
    - Get model recommendations based on task type
    - Find best model matching specific requirements
    - Cost estimation
    - Model metadata and capabilities
    
    Example:
        registry = ModelRegistry(db_session)
        models = registry.get_all_models(provider='openai')
        best = registry.find_best_model({
            'task_type': 'code_analysis',
            'max_cost': 0.05,
            'min_context': 8000
        })
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize model registry.
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self._cache: Optional[Dict[str, ModelInfo]] = None
        self.logger = logging.getLogger(__name__)
    
    def get_all_models(self, 
                      provider: Optional[str] = None, 
                      status: str = 'active') -> List[ModelInfo]:
        """
        Get all available models, optionally filtered.
        
        Args:
            provider: Filter by provider (openai, anthropic, huggingface)
            status: Filter by status (active, deprecated, beta)
            
        Returns:
            List of ModelInfo objects
        """
        query = self.db.query(LLMModel).filter(LLMModel.status == status)
        
        if provider:
            query = query.filter(LLMModel.provider == provider)
        
        models = query.order_by(LLMModel.provider, LLMModel.display_name).all()
        result = [self._model_to_info(m) for m in models]
        
        self.logger.info(f"Retrieved {len(result)} models (provider={provider}, status={status})")
        return result
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get specific model by ID.
        
        Args:
            model_id: Unique model identifier (e.g., 'gpt-4', 'claude-3-opus-20240229')
            
        Returns:
            ModelInfo if found, None otherwise
        """
        model = self.db.query(LLMModel).filter(
            LLMModel.model_id == model_id
        ).first()
        
        if model:
            self.logger.debug(f"Found model: {model_id}")
            return self._model_to_info(model)
        
        self.logger.warning(f"Model not found: {model_id}")
        return None
    
    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """
        Get all models for a specific provider.
        
        Args:
            provider: Provider name (openai, anthropic, huggingface)
            
        Returns:
            List of ModelInfo objects
        """
        return self.get_all_models(provider=provider)
    
    def get_recommended_models(self, task_type: str) -> List[ModelInfo]:
        """
        Get models recommended for a specific task type.
        
        Args:
            task_type: Task type (e.g., 'code_analysis', 'creative_writing')
            
        Returns:
            List of recommended ModelInfo objects, sorted by relevance
        """
        models = self.get_all_models()
        
        # Filter models that include this task type in recommendations
        recommended = [
            m for m in models 
            if task_type in m.recommended_for
        ]
        
        # Sort by cost (prefer cheaper models for same recommendation)
        recommended.sort(key=lambda m: (m.input_cost_per_1k + m.output_cost_per_1k) / 2)
        
        self.logger.info(f"Found {len(recommended)} models recommended for '{task_type}'")
        return recommended
    
    def find_best_model(self, requirements: Dict[str, Any]) -> Optional[ModelInfo]:
        """
        Find best model based on requirements.
        
        Uses a scoring algorithm that considers:
        - Task type recommendations
        - Context window requirements
        - Cost constraints
        - Required capabilities
        - Provider preference
        
        Args:
            requirements: Dictionary with:
                - task_type (str): Type of task
                - max_cost (float): Maximum acceptable cost per 1K tokens
                - min_context (int): Minimum context window size
                - required_capabilities (List[str]): Required capabilities
                - prefer_provider (str): Preferred provider
                - require_functions (bool): Requires function calling
                - require_vision (bool): Requires vision support
                
        Returns:
            Best matching ModelInfo or None if no match
            
        Example:
            best = registry.find_best_model({
                'task_type': 'code_analysis',
                'max_cost': 0.05,
                'min_context': 8000,
                'required_capabilities': ['reasoning', 'coding'],
                'prefer_provider': 'openai'
            })
        """
        models = self.get_all_models()
        candidates = []
        
        self.logger.debug(f"Finding best model for requirements: {requirements}")
        
        for model in models:
            # Check hard requirements
            if requirements.get('min_context') and model.context_window < requirements['min_context']:
                continue
            
            if requirements.get('require_functions') and not model.supports_functions:
                continue
            
            if requirements.get('require_vision') and not model.supports_vision:
                continue
            
            # Check task type recommendation
            task_type = requirements.get('task_type')
            if task_type and task_type not in model.recommended_for:
                continue
            
            # Check capabilities
            required_caps = requirements.get('required_capabilities', [])
            has_all_caps = all(
                cap in model.capabilities 
                for cap in required_caps
            )
            if not has_all_caps:
                continue
            
            # Calculate score
            score = 0.0
            
            # Provider preference (strong weight)
            if requirements.get('prefer_provider') and model.provider == requirements['prefer_provider']:
                score += 15
            
            # Task recommendation (strong weight)
            if task_type and task_type in model.recommended_for:
                score += 10
            
            # Cost efficiency (moderate weight)
            avg_cost = (model.input_cost_per_1k + model.output_cost_per_1k) / 2
            max_cost = requirements.get('max_cost', 1.0)
            if avg_cost <= max_cost:
                # Prefer cheaper within budget
                cost_score = 5 * (1 - (avg_cost / max_cost))
                score += cost_score
            
            # Context window (moderate weight)
            if model.context_window >= 100000:
                score += 5
            elif model.context_window >= 32000:
                score += 3
            elif model.context_window >= 16000:
                score += 2
            
            # Capability quality (light weight)
            for cap in required_caps:
                cap_score = model.get_capability_score(cap)
                score += cap_score * 2
            
            # Function calling (light weight)
            if model.supports_functions:
                score += 1
            
            candidates.append((model, score))
        
        if not candidates:
            self.logger.warning(f"No models found matching requirements: {requirements}")
            return None
        
        # Sort by score and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score = candidates[0]
        
        self.logger.info(
            f"Best model: {best_model.model_id} "
            f"(score={best_score:.2f}, provider={best_model.provider})"
        )
        
        return best_model
    
    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about available providers.
        
        Returns:
            Dictionary mapping provider name to stats:
            {
                'openai': {
                    'model_count': 3,
                    'avg_input_cost': 0.01,
                    'avg_output_cost': 0.03,
                    'total_context_capacity': 152577
                },
                ...
            }
        """
        models = self.get_all_models()
        stats = {}
        
        for model in models:
            if model.provider not in stats:
                stats[model.provider] = {
                    'model_count': 0,
                    'total_input_cost': 0,
                    'total_output_cost': 0,
                    'total_context_capacity': 0,
                    'models': []
                }
            
            stats[model.provider]['model_count'] += 1
            stats[model.provider]['total_input_cost'] += model.input_cost_per_1k
            stats[model.provider]['total_output_cost'] += model.output_cost_per_1k
            stats[model.provider]['total_context_capacity'] += model.context_window
            stats[model.provider]['models'].append({
                'model_id': model.model_id,
                'display_name': model.display_name
            })
        
        # Calculate averages
        for provider in stats:
            count = stats[provider]['model_count']
            if count > 0:
                stats[provider]['avg_input_cost'] = stats[provider]['total_input_cost'] / count
                stats[provider]['avg_output_cost'] = stats[provider]['total_output_cost'] / count
        
        return stats
    
    def _model_to_info(self, model: LLMModel) -> ModelInfo:
        """
        Convert database model to ModelInfo dataclass.
        
        Args:
            model: LLMModel database object
            
        Returns:
            ModelInfo dataclass
        """
        return ModelInfo(
            id=model.id,
            provider=model.provider,
            model_id=model.model_id,
            display_name=model.display_name,
            model_family=model.model_family or '',
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            input_cost_per_1k=float(model.input_cost_per_1k_tokens or 0),
            output_cost_per_1k=float(model.output_cost_per_1k_tokens or 0),
            capabilities=model.capabilities or {},
            recommended_for=model.recommended_for or [],
            supports_functions=model.supports_functions or False,
            supports_vision=model.supports_vision or False,
            supports_streaming=model.supports_streaming or True,
            status=model.status or 'active',
            description=model.description or ''
        )
    
    def refresh_cache(self):
        """Refresh the internal cache (if implemented)"""
        self._cache = None
        self.logger.info("Model cache refreshed")


# Convenience function for quick access
def get_model_registry(db_session: Session) -> ModelRegistry:
    """
    Get a ModelRegistry instance.
    
    Args:
        db_session: SQLAlchemy database session
        
    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(db_session)

