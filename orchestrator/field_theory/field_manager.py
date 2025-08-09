
"""
Field Theory Context Manager
============================

Advanced field-based context management for Automatos AI with:
- Scalar and vector field representations
- Influence propagation and gradient calculations
- Dynamic field updates and stability analysis
- Multi-objective field optimization

Mathematical foundation:
- C(x) = Σ w_i * f_i(x) for scalar field modeling
- ∇C(x) for influence propagation
- dC/dt = α * ∇C + β * I(x, y) for dynamic updates
- Stability = min(ΔC_i) for stability analysis
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    cosine_similarity = None
from scipy.optimize import minimize
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Import models
from models import Task, User
from database import get_db

logger = logging.getLogger(__name__)

class FieldType(Enum):
    """Types of context fields"""
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"

@dataclass
class FieldState:
    """Field state representation"""
    field_id: str
    field_type: FieldType
    value: float
    gradient: List[float]
    weights: Dict[str, float]
    stability: float
    timestamp: datetime

@dataclass
class FieldInteraction:
    """Field interaction representation"""
    source_field: str
    target_field: str
    interaction_type: str
    strength: float
    semantic_similarity: float
    timestamp: datetime

class FieldContextManager:
    """
    Advanced field-based context management system
    """
    
    def __init__(self, max_field_size: int = 1000):
        self.contexts = {}  # Dict[str, Dict[str, Any]]
        self.field_states = {}  # Dict[str, FieldState]
        self.field_interactions = []  # List[FieldInteraction]
        
        # Field configuration
        self.max_field_size = max_field_size
        self.field_weights = {
            "task_priority": 0.4,
            "agent_activity": 0.3,
            "context_relevance": 0.3
        }
        
        # Field dynamics parameters
        self.propagation_alpha = 0.1  # Gradient influence factor
        self.interaction_beta = 0.2   # Interaction influence factor
        self.stability_threshold = 0.05
        
        # Embedding model for semantic interactions
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Successfully loaded SentenceTransformer embedding model")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        else:
            logger.info("SentenceTransformers not available, using basic text similarity")
            self.embedding_model = None
        
        # Performance tracking
        self.field_metrics = {
            "fields_created": 0,
            "field_updates": 0,
            "propagation_events": 0,
            "stability_warnings": 0,
            "optimization_runs": 0
        }
        
        logger.info("Field Context Manager initialized")
    
    async def update_field(
        self,
        session_id: str,
        context_data: Dict[str, Any],
        field_type: FieldType = FieldType.SCALAR
    ) -> Dict[str, Any]:
        """
        Update field representation for context data
        
        Args:
            session_id: Session identifier
            context_data: Context data to process
            field_type: Type of field to create/update
            
        Returns:
            Updated context data with field information
        """
        field_start = time.time()
        
        try:
            # Validate input
            if not isinstance(context_data, dict):
                raise ValueError("Invalid context data format")
            
            # Create field ID
            field_id = f"field_{session_id}_{int(time.time())}"
            
            if field_type == FieldType.SCALAR:
                field_result = await self._update_scalar_field(
                    session_id, field_id, context_data
                )
            elif field_type == FieldType.VECTOR:
                field_result = await self._update_vector_field(
                    session_id, field_id, context_data
                )
            else:  # TENSOR
                field_result = await self._update_tensor_field(
                    session_id, field_id, context_data
                )
            
            # Update context with field information
            context_data.update(field_result)
            
            # Store in contexts
            if session_id not in self.contexts:
                self.contexts[session_id] = {}
            
            self.contexts[session_id].update(context_data)
            
            # Update metrics
            self.field_metrics["field_updates"] += 1
            
            logger.info(
                f"Updated {field_type.value} field for session {session_id}: "
                f"value={field_result.get('field_value', 0):.3f}, "
                f"time={time.time() - field_start:.3f}s"
            )
            
            return context_data
            
        except Exception as e:
            logger.error(f"Error updating field for session {session_id}: {str(e)}")
            raise
    
    async def _update_scalar_field(
        self,
        session_id: str,
        field_id: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update scalar field: C(x) = Σ w_i * f_i(x)"""
        
        # Extract field components
        task_priority = context_data.get("task_priority", 0.5)
        agent_activity = len(context_data.get("agents", [])) / 10.0  # Normalize
        context_relevance = len(context_data.get("description", "")) / 100.0  # Normalize
        
        # Additional factors
        importance = context_data.get("importance", 0.5)
        complexity = context_data.get("complexity", 0.5)
        urgency = context_data.get("urgency", 0.5)
        
        # Calculate scalar field value
        field_value = (
            self.field_weights["task_priority"] * task_priority +
            self.field_weights["agent_activity"] * agent_activity +
            self.field_weights["context_relevance"] * context_relevance +
            0.1 * importance +
            0.1 * complexity +
            0.1 * urgency
        )
        
        # Enforce capacity constraint
        if field_value > self.max_field_size:
            logger.warning(f"Field value {field_value} exceeds max size {self.max_field_size}")
            field_value = min(field_value, self.max_field_size)
        
        # Calculate gradient (simplified directional derivatives)
        gradient = self._calculate_scalar_gradient(context_data, field_value)
        
        # Create field state
        field_state = FieldState(
            field_id=field_id,
            field_type=FieldType.SCALAR,
            value=field_value,
            gradient=gradient,
            weights=self.field_weights.copy(),
            stability=0.8,  # Initial stability
            timestamp=datetime.utcnow()
        )
        
        # Store field state
        self.field_states[field_id] = field_state
        
        return {
            "field_id": field_id,
            "field_type": FieldType.SCALAR.value,
            "field_value": float(field_value),
            "gradient": gradient,
            "field_timestamp": datetime.utcnow().isoformat(),
            "influence_weights": self.field_weights,
            "stability": field_state.stability
        }
    
    async def _update_vector_field(
        self,
        session_id: str,
        field_id: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update vector field representation"""
        
        try:
            # Generate embedding for context
            if self.embedding_model:
                context_text = self._extract_context_text(context_data)
                embedding = self.embedding_model.encode([context_text])[0]
                
                # Calculate vector field value (magnitude)
                field_value = float(np.linalg.norm(embedding))
                
                # Calculate gradient (direction of steepest increase)
                gradient = (embedding / field_value).tolist() if field_value > 0 else [0.0] * len(embedding)
                
            else:
                # Fallback to basic vector representation
                features = [
                    context_data.get("task_priority", 0.5),
                    len(context_data.get("agents", [])) / 10.0,
                    len(context_data.get("description", "")) / 100.0,
                    context_data.get("importance", 0.5),
                    context_data.get("complexity", 0.5)
                ]
                
                field_value = float(np.linalg.norm(features))
                gradient = (np.array(features) / field_value).tolist() if field_value > 0 else [0.0] * len(features)
            
            # Create field state
            field_state = FieldState(
                field_id=field_id,
                field_type=FieldType.VECTOR,
                value=field_value,
                gradient=gradient,
                weights=self.field_weights.copy(),
                stability=0.7,  # Vector fields slightly less stable
                timestamp=datetime.utcnow()
            )
            
            # Store field state
            self.field_states[field_id] = field_state
            
            return {
                "field_id": field_id,
                "field_type": FieldType.VECTOR.value,
                "field_value": field_value,
                "gradient": gradient[:3],  # Limit gradient size for storage
                "field_timestamp": datetime.utcnow().isoformat(),
                "vector_dimension": len(gradient),
                "stability": field_state.stability
            }
            
        except Exception as e:
            logger.error(f"Vector field update failed: {e}")
            # Fallback to scalar field
            return await self._update_scalar_field(session_id, field_id, context_data)
    
    async def _update_tensor_field(
        self,
        session_id: str,
        field_id: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update tensor field representation (simplified as matrix)"""
        
        try:
            # Create simple 2D tensor from context features
            features = [
                [context_data.get("task_priority", 0.5), context_data.get("importance", 0.5)],
                [len(context_data.get("agents", [])) / 10.0, context_data.get("complexity", 0.5)],
                [len(context_data.get("description", "")) / 100.0, context_data.get("urgency", 0.5)]
            ]
            
            tensor = np.array(features)
            field_value = float(np.linalg.norm(tensor))
            
            # Flatten tensor for gradient
            gradient = tensor.flatten().tolist()
            
            # Create field state
            field_state = FieldState(
                field_id=field_id,
                field_type=FieldType.TENSOR,
                value=field_value,
                gradient=gradient,
                weights=self.field_weights.copy(),
                stability=0.6,  # Tensor fields less stable
                timestamp=datetime.utcnow()
            )
            
            # Store field state
            self.field_states[field_id] = field_state
            
            return {
                "field_id": field_id,
                "field_type": FieldType.TENSOR.value,
                "field_value": field_value,
                "gradient": gradient,
                "field_timestamp": datetime.utcnow().isoformat(),
                "tensor_shape": list(tensor.shape),
                "stability": field_state.stability
            }
            
        except Exception as e:
            logger.error(f"Tensor field update failed: {e}")
            # Fallback to scalar field
            return await self._update_scalar_field(session_id, field_id, context_data)
    
    def _calculate_scalar_gradient(
        self,
        context_data: Dict[str, Any],
        field_value: float
    ) -> List[float]:
        """Calculate gradient for scalar field"""
        
        # Simplified 3D gradient calculation
        h = 0.01  # Step size for numerical differentiation
        
        # Calculate partial derivatives
        base_components = [
            context_data.get("task_priority", 0.5),
            len(context_data.get("agents", [])) / 10.0,
            len(context_data.get("description", "")) / 100.0
        ]
        
        gradient = []
        for i, component in enumerate(base_components):
            # Perturb component
            perturbed_data = context_data.copy()
            if i == 0:
                perturbed_data["task_priority"] = component + h
            elif i == 1:
                # Simulate adding an agent
                agents = perturbed_data.get("agents", [])
                perturbed_data["agents"] = agents + ["virtual_agent"]
            else:
                # Extend description
                desc = perturbed_data.get("description", "")
                perturbed_data["description"] = desc + " additional context"
            
            # Calculate perturbed field value
            perturbed_priority = perturbed_data.get("task_priority", 0.5)
            perturbed_activity = len(perturbed_data.get("agents", [])) / 10.0
            perturbed_relevance = len(perturbed_data.get("description", "")) / 100.0
            
            perturbed_value = (
                self.field_weights["task_priority"] * perturbed_priority +
                self.field_weights["agent_activity"] * perturbed_activity +
                self.field_weights["context_relevance"] * perturbed_relevance
            )
            
            # Partial derivative
            partial_derivative = (perturbed_value - field_value) / h
            gradient.append(partial_derivative)
        
        return gradient
    
    def _extract_context_text(self, context_data: Dict[str, Any]) -> str:
        """Extract text content from context data for embedding"""
        
        text_parts = []
        
        # Extract text fields
        if "title" in context_data:
            text_parts.append(str(context_data["title"]))
        
        if "description" in context_data:
            text_parts.append(str(context_data["description"]))
        
        # Extract from agents list
        if "agents" in context_data:
            agents = context_data["agents"]
            if isinstance(agents, list):
                text_parts.extend([str(agent) for agent in agents])
        
        # Extract from metadata
        if "metadata" in context_data:
            metadata = context_data["metadata"]
            if isinstance(metadata, dict):
                text_parts.extend([str(v) for v in metadata.values() if isinstance(v, (str, int, float))])
        
        return " ".join(text_parts) if text_parts else "empty context"
    
    async def propagate_influence(
        self,
        session_id: str,
        propagation_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Propagate field influence using gradient: ∇C(x)
        
        Args:
            session_id: Session identifier
            propagation_steps: Number of propagation iterations
            
        Returns:
            Propagation results and updated field state
        """
        propagation_start = time.time()
        
        try:
            context = self.contexts.get(session_id)
            if not context:
                raise ValueError(f"No context found for session {session_id}")
            
            field_id = context.get("field_id")
            if not field_id or field_id not in self.field_states:
                raise ValueError("No field state found for propagation")
            
            field_state = self.field_states[field_id]
            original_value = field_state.value
            original_gradient = field_state.gradient.copy()
            
            propagation_history = []
            
            # Iterative propagation
            for step in range(propagation_steps):
                # Update field value based on gradient
                gradient_magnitude = np.linalg.norm(field_state.gradient)
                
                if gradient_magnitude > 0:
                    # Normalize gradient and apply propagation
                    normalized_gradient = np.array(field_state.gradient) / gradient_magnitude
                    propagation_step = self.propagation_alpha * normalized_gradient
                    
                    # Update field value
                    new_value = field_state.value + np.sum(propagation_step)
                    
                    # Update gradient (decay over propagation)
                    decay_factor = 0.9 ** (step + 1)
                    new_gradient = (np.array(field_state.gradient) * decay_factor).tolist()
                    
                    # Create new field state
                    field_state = FieldState(
                        field_id=field_state.field_id,
                        field_type=field_state.field_type,
                        value=new_value,
                        gradient=new_gradient,
                        weights=field_state.weights,
                        stability=field_state.stability * 0.95,  # Slight stability decay
                        timestamp=datetime.utcnow()
                    )
                    
                    propagation_history.append({
                        "step": step + 1,
                        "value": new_value,
                        "gradient_magnitude": float(np.linalg.norm(new_gradient)),
                        "propagation_strength": float(np.sum(propagation_step))
                    })
                else:
                    # No gradient - stop propagation
                    break
            
            # Update stored field state
            self.field_states[field_id] = field_state
            
            # Update context
            context.update({
                "field_value": field_state.value,
                "gradient": field_state.gradient,
                "propagation_timestamp": datetime.utcnow().isoformat(),
                "propagation_steps": len(propagation_history),
                "stability": field_state.stability
            })
            
            # Calculate propagation metrics
            value_change = field_state.value - original_value
            gradient_change = np.linalg.norm(
                np.array(field_state.gradient) - np.array(original_gradient)
            )
            
            result = {
                "session_id": session_id,
                "field_id": field_id,
                "original_value": original_value,
                "final_value": field_state.value,
                "value_change": float(value_change),
                "gradient_change": float(gradient_change),
                "propagation_history": propagation_history,
                "propagation_time": time.time() - propagation_start,
                "stability": field_state.stability
            }
            
            # Update metrics
            self.field_metrics["propagation_events"] += 1
            if field_state.stability < self.stability_threshold:
                self.field_metrics["stability_warnings"] += 1
            
            logger.info(
                f"Field propagation completed for session {session_id}: "
                f"value_change={value_change:.3f}, steps={len(propagation_history)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error propagating influence for session {session_id}: {str(e)}")
            raise
    
    async def model_context_interactions(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Model interactions between task contexts using field theory
        
        Args:
            db: Database session
            task_id: Primary task identifier
            user_id: User identifier
            similarity_threshold: Minimum similarity for interactions
            
        Returns:
            Interaction analysis and field updates
        """
        interaction_start = time.time()
        
        try:
            # Retrieve primary task
            primary_task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not primary_task:
                raise ValueError(f"Task {task_id} not found")
            
            # Get related tasks
            related_tasks = db.query(Task).filter(
                and_(Task.owner_id == user_id, Task.id != task_id)
            ).limit(10).all()  # Limit for performance
            
            if not related_tasks:
                return {"message": "No related tasks found for interaction modeling"}
            
            interactions = []
            
            if self.embedding_model:
                # Generate embeddings for semantic similarity
                primary_text = f"{primary_task.title} {primary_task.description or ''}"
                primary_embedding = self.embedding_model.encode([primary_text])[0]
                
                for related_task in related_tasks:
                    related_text = f"{related_task.title} {related_task.description or ''}"
                    related_embedding = self.embedding_model.encode([related_text])[0]
                    
                    # Calculate semantic similarity
                    if SKLEARN_AVAILABLE:
                        similarity = cosine_similarity(
                            primary_embedding.reshape(1, -1),
                            related_embedding.reshape(1, -1)
                        )[0, 0]
                    else:
                        # Fallback to dot product similarity
                        similarity = np.dot(primary_embedding, related_embedding) / (
                            np.linalg.norm(primary_embedding) * np.linalg.norm(related_embedding)
                        )
                    
                    if similarity >= similarity_threshold:
                        # Calculate interaction strength
                        importance_factor = (
                            (primary_task.importance or 0.5) + 
                            (related_task.importance or 0.5)
                        ) / 2.0
                        
                        interaction_strength = similarity * importance_factor
                        
                        # Create field interaction
                        interaction = FieldInteraction(
                            source_field=f"task_{primary_task.id}",
                            target_field=f"task_{related_task.id}",
                            interaction_type="semantic_similarity",
                            strength=float(interaction_strength),
                            semantic_similarity=float(similarity),
                            timestamp=datetime.utcnow()
                        )
                        
                        interactions.append(interaction)
                        
                        # Store interaction data for task
                        interaction_data = {
                            "task_id": related_task.id,
                            "interaction_strength": float(interaction_strength),
                            "semantic_similarity": float(similarity),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # Add to primary task's interactions
                        if not primary_task.interactions:
                            primary_task.interactions = []
                        primary_task.interactions.append(interaction_data)
            
            else:
                # Fallback to simple text-based similarity
                primary_words = set((primary_task.title + " " + (primary_task.description or "")).lower().split())
                
                for related_task in related_tasks:
                    related_words = set((related_task.title + " " + (related_task.description or "")).lower().split())
                    
                    # Jaccard similarity
                    if primary_words and related_words:
                        intersection = len(primary_words.intersection(related_words))
                        union = len(primary_words.union(related_words))
                        similarity = intersection / union if union > 0 else 0.0
                        
                        if similarity >= similarity_threshold:
                            interaction = FieldInteraction(
                                source_field=f"task_{primary_task.id}",
                                target_field=f"task_{related_task.id}",
                                interaction_type="text_similarity",
                                strength=similarity,
                                semantic_similarity=similarity,
                                timestamp=datetime.utcnow()
                            )
                            
                            interactions.append(interaction)
            
            # Calculate emergent effect
            emergent_effect = 0.0
            if interactions:
                interaction_strengths = [i.strength for i in interactions]
                emergent_effect = np.mean(interaction_strengths) * len(interactions) / 10.0
                emergent_effect = min(emergent_effect, 1.0)  # Normalize
            
            # Store interaction results
            primary_task.emergent_effect = emergent_effect
            
            # Cache embeddings if available
            if self.embedding_model and not primary_task.embeddings:
                primary_task.embeddings = primary_embedding.tolist()
            
            db.commit()
            
            # Store interactions in manager
            self.field_interactions.extend(interactions)
            
            result = {
                "task_id": task_id,
                "interactions_found": len(interactions),
                "interactions": [asdict(i) for i in interactions],
                "emergent_effect": emergent_effect,
                "similarity_threshold": similarity_threshold,
                "interaction_time": time.time() - interaction_start
            }
            
            logger.info(
                f"Context interaction modeling completed for task {task_id}: "
                f"interactions={len(interactions)}, emergent_effect={emergent_effect:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error modeling context interactions for task {task_id}: {str(e)}")
            raise
    
    async def manage_dynamic_field(
        self,
        session_id: str,
        alpha: float = 0.1,
        beta: float = 0.2
    ) -> Dict[str, Any]:
        """
        Dynamic field management: dC/dt = α * ∇C + β * I(x, y)
        
        Args:
            session_id: Session identifier
            alpha: Gradient influence factor
            beta: Interaction influence factor
            
        Returns:
            Dynamic field update results
        """
        dynamic_start = time.time()
        
        try:
            context = self.contexts.get(session_id)
            if not context:
                raise ValueError(f"No context found for session {session_id}")
            
            field_id = context.get("field_id")
            if not field_id or field_id not in self.field_states:
                raise ValueError("No field state found for dynamic management")
            
            field_state = self.field_states[field_id]
            current_value = field_state.value
            prev_value = context.get("prev_field_value", current_value)
            
            # Dynamic update calculation
            gradient = np.array(field_state.gradient)
            gradient_contribution = alpha * np.sum(gradient)
            
            # Interaction contribution
            session_interactions = [
                i for i in self.field_interactions 
                if session_id in i.source_field or session_id in i.target_field
            ]
            
            interaction_sum = sum(i.strength for i in session_interactions) if session_interactions else 0.1
            interaction_contribution = beta * interaction_sum
            
            # Calculate new field value
            new_field_value = current_value + gradient_contribution + interaction_contribution
            
            # Stability analysis: Stability = min(ΔC_i)
            field_delta = abs(new_field_value - prev_value)
            stability = max(0.0, 1.0 - field_delta)  # Stability decreases with large changes
            
            # Stability warning
            if stability < self.stability_threshold:
                logger.warning(f"Low stability {stability:.3f} detected for session {session_id}")
                self.field_metrics["stability_warnings"] += 1
            
            # Update field state
            updated_field_state = FieldState(
                field_id=field_state.field_id,
                field_type=field_state.field_type,
                value=new_field_value,
                gradient=field_state.gradient,  # Gradient remains same for this update
                weights=field_state.weights,
                stability=stability,
                timestamp=datetime.utcnow()
            )
            
            # Store updated field state
            self.field_states[field_id] = updated_field_state
            
            # Update context
            context.update({
                "field_value": new_field_value,
                "prev_field_value": current_value,
                "stability": stability,
                "gradient_contribution": gradient_contribution,
                "interaction_contribution": interaction_contribution,
                "update_timestamp": datetime.utcnow().isoformat()
            })
            
            result = {
                "session_id": session_id,
                "field_id": field_id,
                "previous_value": current_value,
                "new_value": new_field_value,
                "value_change": new_field_value - current_value,
                "stability": stability,
                "gradient_contribution": gradient_contribution,
                "interaction_contribution": interaction_contribution,
                "interactions_used": len(session_interactions),
                "update_time": time.time() - dynamic_start
            }
            
            logger.info(
                f"Dynamic field management completed for session {session_id}: "
                f"value={new_field_value:.3f}, stability={stability:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error managing dynamic field for session {session_id}: {str(e)}")
            raise
    
    async def optimize_field(
        self,
        db: Session,
        task_id: int,
        user_id: int,
        objectives: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Multi-objective field optimization: O* = arg max_O [Performance(C), Stability(C), Adaptability(C)]
        
        Args:
            db: Database session
            task_id: Task identifier
            user_id: User identifier
            objectives: Optimization objectives and weights
            
        Returns:
            Field optimization results
        """
        optimization_start = time.time()
        
        try:
            # Retrieve task
            task = db.query(Task).filter(
                and_(Task.id == task_id, Task.owner_id == user_id)
            ).first()
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            # Default objectives
            if objectives is None:
                objectives = {
                    "performance": 0.4,
                    "stability": 0.3,
                    "adaptability": 0.3
                }
            
            # Get current field state
            current_field_value = task.field_value or 0.5
            current_stability = task.stability or 0.5
            current_interactions = task.interactions or []
            
            # Define optimization function
            def objective_function(weights):
                """Multi-objective optimization function"""
                try:
                    # Unpack optimization variables
                    task_priority_weight = max(0.0, min(1.0, weights[0]))
                    agent_activity_weight = max(0.0, min(1.0, weights[1]))
                    context_relevance_weight = max(0.0, min(1.0, weights[2]))
                    
                    # Normalize weights
                    total_weight = task_priority_weight + agent_activity_weight + context_relevance_weight
                    if total_weight > 0:
                        task_priority_weight /= total_weight
                        agent_activity_weight /= total_weight
                        context_relevance_weight /= total_weight
                    
                    # Calculate performance objective
                    performance = (
                        task_priority_weight * (task.importance or 0.5) +
                        agent_activity_weight * 0.8 +  # Assume good agent activity
                        context_relevance_weight * min(len(task.description or "") / 100.0, 1.0)
                    )
                    
                    # Calculate stability objective
                    weight_variance = np.var([task_priority_weight, agent_activity_weight, context_relevance_weight])
                    stability = max(0.1, 1.0 - weight_variance)  # Lower variance = higher stability
                    
                    # Calculate adaptability objective
                    interaction_count = len(current_interactions)
                    adaptability = min(1.0, interaction_count / 5.0)  # Normalize by expected max
                    
                    # Combined objective (maximize)
                    total_objective = (
                        objectives["performance"] * performance +
                        objectives["stability"] * stability +
                        objectives["adaptability"] * adaptability
                    )
                    
                    return -total_objective  # Minimize negative (maximize objective)
                    
                except Exception as e:
                    logger.warning(f"Objective function evaluation failed: {e}")
                    return float('inf')
            
            # Initial guess (current field weights)
            initial_weights = [
                self.field_weights["task_priority"],
                self.field_weights["agent_activity"],
                self.field_weights["context_relevance"]
            ]
            
            # Optimization bounds
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            
            # Perform optimization
            result = minimize(
                objective_function,
                initial_weights,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                # Extract optimal weights
                optimal_weights = {
                    "task_priority": result.x[0],
                    "agent_activity": result.x[1],
                    "context_relevance": result.x[2]
                }
                
                # Normalize weights
                total_weight = sum(optimal_weights.values())
                if total_weight > 0:
                    optimal_weights = {k: v / total_weight for k, v in optimal_weights.items()}
                
                # Calculate optimized field value
                optimized_field_value = (
                    optimal_weights["task_priority"] * (task.importance or 0.5) +
                    optimal_weights["agent_activity"] * 0.8 +
                    optimal_weights["context_relevance"] * min(len(task.description or "") / 100.0, 1.0)
                )
                
                # Calculate improvement metrics
                field_improvement = optimized_field_value - current_field_value
                objective_improvement = abs(result.fun)
                
                # Store optimization results in task
                optimization_result = {
                    "optimal_weights": optimal_weights,
                    "optimized_field_value": optimized_field_value,
                    "field_improvement": field_improvement,
                    "objective_improvement": objective_improvement,
                    "objectives_used": objectives,
                    "optimization_success": True,
                    "iterations": result.nit,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Update task field values
                task.influence_weights = optimal_weights
                task.field_value = optimized_field_value
                
                db.commit()
                
                # Update manager weights if significant improvement
                if field_improvement > 0.1:
                    self.field_weights.update(optimal_weights)
                
                optimization_result.update({
                    "task_id": task_id,
                    "optimization_time": time.time() - optimization_start
                })
                
                # Update metrics
                self.field_metrics["optimization_runs"] += 1
                
                logger.info(
                    f"Field optimization completed for task {task_id}: "
                    f"improvement={field_improvement:.3f}, iterations={result.nit}"
                )
                
                return optimization_result
                
            else:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
        except Exception as e:
            logger.error(f"Error optimizing field for task {task_id}: {str(e)}")
            return {
                "task_id": task_id,
                "optimization_success": False,
                "error": str(e),
                "optimization_time": time.time() - optimization_start
            }
    
    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get context for session"""
        return self.contexts.get(session_id, {})
    
    async def update_context(self, session_id: str, context: Dict[str, Any]):
        """Update context for session"""
        self.contexts[session_id] = context
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get comprehensive field theory statistics"""
        
        if not self.field_states:
            return {
                "message": "No field states available",
                "metrics": self.field_metrics
            }
        
        # Field state statistics
        field_types = {}
        stability_scores = []
        field_values = []
        
        for field_state in self.field_states.values():
            field_type = field_state.field_type.value
            if field_type not in field_types:
                field_types[field_type] = 0
            field_types[field_type] += 1
            
            stability_scores.append(field_state.stability)
            field_values.append(field_state.value)
        
        # Interaction statistics
        interaction_types = {}
        interaction_strengths = []
        
        for interaction in self.field_interactions:
            interaction_type = interaction.interaction_type
            if interaction_type not in interaction_types:
                interaction_types[interaction_type] = 0
            interaction_types[interaction_type] += 1
            
            interaction_strengths.append(interaction.strength)
        
        return {
            "total_field_states": len(self.field_states),
            "field_type_distribution": field_types,
            "average_stability": np.mean(stability_scores) if stability_scores else 0.0,
            "average_field_value": np.mean(field_values) if field_values else 0.0,
            "stability_variance": np.var(stability_scores) if stability_scores else 0.0,
            "total_interactions": len(self.field_interactions),
            "interaction_type_distribution": interaction_types,
            "average_interaction_strength": np.mean(interaction_strengths) if interaction_strengths else 0.0,
            "active_contexts": len(self.contexts),
            "field_metrics": self.field_metrics,
            "field_weights": self.field_weights,
            "stability_threshold": self.stability_threshold,
            "max_field_size": self.max_field_size
        }
