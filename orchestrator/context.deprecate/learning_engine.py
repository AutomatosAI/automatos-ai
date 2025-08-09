
"""
Learning and Pattern Recognition Engine
=======================================

Advanced learning system with historical pattern recognition and adaptive intelligence.
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pickle
import hashlib
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import PgVectorStore
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class TaskPattern:
    """Represents a learned task pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    success_rate: float
    usage_count: int
    context_features: Dict[str, Any]
    outcome_features: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class LearningEvent:
    """Represents a learning event"""
    event_id: str
    task_description: str
    task_type: str
    context_used: Dict[str, Any]
    outcome: str
    success: bool
    execution_time: float
    agent_used: str
    user_feedback: Dict[str, Any]
    timestamp: datetime

class PatternRecognitionEngine:
    """Advanced pattern recognition for task analysis"""
    
    def __init__(self, vector_store: PgVectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.patterns = {}
        self.pattern_clusters = {}
        self.success_predictors = {}
        
    async def analyze_task_patterns(self, historical_tasks: List[Dict[str, Any]]) -> List[TaskPattern]:
        """Analyze historical tasks to identify patterns"""
        
        if not historical_tasks:
            return []
        
        try:
            # Extract features from tasks
            task_features = await self._extract_task_features(historical_tasks)
            
            # Cluster similar tasks
            clusters = await self._cluster_tasks(task_features)
            
            # Analyze each cluster for patterns
            patterns = []
            for cluster_id, cluster_tasks in clusters.items():
                pattern = await self._analyze_cluster_pattern(cluster_id, cluster_tasks)
                if pattern:
                    patterns.append(pattern)
            
            # Store patterns
            await self._store_patterns(patterns)
            
            logger.info(f"Identified {len(patterns)} task patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing task patterns: {str(e)}")
            return []
    
    async def _extract_task_features(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features from tasks for pattern analysis"""
        
        features = []
        
        for task in tasks:
            try:
                # Generate embedding for task description
                task_embedding = await self.embedding_generator.generate_embeddings([task['task_description']])
                
                if task_embedding:
                    feature_vector = {
                        'task_id': task.get('id'),
                        'embedding': task_embedding[0]['embedding'],
                        'task_type': task.get('task_type', 'unknown'),
                        'success': task.get('success', False),
                        'execution_time': task.get('execution_time', 0),
                        'agent_used': task.get('agent_used', 'unknown'),
                        'context_size': len(str(task.get('context_used', {}))),
                        'outcome_length': len(task.get('outcome', '')),
                        'timestamp': task.get('created_at', datetime.now())
                    }
                    
                    # Extract linguistic features
                    linguistic_features = self._extract_linguistic_features(task['task_description'])
                    feature_vector.update(linguistic_features)
                    
                    features.append(feature_vector)
                    
            except Exception as e:
                logger.warning(f"Error extracting features for task {task.get('id')}: {str(e)}")
                continue
        
        return features
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from task description"""
        
        words = text.lower().split()
        
        # Action words (verbs)
        action_words = ['create', 'build', 'develop', 'implement', 'fix', 'debug', 'optimize', 'test', 'deploy']
        action_count = sum(1 for word in words if word in action_words)
        
        # Technical terms
        tech_terms = ['api', 'database', 'frontend', 'backend', 'authentication', 'security', 'performance']
        tech_count = sum(1 for word in words if word in tech_terms)
        
        # Complexity indicators
        complexity_words = ['complex', 'advanced', 'sophisticated', 'enterprise', 'scalable', 'distributed']
        complexity_score = sum(1 for word in words if word in complexity_words)
        
        return {
            'word_count': len(words),
            'action_word_count': action_count,
            'tech_term_count': tech_count,
            'complexity_score': complexity_score,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
    
    async def _cluster_tasks(self, task_features: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster tasks based on similarity"""
        
        if len(task_features) < 2:
            return {0: task_features}
        
        try:
            # Extract embeddings for clustering
            embeddings = [feature['embedding'] for feature in task_features]
            embeddings_array = np.array(embeddings)
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(task_features) // 5), 10)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)
            
            # Group tasks by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(task_features[i])
            
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error clustering tasks: {str(e)}")
            return {0: task_features}
    
    async def _analyze_cluster_pattern(self, cluster_id: int, cluster_tasks: List[Dict[str, Any]]) -> Optional[TaskPattern]:
        """Analyze a cluster to identify patterns"""
        
        if len(cluster_tasks) < 2:
            return None
        
        try:
            # Calculate success rate
            successful_tasks = [task for task in cluster_tasks if task['success']]
            success_rate = len(successful_tasks) / len(cluster_tasks)
            
            # Identify common characteristics
            task_types = [task['task_type'] for task in cluster_tasks]
            most_common_type = Counter(task_types).most_common(1)[0][0]
            
            agents_used = [task['agent_used'] for task in cluster_tasks]
            most_common_agent = Counter(agents_used).most_common(1)[0][0]
            
            # Calculate average execution time
            avg_execution_time = np.mean([task['execution_time'] for task in cluster_tasks])
            
            # Extract context features
            context_features = self._extract_context_features(cluster_tasks)
            
            # Generate pattern description
            description = self._generate_pattern_description(cluster_tasks, most_common_type, success_rate)
            
            # Create pattern
            pattern_id = f"pattern_{cluster_id}_{hashlib.md5(description.encode()).hexdigest()[:8]}"
            
            pattern = TaskPattern(
                pattern_id=pattern_id,
                pattern_type=most_common_type,
                description=description,
                success_rate=success_rate,
                usage_count=len(cluster_tasks),
                context_features=context_features,
                outcome_features={
                    'avg_execution_time': avg_execution_time,
                    'preferred_agent': most_common_agent,
                    'success_indicators': self._identify_success_indicators(successful_tasks)
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing cluster pattern: {str(e)}")
            return None
    
    def _extract_context_features(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common context features from tasks"""
        
        context_sizes = [task['context_size'] for task in tasks]
        word_counts = [task['word_count'] for task in tasks]
        complexity_scores = [task['complexity_score'] for task in tasks]
        
        return {
            'avg_context_size': np.mean(context_sizes),
            'avg_word_count': np.mean(word_counts),
            'avg_complexity': np.mean(complexity_scores),
            'context_size_range': [min(context_sizes), max(context_sizes)],
            'common_features': self._find_common_features(tasks)
        }
    
    def _find_common_features(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common features across tasks"""
        
        # Aggregate linguistic features
        action_counts = [task.get('action_word_count', 0) for task in tasks]
        tech_counts = [task.get('tech_term_count', 0) for task in tasks]
        
        return {
            'high_action_words': np.mean(action_counts) > 2,
            'high_tech_terms': np.mean(tech_counts) > 1,
            'typical_length': 'short' if np.mean([task['word_count'] for task in tasks]) < 20 else 'long'
        }
    
    def _identify_success_indicators(self, successful_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify indicators of successful task completion"""
        
        if not successful_tasks:
            return {}
        
        # Analyze successful tasks for common patterns
        avg_execution_time = np.mean([task['execution_time'] for task in successful_tasks])
        common_agents = Counter([task['agent_used'] for task in successful_tasks])
        
        return {
            'optimal_execution_time_range': [avg_execution_time * 0.8, avg_execution_time * 1.2],
            'successful_agents': dict(common_agents.most_common(3)),
            'success_context_size': np.mean([task['context_size'] for task in successful_tasks])
        }
    
    def _generate_pattern_description(self, tasks: List[Dict[str, Any]], 
                                    task_type: str, success_rate: float) -> str:
        """Generate human-readable pattern description"""
        
        avg_words = np.mean([task['word_count'] for task in tasks])
        avg_complexity = np.mean([task['complexity_score'] for task in tasks])
        
        description_parts = [f"Tasks of type '{task_type}'"]
        
        if avg_words < 10:
            description_parts.append("with brief descriptions")
        elif avg_words > 30:
            description_parts.append("with detailed descriptions")
        
        if avg_complexity > 1:
            description_parts.append("involving complex requirements")
        
        if success_rate > 0.8:
            description_parts.append("typically successful")
        elif success_rate < 0.5:
            description_parts.append("often challenging")
        
        return " ".join(description_parts)
    
    async def _store_patterns(self, patterns: List[TaskPattern]):
        """Store identified patterns in the database"""
        
        try:
            for pattern in patterns:
                # Generate embedding for pattern description
                pattern_embedding = await self.embedding_generator.generate_embeddings([pattern.description])
                
                if pattern_embedding and hasattr(self.vector_store, 'add_context_pattern'):
                    await self.vector_store.add_context_pattern(
                        pattern_name=pattern.pattern_id,
                        pattern_type=pattern.pattern_type,
                        embedding=pattern_embedding[0]['embedding'],
                        metadata=asdict(pattern)
                    )
                
                # Store in local cache
                self.patterns[pattern.pattern_id] = pattern
                
        except Exception as e:
            logger.error(f"Error storing patterns: {str(e)}")

class AdaptiveLearningEngine:
    """Main learning engine with adaptive capabilities"""
    
    def __init__(self, vector_store: PgVectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.pattern_engine = PatternRecognitionEngine(vector_store, embedding_generator)
        
        self.learning_events = []
        self.success_predictors = {}
        self.adaptation_rules = {}
        self.performance_metrics = {}
    
    async def learn_from_task_execution(self, task_description: str, task_type: str,
                                      context_used: Dict[str, Any], outcome: str,
                                      success: bool, execution_time: float,
                                      agent_used: str, user_feedback: Dict[str, Any] = None):
        """Learn from a completed task execution"""
        
        try:
            # Create learning event
            event = LearningEvent(
                event_id=f"event_{datetime.now().timestamp()}",
                task_description=task_description,
                task_type=task_type,
                context_used=context_used,
                outcome=outcome,
                success=success,
                execution_time=execution_time,
                agent_used=agent_used,
                user_feedback=user_feedback or {},
                timestamp=datetime.now()
            )
            
            self.learning_events.append(event)
            
            # Store in database
            await self._store_learning_event(event)
            
            # Update patterns
            await self._update_patterns(event)
            
            # Update success predictors
            await self._update_success_predictors(event)
            
            # Learn adaptation rules
            await self._learn_adaptation_rules(event)
            
            logger.info(f"Learned from task execution: {task_description[:50]}...")
            
        except Exception as e:
            logger.error(f"Error learning from task execution: {str(e)}")
    
    async def _store_learning_event(self, event: LearningEvent):
        """Store learning event in database"""
        
        try:
            if hasattr(self.vector_store, 'record_task_outcome'):
                await self.vector_store.record_task_outcome(
                    task_description=event.task_description,
                    task_type=event.task_type,
                    context_used=event.context_used,
                    success=event.success,
                    execution_time=event.execution_time,
                    agent_used=event.agent_used
                )
        except Exception as e:
            logger.error(f"Error storing learning event: {str(e)}")
    
    async def _update_patterns(self, event: LearningEvent):
        """Update patterns based on new learning event"""
        
        try:
            # Find similar patterns
            event_embedding = await self.embedding_generator.generate_embeddings([event.task_description])
            
            if event_embedding and hasattr(self.vector_store, 'find_similar_patterns'):
                similar_patterns = await self.vector_store.find_similar_patterns(
                    query_embedding=event_embedding[0]['embedding'],
                    pattern_type=event.task_type,
                    limit=3
                )
                
                # Update existing patterns or create new ones
                if similar_patterns and similar_patterns[0]['similarity'] > 0.8:
                    # Update existing pattern
                    await self._update_existing_pattern(similar_patterns[0], event)
                else:
                    # Create new pattern if this represents a new type of task
                    await self._create_new_pattern(event)
                    
        except Exception as e:
            logger.error(f"Error updating patterns: {str(e)}")
    
    async def _update_existing_pattern(self, pattern: Dict[str, Any], event: LearningEvent):
        """Update an existing pattern with new information"""
        
        try:
            pattern_name = pattern['pattern_name']
            
            # Update success rate using exponential moving average
            alpha = 0.1  # Learning rate
            current_success_rate = pattern.get('success_count', 0) / max(pattern.get('usage_count', 1), 1)
            new_success_rate = (1 - alpha) * current_success_rate + alpha * (1 if event.success else 0)
            
            # Update metadata
            updated_metadata = pattern.get('metadata', {})
            updated_metadata.update({
                'success_rate': new_success_rate,
                'last_updated': datetime.now().isoformat(),
                'recent_success': event.success
            })
            
            # Store updated pattern
            if hasattr(self.vector_store, 'add_context_pattern'):
                pattern_embedding = await self.embedding_generator.generate_embeddings([pattern_name])
                if pattern_embedding:
                    await self.vector_store.add_context_pattern(
                        pattern_name=pattern_name,
                        pattern_type=event.task_type,
                        embedding=pattern_embedding[0]['embedding'],
                        metadata=updated_metadata
                    )
                    
        except Exception as e:
            logger.error(f"Error updating existing pattern: {str(e)}")
    
    async def _create_new_pattern(self, event: LearningEvent):
        """Create a new pattern from a learning event"""
        
        try:
            pattern_name = f"pattern_{event.task_type}_{datetime.now().timestamp()}"
            
            metadata = {
                'created_from_event': event.event_id,
                'initial_success': event.success,
                'initial_execution_time': event.execution_time,
                'initial_agent': event.agent_used,
                'created_at': datetime.now().isoformat()
            }
            
            # Generate embedding for the pattern
            pattern_embedding = await self.embedding_generator.generate_embeddings([event.task_description])
            
            if pattern_embedding and hasattr(self.vector_store, 'add_context_pattern'):
                await self.vector_store.add_context_pattern(
                    pattern_name=pattern_name,
                    pattern_type=event.task_type,
                    embedding=pattern_embedding[0]['embedding'],
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Error creating new pattern: {str(e)}")
    
    async def _update_success_predictors(self, event: LearningEvent):
        """Update success prediction models"""
        
        try:
            task_type = event.task_type
            
            if task_type not in self.success_predictors:
                self.success_predictors[task_type] = {
                    'execution_time_success': [],
                    'context_size_success': [],
                    'agent_success': defaultdict(list)
                }
            
            predictor = self.success_predictors[task_type]
            
            # Record execution time vs success
            predictor['execution_time_success'].append((event.execution_time, event.success))
            
            # Record context size vs success
            context_size = len(str(event.context_used))
            predictor['context_size_success'].append((context_size, event.success))
            
            # Record agent vs success
            predictor['agent_success'][event.agent_used].append(event.success)
            
            # Keep only recent data (last 100 events per type)
            for key in ['execution_time_success', 'context_size_success']:
                if len(predictor[key]) > 100:
                    predictor[key] = predictor[key][-100:]
            
        except Exception as e:
            logger.error(f"Error updating success predictors: {str(e)}")
    
    async def _learn_adaptation_rules(self, event: LearningEvent):
        """Learn adaptation rules from user feedback and outcomes"""
        
        try:
            if not event.user_feedback:
                return
            
            # Extract adaptation signals from feedback
            feedback = event.user_feedback
            
            if 'context_too_much' in feedback and feedback['context_too_much']:
                self._add_adaptation_rule(event.task_type, 'reduce_context', 0.8)
            
            if 'context_too_little' in feedback and feedback['context_too_little']:
                self._add_adaptation_rule(event.task_type, 'increase_context', 1.2)
            
            if 'wrong_agent' in feedback and feedback['wrong_agent']:
                self._add_adaptation_rule(event.task_type, 'avoid_agent', event.agent_used)
            
            if 'preferred_style' in feedback:
                self._add_adaptation_rule(event.task_type, 'preferred_style', feedback['preferred_style'])
                
        except Exception as e:
            logger.error(f"Error learning adaptation rules: {str(e)}")
    
    def _add_adaptation_rule(self, task_type: str, rule_type: str, rule_value: Any):
        """Add an adaptation rule"""
        
        if task_type not in self.adaptation_rules:
            self.adaptation_rules[task_type] = {}
        
        if rule_type not in self.adaptation_rules[task_type]:
            self.adaptation_rules[task_type][rule_type] = []
        
        self.adaptation_rules[task_type][rule_type].append({
            'value': rule_value,
            'timestamp': datetime.now(),
            'confidence': 1.0
        })
    
    async def predict_task_success(self, task_description: str, task_type: str,
                                 context_size: int, agent: str) -> Dict[str, Any]:
        """Predict the likelihood of task success"""
        
        try:
            if task_type not in self.success_predictors:
                return {'success_probability': 0.5, 'confidence': 0.0, 'recommendations': []}
            
            predictor = self.success_predictors[task_type]
            predictions = []
            recommendations = []
            
            # Execution time prediction (if we have historical data)
            if predictor['execution_time_success']:
                # Simple heuristic: tasks with moderate execution times tend to be more successful
                avg_successful_time = np.mean([time for time, success in predictor['execution_time_success'] if success])
                if avg_successful_time:
                    predictions.append(0.7)  # Moderate confidence
                    recommendations.append(f"Optimal execution time around {avg_successful_time:.1f}s")
            
            # Context size prediction
            if predictor['context_size_success']:
                successful_context_sizes = [size for size, success in predictor['context_size_success'] if success]
                if successful_context_sizes:
                    avg_successful_context = np.mean(successful_context_sizes)
                    if abs(context_size - avg_successful_context) < avg_successful_context * 0.3:
                        predictions.append(0.8)
                    else:
                        predictions.append(0.4)
                        recommendations.append(f"Consider adjusting context size (optimal: ~{avg_successful_context:.0f})")
            
            # Agent prediction
            if agent in predictor['agent_success']:
                agent_success_rate = np.mean(predictor['agent_success'][agent])
                predictions.append(agent_success_rate)
                if agent_success_rate < 0.6:
                    recommendations.append(f"Consider using a different agent (current success rate: {agent_success_rate:.1%})")
            
            # Overall prediction
            if predictions:
                success_probability = np.mean(predictions)
                confidence = min(len(predictions) / 3.0, 1.0)  # Higher confidence with more data points
            else:
                success_probability = 0.5
                confidence = 0.0
            
            return {
                'success_probability': success_probability,
                'confidence': confidence,
                'recommendations': recommendations,
                'data_points': len(predictor['execution_time_success'])
            }
            
        except Exception as e:
            logger.error(f"Error predicting task success: {str(e)}")
            return {'success_probability': 0.5, 'confidence': 0.0, 'recommendations': []}
    
    async def get_adaptation_recommendations(self, task_type: str) -> Dict[str, Any]:
        """Get adaptation recommendations for a task type"""
        
        try:
            if task_type not in self.adaptation_rules:
                return {'recommendations': [], 'confidence': 0.0}
            
            rules = self.adaptation_rules[task_type]
            recommendations = []
            
            for rule_type, rule_list in rules.items():
                if rule_list:
                    # Get most recent and confident rule
                    latest_rule = max(rule_list, key=lambda x: x['timestamp'])
                    
                    if rule_type == 'reduce_context':
                        recommendations.append({
                            'type': 'context_adjustment',
                            'action': 'reduce',
                            'factor': latest_rule['value'],
                            'reason': 'Users found context overwhelming'
                        })
                    elif rule_type == 'increase_context':
                        recommendations.append({
                            'type': 'context_adjustment',
                            'action': 'increase',
                            'factor': latest_rule['value'],
                            'reason': 'Users needed more context'
                        })
                    elif rule_type == 'avoid_agent':
                        recommendations.append({
                            'type': 'agent_selection',
                            'action': 'avoid',
                            'agent': latest_rule['value'],
                            'reason': 'Poor performance feedback'
                        })
                    elif rule_type == 'preferred_style':
                        recommendations.append({
                            'type': 'style_preference',
                            'style': latest_rule['value'],
                            'reason': 'User preference'
                        })
            
            confidence = min(len(recommendations) / 3.0, 1.0)
            
            return {
                'recommendations': recommendations,
                'confidence': confidence,
                'total_rules': sum(len(rule_list) for rule_list in rules.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting adaptation recommendations: {str(e)}")
            return {'recommendations': [], 'confidence': 0.0}
    
    async def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        try:
            if not self.learning_events:
                return {'trends': {}, 'insights': []}
            
            # Group events by time periods
            recent_events = [e for e in self.learning_events if e.timestamp > datetime.now() - timedelta(days=7)]
            older_events = [e for e in self.learning_events if e.timestamp <= datetime.now() - timedelta(days=7)]
            
            trends = {}
            insights = []
            
            # Success rate trend
            if recent_events and older_events:
                recent_success_rate = np.mean([e.success for e in recent_events])
                older_success_rate = np.mean([e.success for e in older_events])
                
                trends['success_rate'] = {
                    'recent': recent_success_rate,
                    'previous': older_success_rate,
                    'change': recent_success_rate - older_success_rate
                }
                
                if recent_success_rate > older_success_rate + 0.1:
                    insights.append("Success rate is improving over time")
                elif recent_success_rate < older_success_rate - 0.1:
                    insights.append("Success rate is declining - may need attention")
            
            # Execution time trend
            if recent_events and older_events:
                recent_avg_time = np.mean([e.execution_time for e in recent_events])
                older_avg_time = np.mean([e.execution_time for e in older_events])
                
                trends['execution_time'] = {
                    'recent': recent_avg_time,
                    'previous': older_avg_time,
                    'change': recent_avg_time - older_avg_time
                }
                
                if recent_avg_time < older_avg_time * 0.8:
                    insights.append("Execution times are improving (getting faster)")
                elif recent_avg_time > older_avg_time * 1.2:
                    insights.append("Execution times are increasing - may indicate complexity growth")
            
            # Task type distribution
            task_types = Counter([e.task_type for e in self.learning_events])
            trends['task_distribution'] = dict(task_types.most_common(5))
            
            return {
                'trends': trends,
                'insights': insights,
                'total_events': len(self.learning_events),
                'recent_events': len(recent_events)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {str(e)}")
            return {'trends': {}, 'insights': []}
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        
        return {
            'total_learning_events': len(self.learning_events),
            'patterns_identified': len(self.pattern_engine.patterns),
            'success_predictors': len(self.success_predictors),
            'adaptation_rules': sum(len(rules) for rules in self.adaptation_rules.values()),
            'recent_activity': len([e for e in self.learning_events if e.timestamp > datetime.now() - timedelta(hours=24)])
        }
