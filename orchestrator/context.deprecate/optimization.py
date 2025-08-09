
"""
Context Optimization Module
Advanced performance monitoring, caching, and adaptive optimization
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
import asyncio
import time
import json
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Context optimization strategies"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    QUALITY = "quality"
    BALANCED = "balanced"

@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    operation: str
    duration: float
    memory_usage: float
    cache_hit: bool
    timestamp: datetime
    session_id: str
    optimization_applied: Optional[str] = None

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    importance_score: float = 1.0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            try:
                self.size_bytes = len(json.dumps(self.data, default=str).encode('utf-8'))
            except Exception:
                self.size_bytes = 1024  # Default estimate

class ContextOptimizationEngine:
    """Advanced context optimization with adaptive strategies"""
    
    def __init__(self, max_cache_size_mb: int = 256, max_cache_entries: int = 10000):
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=10000)
        self.operation_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        })
        
        # Caching system
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_cache_entries = max_cache_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_cache_size = 0
        
        # Optimization strategies
        self.current_strategy = OptimizationStrategy.BALANCED
        self.strategy_performance: Dict[OptimizationStrategy, Dict[str, float]] = {
            strategy: {'score': 0.0, 'usage_count': 0} for strategy in OptimizationStrategy
        }
        
        # Adaptive parameters
        self.adaptive_params = {
            'cache_hit_threshold': 0.8,
            'performance_degradation_threshold': 1.5,
            'memory_pressure_threshold': 0.9,
            'quality_loss_threshold': 0.15
        }
        
        logger.info(f"Initialized ContextOptimizationEngine with {max_cache_size_mb}MB cache")
    
    async def measure_operation(self, operation_name: str, session_id: str):
        """Context manager for measuring operation performance"""
        return PerformanceMeasurement(self, operation_name, session_id)
    
    async def optimize_context_access(self, session_id: str, context_key: str, 
                                    context_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize context access with caching and performance monitoring"""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = f"{session_id}:{context_key}"
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result is not None:
                # Cache hit
                duration = time.time() - start_time
                await self._record_performance(
                    "context_access", duration, session_id, cache_hit=True
                )
                
                optimization_info = {
                    "strategy": "cache_hit",
                    "duration_ms": duration * 1000,
                    "cache_hit": True,
                    "optimizations_applied": []
                }
                
                return cached_result, optimization_info
            
            # Cache miss - process and optimize
            optimized_data, optimizations = await self._apply_optimization_strategy(
                context_data, session_id
            )
            
            # Store in cache
            await self._store_in_cache(cache_key, optimized_data, session_id)
            
            # Record performance
            duration = time.time() - start_time
            await self._record_performance(
                "context_access", duration, session_id, cache_hit=False
            )
            
            optimization_info = {
                "strategy": self.current_strategy.value,
                "duration_ms": duration * 1000,
                "cache_hit": False,
                "optimizations_applied": optimizations
            }
            
            return optimized_data, optimization_info
            
        except Exception as e:
            logger.error(f"Error optimizing context access: {e}")
            return context_data, {"error": str(e)}
    
    async def _apply_optimization_strategy(self, context_data: Dict[str, Any], 
                                         session_id: str) -> Tuple[Dict[str, Any], List[str]]:
        """Apply current optimization strategy to context data"""
        try:
            optimizations_applied = []
            optimized_data = context_data.copy()
            
            if self.current_strategy == OptimizationStrategy.PERFORMANCE:
                optimized_data, perf_opts = await self._apply_performance_optimizations(optimized_data)
                optimizations_applied.extend(perf_opts)
                
            elif self.current_strategy == OptimizationStrategy.MEMORY:
                optimized_data, mem_opts = await self._apply_memory_optimizations(optimized_data)
                optimizations_applied.extend(mem_opts)
                
            elif self.current_strategy == OptimizationStrategy.QUALITY:
                optimized_data, qual_opts = await self._apply_quality_optimizations(optimized_data)
                optimizations_applied.extend(qual_opts)
                
            elif self.current_strategy == OptimizationStrategy.BALANCED:
                optimized_data, bal_opts = await self._apply_balanced_optimizations(optimized_data)
                optimizations_applied.extend(bal_opts)
            
            return optimized_data, optimizations_applied
            
        except Exception as e:
            logger.error(f"Error applying optimization strategy: {e}")
            return context_data, []
    
    async def _apply_performance_optimizations(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply performance-focused optimizations"""
        optimizations = []
        optimized = data.copy()
        
        try:
            # Remove or simplify large data structures
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 1000:
                    # Keep only most recent items
                    optimized[key] = value[-100:]
                    optimizations.append(f"truncated_large_list_{key}")
                
                elif isinstance(value, str) and len(value) > 10000:
                    # Truncate long strings
                    optimized[key] = value[:5000] + "..."
                    optimizations.append(f"truncated_long_string_{key}")
                
                elif isinstance(value, dict) and len(json.dumps(value, default=str)) > 50000:
                    # Simplify complex objects
                    simplified = {k: v for k, v in list(value.items())[:50]}
                    optimized[key] = simplified
                    optimizations.append(f"simplified_complex_object_{key}")
            
            return optimized, optimizations
            
        except Exception as e:
            logger.error(f"Error in performance optimizations: {e}")
            return data, []
    
    async def _apply_memory_optimizations(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply memory-focused optimizations"""
        optimizations = []
        optimized = {}
        
        try:
            # Keep only essential fields
            essential_fields = ['id', 'title', 'description', 'status', 'timestamp']
            
            for key, value in data.items():
                if key in essential_fields or len(str(value)) < 1000:
                    optimized[key] = value
                else:
                    optimizations.append(f"removed_non_essential_{key}")
            
            # Compress remaining data
            for key, value in optimized.items():
                if isinstance(value, str) and len(value) > 500:
                    # Simple compression: remove extra whitespace
                    compressed = ' '.join(value.split())
                    if len(compressed) < len(value) * 0.8:
                        optimized[key] = compressed
                        optimizations.append(f"compressed_string_{key}")
            
            return optimized, optimizations
            
        except Exception as e:
            logger.error(f"Error in memory optimizations: {e}")
            return data, []
    
    async def _apply_quality_optimizations(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply quality-preserving optimizations"""
        optimizations = []
        optimized = data.copy()
        
        try:
            # Enhance data quality while maintaining completeness
            for key, value in data.items():
                if isinstance(value, str):
                    # Clean up text while preserving meaning
                    cleaned = self._clean_text_quality(value)
                    if cleaned != value:
                        optimized[key] = cleaned
                        optimizations.append(f"quality_cleaned_{key}")
                
                elif isinstance(value, list):
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_list = []
                    for item in value:
                        item_str = json.dumps(item, default=str, sort_keys=True)
                        if item_str not in seen:
                            seen.add(item_str)
                            unique_list.append(item)
                    
                    if len(unique_list) < len(value):
                        optimized[key] = unique_list
                        optimizations.append(f"removed_duplicates_{key}")
            
            return optimized, optimizations
            
        except Exception as e:
            logger.error(f"Error in quality optimizations: {e}")
            return data, []
    
    async def _apply_balanced_optimizations(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Apply balanced optimizations considering all factors"""
        optimizations = []
        optimized = data.copy()
        
        try:
            # Balance between performance, memory, and quality
            data_size = len(json.dumps(data, default=str))
            
            if data_size > 100000:  # Large data - prioritize memory
                optimized, mem_opts = await self._apply_memory_optimizations(data)
                optimizations.extend([f"balanced_memory_{opt}" for opt in mem_opts])
            
            elif data_size > 50000:  # Medium data - prioritize performance
                optimized, perf_opts = await self._apply_performance_optimizations(optimized)
                optimizations.extend([f"balanced_performance_{opt}" for opt in perf_opts])
            
            else:  # Small data - prioritize quality
                optimized, qual_opts = await self._apply_quality_optimizations(optimized)
                optimizations.extend([f"balanced_quality_{opt}" for opt in qual_opts])
            
            return optimized, optimizations
            
        except Exception as e:
            logger.error(f"Error in balanced optimizations: {e}")
            return data, []
    
    def _clean_text_quality(self, text: str) -> str:
        """Clean text while preserving quality"""
        try:
            # Remove excessive whitespace
            cleaned = ' '.join(text.split())
            
            # Fix common formatting issues
            cleaned = cleaned.replace('  ', ' ')
            cleaned = cleaned.replace(' .', '.')
            cleaned = cleaned.replace(' ,', ',')
            cleaned = cleaned.replace('( ', '(')
            cleaned = cleaned.replace(' )', ')')
            
            return cleaned
            
        except Exception:
            return text
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache with LRU management"""
        try:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(cache_key)
                
                return entry.data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def _store_in_cache(self, cache_key: str, data: Dict[str, Any], session_id: str):
        """Store data in cache with intelligent eviction"""
        try:
            # Calculate entry size
            data_size = len(json.dumps(data, default=str).encode('utf-8'))
            
            # Check if we need to evict entries
            await self._evict_cache_entries_if_needed(data_size)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                data=data,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                size_bytes=data_size,
                importance_score=self._calculate_cache_importance(data, session_id)
            )
            
            # Store in cache
            self.cache[cache_key] = entry
            self.current_cache_size += data_size
            
            logger.debug(f"Stored {cache_key} in cache ({data_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
    
    async def _evict_cache_entries_if_needed(self, incoming_size: int):
        """Evict cache entries using intelligent strategy"""
        try:
            # Check size limits
            while (self.current_cache_size + incoming_size > self.max_cache_size_bytes or 
                   len(self.cache) >= self.max_cache_entries):
                
                if not self.cache:
                    break
                
                # Find entry to evict based on importance and access patterns
                eviction_candidate = await self._find_eviction_candidate()
                
                if eviction_candidate:
                    entry = self.cache[eviction_candidate]
                    self.current_cache_size -= entry.size_bytes
                    del self.cache[eviction_candidate]
                    
                    logger.debug(f"Evicted cache entry {eviction_candidate}")
                else:
                    break
            
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    async def _find_eviction_candidate(self) -> Optional[str]:
        """Find best candidate for cache eviction"""
        try:
            if not self.cache:
                return None
            
            # Score each entry for eviction (higher score = more likely to evict)
            eviction_scores = {}
            current_time = datetime.utcnow()
            
            for key, entry in self.cache.items():
                # Age factor (older = higher eviction score)
                age_hours = (current_time - entry.last_accessed).total_seconds() / 3600
                age_score = min(age_hours / 24, 5.0)  # Cap at 5 days
                
                # Access frequency factor (less accessed = higher eviction score)
                access_score = max(0, 5.0 - entry.access_count)
                
                # Size factor (larger = higher eviction score for memory pressure)
                size_score = entry.size_bytes / (1024 * 1024)  # MB
                
                # Importance factor (less important = higher eviction score)
                importance_score = max(0, 5.0 - entry.importance_score * 5)
                
                # Combined score
                total_score = (age_score * 0.4 + access_score * 0.3 + 
                              size_score * 0.2 + importance_score * 0.1)
                
                eviction_scores[key] = total_score
            
            # Return entry with highest eviction score
            return max(eviction_scores, key=eviction_scores.get)
            
        except Exception as e:
            logger.error(f"Error finding eviction candidate: {e}")
            return list(self.cache.keys())[0] if self.cache else None
    
    def _calculate_cache_importance(self, data: Dict[str, Any], session_id: str) -> float:
        """Calculate importance score for cache entry"""
        try:
            importance = 1.0
            
            # Data size factor (smaller data is often more important/frequently accessed)
            data_size = len(json.dumps(data, default=str))
            if data_size < 1000:
                importance += 0.5
            elif data_size > 100000:
                importance -= 0.3
            
            # Content type factors
            if 'id' in data:
                importance += 0.3  # ID-based data is often important
            
            if 'timestamp' in data:
                importance += 0.2  # Timestamped data has temporal importance
            
            if any(key in data for key in ['status', 'state', 'result']):
                importance += 0.4  # State information is important
            
            return max(0.1, min(2.0, importance))
            
        except Exception:
            return 1.0
    
    async def _record_performance(self, operation: str, duration: float, 
                                session_id: str, cache_hit: bool = False, 
                                memory_usage: float = 0.0):
        """Record performance metrics"""
        try:
            metrics = PerformanceMetrics(
                operation=operation,
                duration=duration,
                memory_usage=memory_usage,
                cache_hit=cache_hit,
                timestamp=datetime.utcnow(),
                session_id=session_id,
                optimization_applied=self.current_strategy.value
            )
            
            self.performance_history.append(metrics)
            
            # Update operation statistics
            stats = self.operation_stats[operation]
            stats['total_time'] += duration
            stats['count'] += 1
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    async def adapt_optimization_strategy(self):
        """Adaptively adjust optimization strategy based on performance"""
        try:
            if len(self.performance_history) < 100:
                return  # Need sufficient data
            
            # Analyze recent performance
            recent_metrics = list(self.performance_history)[-100:]
            current_performance = self._analyze_performance_metrics(recent_metrics)
            
            # Evaluate current strategy
            strategy_score = self._calculate_strategy_score(current_performance)
            self.strategy_performance[self.current_strategy]['score'] = strategy_score
            self.strategy_performance[self.current_strategy]['usage_count'] += 1
            
            # Check if we should switch strategies
            should_switch, best_strategy = await self._should_switch_strategy(current_performance)
            
            if should_switch and best_strategy != self.current_strategy:
                logger.info(f"Switching optimization strategy from {self.current_strategy.value} to {best_strategy.value}")
                self.current_strategy = best_strategy
            
        except Exception as e:
            logger.error(f"Error adapting optimization strategy: {e}")
    
    def _analyze_performance_metrics(self, metrics: List[PerformanceMetrics]) -> Dict[str, float]:
        """Analyze performance metrics to determine system state"""
        try:
            if not metrics:
                return {}
            
            # Calculate key performance indicators
            avg_duration = sum(m.duration for m in metrics) / len(metrics)
            cache_hit_rate = sum(1 for m in metrics if m.cache_hit) / len(metrics)
            avg_memory = sum(m.memory_usage for m in metrics) / len(metrics)
            
            # Calculate trends
            first_half = metrics[:len(metrics)//2]
            second_half = metrics[len(metrics)//2:]
            
            first_avg = sum(m.duration for m in first_half) / len(first_half) if first_half else 0
            second_avg = sum(m.duration for m in second_half) / len(second_half) if second_half else 0
            
            performance_trend = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
            
            return {
                'avg_duration': avg_duration,
                'cache_hit_rate': cache_hit_rate,
                'avg_memory_usage': avg_memory,
                'performance_trend': performance_trend,
                'total_operations': len(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return {}
    
    def _calculate_strategy_score(self, performance: Dict[str, float]) -> float:
        """Calculate score for current optimization strategy"""
        try:
            if not performance:
                return 0.5
            
            score = 1.0
            
            # Duration factor
            if performance['avg_duration'] > 1.0:  # More than 1 second
                score -= 0.3
            elif performance['avg_duration'] < 0.1:  # Less than 100ms
                score += 0.2
            
            # Cache hit rate factor
            if performance['cache_hit_rate'] > 0.8:
                score += 0.3
            elif performance['cache_hit_rate'] < 0.5:
                score -= 0.2
            
            # Memory usage factor
            if performance['avg_memory_usage'] > self.max_cache_size_bytes * 0.9:
                score -= 0.4
            
            # Performance trend factor
            if performance['performance_trend'] > 0.2:  # Getting slower
                score -= 0.3
            elif performance['performance_trend'] < -0.1:  # Getting faster
                score += 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    async def _should_switch_strategy(self, current_performance: Dict[str, float]) -> Tuple[bool, OptimizationStrategy]:
        """Determine if strategy should be switched and which one to use"""
        try:
            # Identify primary performance issues
            issues = []
            
            if current_performance.get('avg_duration', 0) > 0.5:
                issues.append('slow_performance')
            
            if current_performance.get('avg_memory_usage', 0) > self.max_cache_size_bytes * 0.8:
                issues.append('high_memory')
            
            if current_performance.get('cache_hit_rate', 1.0) < 0.6:
                issues.append('poor_cache_performance')
            
            if current_performance.get('performance_trend', 0) > 0.15:
                issues.append('degrading_performance')
            
            # Recommend strategy based on issues
            if 'high_memory' in issues:
                recommended = OptimizationStrategy.MEMORY
            elif 'slow_performance' in issues or 'degrading_performance' in issues:
                recommended = OptimizationStrategy.PERFORMANCE
            elif 'poor_cache_performance' in issues:
                recommended = OptimizationStrategy.BALANCED
            else:
                recommended = OptimizationStrategy.QUALITY
            
            # Only switch if recommended strategy is different and has potential
            should_switch = (recommended != self.current_strategy and 
                           self.strategy_performance[recommended]['usage_count'] < 10)
            
            return should_switch, recommended
            
        except Exception as e:
            logger.error(f"Error determining strategy switch: {e}")
            return False, self.current_strategy
    
    async def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        try:
            cache_stats = {
                'total_entries': len(self.cache),
                'total_size_mb': self.current_cache_size / (1024 * 1024),
                'utilization_percent': (self.current_cache_size / self.max_cache_size_bytes) * 100,
                'avg_entry_size_kb': (self.current_cache_size / len(self.cache) / 1024) if self.cache else 0
            }
            
            # Recent performance analysis
            recent_metrics = list(self.performance_history)[-1000:] if self.performance_history else []
            performance_analysis = self._analyze_performance_metrics(recent_metrics)
            
            # Strategy effectiveness
            strategy_effectiveness = {}
            for strategy, data in self.strategy_performance.items():
                strategy_effectiveness[strategy.value] = {
                    'score': data['score'],
                    'usage_count': data['usage_count'],
                    'effectiveness': 'high' if data['score'] > 0.8 else 'medium' if data['score'] > 0.6 else 'low'
                }
            
            return {
                'current_strategy': self.current_strategy.value,
                'cache_statistics': cache_stats,
                'performance_analysis': performance_analysis,
                'strategy_effectiveness': strategy_effectiveness,
                'optimization_parameters': dict(self.adaptive_params),
                'recommendations': await self._generate_optimization_recommendations(performance_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization statistics: {e}")
            return {'error': str(e)}
    
    async def _generate_optimization_recommendations(self, performance: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on performance"""
        recommendations = []
        
        try:
            if performance.get('avg_duration', 0) > 0.5:
                recommendations.append("Consider switching to PERFORMANCE strategy for faster response times")
            
            if performance.get('avg_memory_usage', 0) > self.max_cache_size_bytes * 0.8:
                recommendations.append("High memory usage detected. Consider MEMORY strategy or increase cache size")
            
            if performance.get('cache_hit_rate', 1.0) < 0.6:
                recommendations.append("Low cache hit rate. Consider adjusting cache retention policies")
            
            if performance.get('performance_trend', 0) > 0.1:
                recommendations.append("Performance degradation detected. Consider cache cleanup or strategy change")
            
            if not recommendations:
                recommendations.append("System performance is optimal. Current strategy is working well")
            
        except Exception:
            recommendations.append("Unable to generate recommendations due to insufficient data")
        
        return recommendations

class PerformanceMeasurement:
    """Context manager for measuring operation performance"""
    
    def __init__(self, optimizer: ContextOptimizationEngine, operation_name: str, session_id: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.session_id = session_id
        self.start_time = None
        self.start_memory = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        # Could add memory tracking here if needed
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            await self.optimizer._record_performance(
                self.operation_name, duration, self.session_id
            )
