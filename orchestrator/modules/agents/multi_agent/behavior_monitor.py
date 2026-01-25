
"""
Emergent Behavior Monitor
=========================

Advanced monitoring system for emergent behaviors in multi-agent systems with:
- Real-time behavior pattern detection
- Stability analysis and control
- Interaction strength measurement
- Behavioral anomaly detection

Mathematical foundation:
- E = f(Diversity, Interaction_Strength)
- Stability = min(ΔS_i)
- Diversity = Var(Agent_Behaviors)
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import asyncio
from collections import defaultdict, deque
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class BehaviorType(Enum):
    """Types of emergent behaviors"""
    COORDINATION = "coordination"
    COMPETITION = "competition"
    COLLABORATION = "collaboration"
    CONVERGENCE = "convergence"
    DIVERGENCE = "divergence"
    OSCILLATION = "oscillation"
    STABILIZATION = "stabilization"

@dataclass
class BehaviorPattern:
    """Individual behavior pattern"""
    pattern_id: str
    behavior_type: BehaviorType
    agents_involved: List[str]
    interaction_strength: float
    diversity_score: float
    stability_score: float
    duration: float
    confidence: float
    timestamp: datetime

@dataclass
class InteractionEvent:
    """Agent interaction event"""
    source_agent: str
    target_agent: str
    interaction_type: str
    strength: float
    context: Dict[str, Any]
    timestamp: datetime

class EmergentBehaviorMonitor:
    """
    Advanced behavior monitoring system for multi-agent environments
    """
    
    def __init__(self, history_window: int = 100):
        self.behavior_history = deque(maxlen=history_window)
        self.interaction_history = deque(maxlen=history_window * 2)
        self.agent_behaviors = defaultdict(list)
        self.behavior_patterns = {}
        
        # Monitoring parameters
        self.stability_threshold = 0.05
        self.diversity_threshold = 0.3
        self.interaction_threshold = 0.2
        self.anomaly_threshold = 2.0  # Standard deviations for anomaly detection
        
        # Pattern detection
        self.pattern_detector = None
        self.scaler = StandardScaler()
        
        # Performance metrics
        self.monitoring_metrics = {
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "stability_warnings": 0,
            "behavior_predictions": 0
        }
        
        logger.info("Emergent Behavior Monitor initialized")
    
    async def monitor_emergent_behavior(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Monitor and analyze emergent behaviors in agent interactions
        
        Args:
            session_id: Session identifier
            agents: List of participating agents
            interactions: Recent agent interactions
            context: Optional monitoring context
            
        Returns:
            Dict containing behavior analysis and recommendations
        """
        monitoring_start = time.time()
        
        try:
            # Process interactions
            processed_interactions = await self._process_interactions(
                session_id, agents, interactions
            )
            
            # Calculate behavior metrics
            diversity_score = self._calculate_diversity(agents, processed_interactions)
            interaction_strength = self._calculate_interaction_strength(processed_interactions)
            stability_score = self._calculate_stability(session_id, agents)
            
            # Detect emergent patterns
            detected_patterns = await self._detect_behavior_patterns(
                session_id, agents, processed_interactions, diversity_score, interaction_strength
            )
            
            # Analyze system stability
            stability_analysis = self._analyze_stability(
                session_id, agents, stability_score, detected_patterns
            )
            
            # Detect anomalies
            anomalies = self._detect_behavioral_anomalies(
                agents, processed_interactions, detected_patterns
            )
            
            # Generate behavior insights
            behavior_insights = self._generate_behavior_insights(
                detected_patterns, stability_analysis, anomalies
            )
            
            # Calculate overall behavior score
            behavior_score = self._calculate_behavior_score(
                diversity_score, interaction_strength, stability_score
            )
            
            # Create monitoring result
            monitoring_result = {
                "session_id": session_id,
                "agents": agents,
                "behavior_score": behavior_score,
                "diversity_score": diversity_score,
                "interaction_strength": interaction_strength,
                "stability_score": stability_score,
                "detected_patterns": [asdict(pattern) for pattern in detected_patterns],
                "stability_analysis": stability_analysis,
                "anomalies": anomalies,
                "behavior_insights": behavior_insights,
                "monitoring_time": time.time() - monitoring_start,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in behavior history
            self.behavior_history.append(monitoring_result)
            
            # Update metrics
            self.monitoring_metrics["patterns_detected"] += len(detected_patterns)
            self.monitoring_metrics["anomalies_detected"] += len(anomalies)
            if stability_score < self.stability_threshold:
                self.monitoring_metrics["stability_warnings"] += 1
            
            logger.info(
                f"Behavior monitoring completed for session {session_id}: "
                f"score={behavior_score:.3f}, patterns={len(detected_patterns)}, "
                f"anomalies={len(anomalies)}"
            )
            
            return monitoring_result
            
        except Exception as e:
            logger.error(f"Failed behavior monitoring for session {session_id}: {e}")
            raise
    
    async def _process_interactions(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[Dict[str, Any]]
    ) -> List[InteractionEvent]:
        """Process raw interaction data into structured events"""
        
        processed_interactions = []
        
        for interaction in interactions:
            try:
                # Extract interaction details
                source = interaction.get("source_agent", "unknown")
                target = interaction.get("target_agent", "unknown")
                interaction_type = interaction.get("type", "general")
                
                # Calculate interaction strength
                strength = interaction.get("strength", 0.5)
                if "confidence" in interaction:
                    strength = (strength + interaction["confidence"]) / 2
                if "importance" in interaction:
                    strength = (strength + interaction["importance"]) / 2
                
                # Create interaction event
                event = InteractionEvent(
                    source_agent=source,
                    target_agent=target,
                    interaction_type=interaction_type,
                    strength=strength,
                    context=interaction.get("context", {}),
                    timestamp=datetime.utcnow()
                )
                
                processed_interactions.append(event)
                
            except Exception as e:
                logger.warning(f"Failed to process interaction: {e}")
                continue
        
        # Store in interaction history
        self.interaction_history.extend(processed_interactions)
        
        return processed_interactions
    
    def _calculate_diversity(
        self,
        agents: List[str],
        interactions: List[InteractionEvent]
    ) -> float:
        """Calculate behavioral diversity: Diversity = Var(Agent_Behaviors)"""
        
        if not agents or not interactions:
            return 0.0
        
        # Calculate agent behavior vectors
        agent_vectors = {}
        
        for agent in agents:
            # Interaction frequency
            outgoing = len([i for i in interactions if i.source_agent == agent])
            incoming = len([i for i in interactions if i.target_agent == agent])
            
            # Interaction strength
            out_strength = np.mean([i.strength for i in interactions if i.source_agent == agent] or [0])
            in_strength = np.mean([i.strength for i in interactions if i.target_agent == agent] or [0])
            
            # Interaction type diversity
            out_types = len(set(i.interaction_type for i in interactions if i.source_agent == agent))
            in_types = len(set(i.interaction_type for i in interactions if i.target_agent == agent))
            
            agent_vectors[agent] = np.array([
                outgoing, incoming, out_strength, in_strength, out_types, in_types
            ])
        
        # Calculate diversity as variance across agent vectors
        if len(agent_vectors) < 2:
            return 0.0
        
        all_vectors = list(agent_vectors.values())
        vector_matrix = np.vstack(all_vectors)
        
        # Normalize vectors
        if vector_matrix.std() > 0:
            normalized_matrix = (vector_matrix - vector_matrix.mean(axis=0)) / vector_matrix.std(axis=0)
            diversity = np.mean(np.var(normalized_matrix, axis=0))
        else:
            diversity = 0.0
        
        return min(diversity, 1.0)  # Normalize to [0, 1]
    
    def _calculate_interaction_strength(
        self,
        interactions: List[InteractionEvent]
    ) -> float:
        """Calculate overall interaction strength across agents"""
        
        if not interactions:
            return 0.0
        
        # Base interaction strength
        base_strength = np.mean([i.strength for i in interactions])
        
        # Interaction frequency factor
        unique_pairs = set((i.source_agent, i.target_agent) for i in interactions)
        frequency_factor = min(len(interactions) / (len(unique_pairs) or 1), 2.0) / 2.0
        
        # Bidirectional interaction bonus
        bidirectional_count = 0
        for source, target in unique_pairs:
            reverse_exists = (target, source) in unique_pairs
            if reverse_exists:
                bidirectional_count += 1
        
        bidirectional_factor = bidirectional_count / (len(unique_pairs) or 1)
        
        # Combined interaction strength
        interaction_strength = (
            0.5 * base_strength +
            0.3 * frequency_factor +
            0.2 * bidirectional_factor
        )
        
        return min(interaction_strength, 1.0)
    
    def _calculate_stability(
        self,
        session_id: str,
        agents: List[str]
    ) -> float:
        """Calculate system stability: Stability = min(ΔS_i)"""
        
        # Get recent behavior history for this session or similar contexts
        recent_behaviors = [
            b for b in self.behavior_history 
            if b["session_id"] == session_id or set(b["agents"]) == set(agents)
        ][-5:]  # Last 5 relevant behaviors
        
        if len(recent_behaviors) < 2:
            return 0.5  # Default stability for insufficient history
        
        # Calculate stability metrics
        behavior_deltas = []
        
        for i in range(1, len(recent_behaviors)):
            prev_behavior = recent_behaviors[i-1]
            curr_behavior = recent_behaviors[i]
            
            # Compare key metrics
            diversity_delta = abs(curr_behavior["diversity_score"] - prev_behavior["diversity_score"])
            strength_delta = abs(curr_behavior["interaction_strength"] - prev_behavior["interaction_strength"])
            score_delta = abs(curr_behavior["behavior_score"] - prev_behavior["behavior_score"])
            
            # Combined delta
            combined_delta = (diversity_delta + strength_delta + score_delta) / 3.0
            behavior_deltas.append(combined_delta)
        
        # Stability = 1 - max_delta (lower deltas = higher stability)
        if behavior_deltas:
            max_delta = max(behavior_deltas)
            stability = max(0.0, 1.0 - max_delta)
        else:
            stability = 0.5
        
        return stability
    
    async def _detect_behavior_patterns(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[InteractionEvent],
        diversity_score: float,
        interaction_strength: float
    ) -> List[BehaviorPattern]:
        """Detect emergent behavior patterns using clustering and analysis"""
        
        detected_patterns = []
        
        try:
            # Create behavior feature matrix
            features = await self._extract_behavior_features(agents, interactions)
            
            if len(features) < 3:  # Need minimum features for clustering
                return detected_patterns
            
            # Apply clustering to detect patterns
            if len(features) >= 3:
                n_clusters = min(3, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                # Normalize features
                normalized_features = self.scaler.fit_transform(features)
                clusters = kmeans.fit_predict(normalized_features)
                
                # Analyze each cluster as a potential pattern
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    
                    if len(cluster_indices) >= 2:  # Pattern needs multiple agents
                        pattern = await self._analyze_pattern_cluster(
                            session_id, agents, interactions, cluster_indices, 
                            diversity_score, interaction_strength
                        )
                        
                        if pattern:
                            detected_patterns.append(pattern)
            
            # Detect specific behavior types
            specific_patterns = await self._detect_specific_patterns(
                session_id, agents, interactions, diversity_score, interaction_strength
            )
            detected_patterns.extend(specific_patterns)
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
        
        return detected_patterns
    
    async def _extract_behavior_features(
        self,
        agents: List[str],
        interactions: List[InteractionEvent]
    ) -> np.ndarray:
        """Extract behavior features for pattern detection"""
        
        features = []
        
        for agent in agents:
            # Interaction metrics
            agent_interactions = [i for i in interactions if i.source_agent == agent or i.target_agent == agent]
            
            if agent_interactions:
                # Feature vector for each agent
                avg_strength = np.mean([i.strength for i in agent_interactions])
                interaction_count = len(agent_interactions)
                type_diversity = len(set(i.interaction_type for i in agent_interactions))
                
                # Centrality measures (simplified)
                outgoing = len([i for i in interactions if i.source_agent == agent])
                incoming = len([i for i in interactions if i.target_agent == agent])
                
                features.append([
                    avg_strength,
                    interaction_count,
                    type_diversity,
                    outgoing,
                    incoming,
                    outgoing / (incoming + 1)  # Outgoing/incoming ratio
                ])
            else:
                # Default features for agents with no interactions
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        return np.array(features)
    
    async def _analyze_pattern_cluster(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[InteractionEvent],
        cluster_indices: np.ndarray,
        diversity_score: float,
        interaction_strength: float
    ) -> Optional[BehaviorPattern]:
        """Analyze a behavior cluster to identify pattern"""
        
        try:
            cluster_agents = [agents[i] for i in cluster_indices]
            
            # Analyze cluster characteristics
            cluster_interactions = [
                i for i in interactions 
                if i.source_agent in cluster_agents or i.target_agent in cluster_agents
            ]
            
            if not cluster_interactions:
                return None
            
            # Determine behavior type
            behavior_type = self._classify_behavior_type(
                cluster_agents, cluster_interactions, diversity_score
            )
            
            # Calculate pattern metrics
            pattern_strength = np.mean([i.strength for i in cluster_interactions])
            pattern_diversity = len(set(i.interaction_type for i in cluster_interactions)) / 5.0  # Normalize
            
            # Calculate stability for this pattern
            pattern_stability = self._calculate_pattern_stability(cluster_agents, cluster_interactions)
            
            # Pattern confidence based on cluster cohesion
            confidence = min(pattern_strength * pattern_diversity * len(cluster_agents) / len(agents), 1.0)
            
            pattern_id = f"pattern_{session_id}_{len(self.behavior_patterns)}"
            
            pattern = BehaviorPattern(
                pattern_id=pattern_id,
                behavior_type=behavior_type,
                agents_involved=cluster_agents,
                interaction_strength=pattern_strength,
                diversity_score=pattern_diversity,
                stability_score=pattern_stability,
                duration=0.0,  # Will be updated with tracking
                confidence=confidence,
                timestamp=datetime.utcnow()
            )
            
            # Store pattern for tracking
            self.behavior_patterns[pattern_id] = pattern
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Cluster analysis failed: {e}")
            return None
    
    def _classify_behavior_type(
        self,
        agents: List[str],
        interactions: List[InteractionEvent],
        diversity_score: float
    ) -> BehaviorType:
        """Classify the type of emergent behavior"""
        
        if not interactions:
            return BehaviorType.STABILIZATION
        
        # Analyze interaction patterns
        interaction_counts = defaultdict(int)
        for interaction in interactions:
            interaction_counts[interaction.interaction_type] += 1
        
        # Bidirectional vs unidirectional
        pairs = [(i.source_agent, i.target_agent) for i in interactions]
        bidirectional_count = 0
        for source, target in set(pairs):
            if (target, source) in pairs:
                bidirectional_count += 1
        
        bidirectional_ratio = bidirectional_count / (len(set(pairs)) or 1)
        
        # Classification logic
        if diversity_score > 0.7:
            if bidirectional_ratio > 0.6:
                return BehaviorType.COLLABORATION
            else:
                return BehaviorType.DIVERGENCE
        elif diversity_score < 0.3:
            if bidirectional_ratio > 0.5:
                return BehaviorType.COORDINATION
            else:
                return BehaviorType.CONVERGENCE
        else:
            # Medium diversity - check interaction strength patterns
            strengths = [i.strength for i in interactions]
            strength_variance = np.var(strengths) if strengths else 0
            
            if strength_variance > 0.2:
                return BehaviorType.OSCILLATION
            elif bidirectional_ratio > 0.4:
                return BehaviorType.COORDINATION
            else:
                return BehaviorType.COMPETITION
    
    def _calculate_pattern_stability(
        self,
        agents: List[str],
        interactions: List[InteractionEvent]
    ) -> float:
        """Calculate stability for a specific behavior pattern"""
        
        if not interactions:
            return 0.0
        
        # Analyze interaction consistency over time
        interaction_strengths = [i.strength for i in interactions]
        
        if len(interaction_strengths) < 2:
            return 0.5
        
        # Coefficient of variation (stability measure)
        mean_strength = np.mean(interaction_strengths)
        std_strength = np.std(interaction_strengths)
        
        if mean_strength == 0:
            return 0.0
        
        stability = max(0.0, 1.0 - (std_strength / mean_strength))
        return stability
    
    async def _detect_specific_patterns(
        self,
        session_id: str,
        agents: List[str],
        interactions: List[InteractionEvent],
        diversity_score: float,
        interaction_strength: float
    ) -> List[BehaviorPattern]:
        """Detect specific known behavior patterns"""
        
        specific_patterns = []
        
        # Leader-follower pattern
        leader_pattern = self._detect_leader_follower_pattern(agents, interactions)
        if leader_pattern:
            specific_patterns.append(leader_pattern)
        
        # Echo chamber pattern  
        echo_pattern = self._detect_echo_chamber_pattern(agents, interactions)
        if echo_pattern:
            specific_patterns.append(echo_pattern)
        
        # Competitive pattern
        competitive_pattern = self._detect_competitive_pattern(agents, interactions)
        if competitive_pattern:
            specific_patterns.append(competitive_pattern)
        
        return specific_patterns
    
    def _detect_leader_follower_pattern(
        self,
        agents: List[str], 
        interactions: List[InteractionEvent]
    ) -> Optional[BehaviorPattern]:
        """Detect leader-follower behavior pattern"""
        
        # Calculate centrality scores
        centrality_scores = defaultdict(float)
        
        for interaction in interactions:
            centrality_scores[interaction.source_agent] += interaction.strength
            centrality_scores[interaction.target_agent] += interaction.strength * 0.5
        
        if not centrality_scores:
            return None
        
        # Identify potential leader (highest centrality)
        leader = max(centrality_scores, key=centrality_scores.get)
        leader_score = centrality_scores[leader]
        avg_score = np.mean(list(centrality_scores.values()))
        
        # Leader must be significantly above average
        if leader_score > avg_score * 1.5:
            pattern = BehaviorPattern(
                pattern_id=f"leader_follower_{int(time.time())}",
                behavior_type=BehaviorType.COORDINATION,
                agents_involved=agents,
                interaction_strength=leader_score / len(agents),
                diversity_score=np.var(list(centrality_scores.values())) / avg_score if avg_score > 0 else 0,
                stability_score=0.7,  # Leader patterns tend to be stable
                duration=0.0,
                confidence=min(leader_score / (avg_score * 2), 1.0),
                timestamp=datetime.utcnow()
            )
            return pattern
        
        return None
    
    def _detect_echo_chamber_pattern(
        self,
        agents: List[str],
        interactions: List[InteractionEvent]
    ) -> Optional[BehaviorPattern]:
        """Detect echo chamber (self-reinforcing) pattern"""
        
        # Look for circular interaction patterns
        interaction_graph = nx.DiGraph()
        
        for interaction in interactions:
            interaction_graph.add_edge(
                interaction.source_agent,
                interaction.target_agent,
                weight=interaction.strength
            )
        
        # Find cycles (echo chambers)
        try:
            cycles = list(nx.simple_cycles(interaction_graph))
            
            if cycles and len(cycles[0]) >= 3:  # Need at least 3 agents for echo chamber
                # Analyze largest cycle
                largest_cycle = max(cycles, key=len)
                
                # Calculate cycle strength
                cycle_strength = 0.0
                for i in range(len(largest_cycle)):
                    source = largest_cycle[i]
                    target = largest_cycle[(i + 1) % len(largest_cycle)]
                    
                    if interaction_graph.has_edge(source, target):
                        cycle_strength += interaction_graph[source][target]["weight"]
                
                cycle_strength /= len(largest_cycle)
                
                if cycle_strength > 0.5:  # Strong echo chamber
                    pattern = BehaviorPattern(
                        pattern_id=f"echo_chamber_{int(time.time())}",
                        behavior_type=BehaviorType.CONVERGENCE,
                        agents_involved=largest_cycle,
                        interaction_strength=cycle_strength,
                        diversity_score=0.2,  # Echo chambers reduce diversity
                        stability_score=0.8,  # Very stable once formed
                        duration=0.0,
                        confidence=cycle_strength,
                        timestamp=datetime.utcnow()
                    )
                    return pattern
        
        except nx.NetworkXError:
            pass
        
        return None
    
    def _detect_competitive_pattern(
        self,
        agents: List[str],
        interactions: List[InteractionEvent]
    ) -> Optional[BehaviorPattern]:
        """Detect competitive behavior pattern"""
        
        # Look for asymmetric, high-strength interactions
        competitive_interactions = []
        
        for interaction in interactions:
            # Check if reverse interaction exists with different strength
            reverse_interactions = [
                i for i in interactions 
                if i.source_agent == interaction.target_agent and 
                   i.target_agent == interaction.source_agent
            ]
            
            if reverse_interactions:
                reverse_strength = np.mean([i.strength for i in reverse_interactions])
                strength_diff = abs(interaction.strength - reverse_strength)
                
                if strength_diff > 0.3 and interaction.strength > 0.6:
                    competitive_interactions.append(interaction)
        
        if len(competitive_interactions) >= len(agents) // 2:  # Significant competition
            avg_strength = np.mean([i.strength for i in competitive_interactions])
            
            pattern = BehaviorPattern(
                pattern_id=f"competitive_{int(time.time())}",
                behavior_type=BehaviorType.COMPETITION,
                agents_involved=agents,
                interaction_strength=avg_strength,
                diversity_score=0.8,  # Competition increases diversity
                stability_score=0.4,  # Competition can be unstable
                duration=0.0,
                confidence=len(competitive_interactions) / len(agents),
                timestamp=datetime.utcnow()
            )
            return pattern
        
        return None
    
    def _analyze_stability(
        self,
        session_id: str,
        agents: List[str],
        stability_score: float,
        patterns: List[BehaviorPattern]
    ) -> Dict[str, Any]:
        """Analyze system stability and provide recommendations"""
        
        stability_analysis = {
            "current_stability": stability_score,
            "stability_level": self._classify_stability_level(stability_score),
            "stability_trends": self._analyze_stability_trends(session_id, agents),
            "risk_factors": [],
            "recommendations": []
        }
        
        # Identify risk factors
        if stability_score < self.stability_threshold:
            stability_analysis["risk_factors"].append("Low system stability detected")
            stability_analysis["recommendations"].append("Consider reducing agent interaction frequency")
        
        # Pattern-based stability analysis
        unstable_patterns = [p for p in patterns if p.stability_score < 0.5]
        if unstable_patterns:
            stability_analysis["risk_factors"].append(f"{len(unstable_patterns)} unstable patterns detected")
            stability_analysis["recommendations"].append("Monitor unstable patterns closely")
        
        # Oscillation detection
        oscillating_patterns = [p for p in patterns if p.behavior_type == BehaviorType.OSCILLATION]
        if oscillating_patterns:
            stability_analysis["risk_factors"].append("Oscillating behavior patterns detected")
            stability_analysis["recommendations"].append("Implement damping mechanisms")
        
        # High diversity risk
        high_diversity_patterns = [p for p in patterns if p.diversity_score > 0.8]
        if len(high_diversity_patterns) > len(patterns) // 2:
            stability_analysis["risk_factors"].append("High behavioral diversity may lead to instability")
            stability_analysis["recommendations"].append("Consider coordination mechanisms")
        
        return stability_analysis
    
    def _classify_stability_level(self, stability_score: float) -> str:
        """Classify stability level"""
        if stability_score > 0.8:
            return "high"
        elif stability_score > 0.6:
            return "moderate"
        elif stability_score > 0.4:
            return "low"
        else:
            return "critical"
    
    def _analyze_stability_trends(
        self,
        session_id: str,
        agents: List[str]
    ) -> Dict[str, Any]:
        """Analyze stability trends over time"""
        
        # Get recent stability scores
        recent_behaviors = [
            b for b in self.behavior_history 
            if b["session_id"] == session_id or set(b["agents"]) == set(agents)
        ][-10:]  # Last 10 behaviors
        
        if len(recent_behaviors) < 3:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        stability_scores = [b["stability_score"] for b in recent_behaviors]
        
        # Calculate trend slope
        x = np.arange(len(stability_scores))
        slope = np.polyfit(x, stability_scores, 1)[0]
        
        # Classify trend
        if slope > 0.1:
            trend = "improving"
        elif slope < -0.1:
            trend = "degrading"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "recent_scores": stability_scores,
            "volatility": np.std(stability_scores)
        }
    
    def _detect_behavioral_anomalies(
        self,
        agents: List[str],
        interactions: List[InteractionEvent],
        patterns: List[BehaviorPattern]
    ) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies using statistical methods"""
        
        anomalies = []
        
        try:
            # Anomaly detection based on interaction strength
            strengths = [i.strength for i in interactions] if interactions else [0.5]
            
            if len(strengths) > 3:
                mean_strength = np.mean(strengths)
                std_strength = np.std(strengths)
                
                # Z-score based anomaly detection
                for i, interaction in enumerate(interactions):
                    if std_strength > 0:
                        z_score = abs((interaction.strength - mean_strength) / std_strength)
                        
                        if z_score > self.anomaly_threshold:
                            anomalies.append({
                                "type": "interaction_strength_anomaly",
                                "description": f"Unusual interaction strength: {interaction.strength:.3f}",
                                "agents": [interaction.source_agent, interaction.target_agent],
                                "z_score": z_score,
                                "severity": "high" if z_score > 3.0 else "medium"
                            })
            
            # Pattern-based anomalies
            if patterns:
                pattern_confidences = [p.confidence for p in patterns]
                
                if len(pattern_confidences) > 2:
                    mean_confidence = np.mean(pattern_confidences)
                    std_confidence = np.std(pattern_confidences)
                    
                    low_confidence_patterns = [
                        p for p in patterns 
                        if std_confidence > 0 and 
                           (mean_confidence - p.confidence) / std_confidence > self.anomaly_threshold
                    ]
                    
                    for pattern in low_confidence_patterns:
                        anomalies.append({
                            "type": "low_confidence_pattern",
                            "description": f"Low confidence behavior pattern: {pattern.behavior_type.value}",
                            "agents": pattern.agents_involved,
                            "confidence": pattern.confidence,
                            "severity": "medium"
                        })
            
            # Agent isolation anomaly
            agent_interaction_counts = defaultdict(int)
            for interaction in interactions:
                agent_interaction_counts[interaction.source_agent] += 1
                agent_interaction_counts[interaction.target_agent] += 1
            
            avg_interactions = np.mean(list(agent_interaction_counts.values())) if agent_interaction_counts else 0
            
            for agent in agents:
                if agent_interaction_counts[agent] < avg_interactions * 0.3:  # Less than 30% of average
                    anomalies.append({
                        "type": "agent_isolation",
                        "description": f"Agent {agent} has very low interaction frequency",
                        "agents": [agent],
                        "interaction_count": agent_interaction_counts[agent],
                        "severity": "medium"
                    })
        
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _generate_behavior_insights(
        self,
        patterns: List[BehaviorPattern],
        stability_analysis: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate actionable insights from behavior analysis"""
        
        insights = {
            "summary": "",
            "key_findings": [],
            "recommendations": [],
            "risk_level": "low"
        }
        
        # Pattern insights
        if patterns:
            dominant_behavior = max(patterns, key=lambda p: p.confidence).behavior_type.value
            insights["key_findings"].append(f"Dominant behavior pattern: {dominant_behavior}")
            
            collaborative_patterns = len([p for p in patterns if p.behavior_type in [BehaviorType.COLLABORATION, BehaviorType.COORDINATION]])
            if collaborative_patterns > len(patterns) // 2:
                insights["key_findings"].append("High collaboration detected")
                insights["recommendations"].append("Leverage collaborative momentum")
            
            unstable_patterns = len([p for p in patterns if p.stability_score < 0.5])
            if unstable_patterns > 0:
                insights["key_findings"].append(f"{unstable_patterns} unstable patterns need attention")
                insights["risk_level"] = "medium"
        
        # Stability insights
        stability_level = stability_analysis["stability_level"]
        insights["key_findings"].append(f"System stability: {stability_level}")
        
        if stability_level in ["low", "critical"]:
            insights["risk_level"] = "high"
            insights["recommendations"].append("Immediate stability intervention needed")
        
        # Anomaly insights
        if anomalies:
            high_severity_anomalies = len([a for a in anomalies if a["severity"] == "high"])
            insights["key_findings"].append(f"{len(anomalies)} behavioral anomalies detected")
            
            if high_severity_anomalies > 0:
                insights["risk_level"] = "high"
                insights["recommendations"].append("Investigate high-severity anomalies")
        
        # Generate summary
        risk_descriptor = {"low": "stable", "medium": "concerning", "high": "critical"}[insights["risk_level"]]
        insights["summary"] = f"System shows {risk_descriptor} behavior with {len(patterns)} patterns and {len(anomalies)} anomalies"
        
        return insights
    
    def _calculate_behavior_score(
        self,
        diversity_score: float,
        interaction_strength: float,
        stability_score: float
    ) -> float:
        """Calculate overall behavior score: E = f(Diversity, Interaction_Strength, Stability)"""
        
        # Weighted combination of behavior metrics
        behavior_score = (
            0.3 * diversity_score +      # Diversity contributes to emergence
            0.4 * interaction_strength +  # Strong interactions enable emergence
            0.3 * stability_score         # Stability maintains emergence
        )
        
        return min(behavior_score, 1.0)
    
    async def log_behavior(
        self,
        session_id: str,
        behavior: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log individual behavior event for continuous monitoring"""
        
        try:
            # Store behavior in agent-specific history
            agent = behavior.get("agent", "unknown")
            self.agent_behaviors[agent].append({
                **behavior,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Trim history if too large
            if len(self.agent_behaviors[agent]) > 100:
                self.agent_behaviors[agent] = self.agent_behaviors[agent][-100:]
            
            return {
                "status": "logged",
                "agent": agent,
                "session_id": session_id,
                "behavior_count": len(self.agent_behaviors[agent])
            }
            
        except Exception as e:
            logger.error(f"Failed to log behavior: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        
        if not self.behavior_history:
            return {
                "message": "No monitoring history available",
                "metrics": self.monitoring_metrics
            }
        
        # Aggregate statistics
        recent_behaviors = list(self.behavior_history)[-20:]  # Last 20 behaviors
        
        avg_behavior_score = np.mean([b["behavior_score"] for b in recent_behaviors])
        avg_diversity = np.mean([b["diversity_score"] for b in recent_behaviors])
        avg_interaction_strength = np.mean([b["interaction_strength"] for b in recent_behaviors])
        avg_stability = np.mean([b["stability_score"] for b in recent_behaviors])
        
        # Pattern statistics
        all_patterns = []
        for behavior in recent_behaviors:
            all_patterns.extend(behavior.get("detected_patterns", []))
        
        pattern_types = defaultdict(int)
        for pattern in all_patterns:
            pattern_types[pattern["behavior_type"]] += 1
        
        # Agent activity
        agent_activity = defaultdict(int)
        for behavior in recent_behaviors:
            for agent in behavior.get("agents", []):
                agent_activity[agent] += 1
        
        return {
            "total_monitoring_sessions": len(self.behavior_history),
            "recent_sessions": len(recent_behaviors),
            "average_behavior_score": avg_behavior_score,
            "average_diversity_score": avg_diversity,
            "average_interaction_strength": avg_interaction_strength,
            "average_stability_score": avg_stability,
            "pattern_type_distribution": dict(pattern_types),
            "agent_activity": dict(agent_activity),
            "monitoring_metrics": self.monitoring_metrics,
            "active_patterns": len(self.behavior_patterns),
            "monitoring_parameters": {
                "stability_threshold": self.stability_threshold,
                "diversity_threshold": self.diversity_threshold,
                "interaction_threshold": self.interaction_threshold,
                "anomaly_threshold": self.anomaly_threshold
            }
        }
