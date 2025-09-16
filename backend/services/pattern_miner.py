
"""
Pattern Mining Service - Discovers Successful Workflows from Telemetry
Analyzes run traces to identify successful patterns and create reusable playbooks.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from database import get_db_session
from models.run_telemetry import RunTrace, RunEvent
from models.playbooks import Playbook


@dataclass
class Pattern:
    """Represents a discovered workflow pattern."""
    pattern_id: str
    task_type: str
    steps: List[Dict[str, Any]]
    success_rate: float
    avg_duration_ms: int
    total_runs: int
    avg_cost: float
    context_requirements: Dict[str, Any]
    tools_used: List[str]
    confidence_score: float


class PatternMiner:
    """Service for mining successful patterns from agent execution telemetry."""
    
    def __init__(self):
        self.min_runs_for_pattern = 5  # Minimum runs to consider a pattern
        self.min_success_rate = 0.8    # Minimum success rate to consider pattern
        self.similarity_threshold = 0.8  # Step sequence similarity threshold
    
    def mine_patterns(
        self,
        task_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        days_back: int = 30
    ) -> List[Pattern]:
        """Mine patterns from successful runs."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Get successful runs
            query = db.query(RunTrace).filter(
                and_(
                    RunTrace.success == True,
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.duration_ms.isnot(None)
                )
            )
            
            if task_type:
                query = query.filter(RunTrace.task_type == task_type)
            if agent_id:
                query = query.filter(RunTrace.agent_id == agent_id)
            
            successful_runs = query.all()
            
            if len(successful_runs) < self.min_runs_for_pattern:
                return []
            
            # Group runs by task type
            runs_by_task = defaultdict(list)
            for run in successful_runs:
                runs_by_task[run.task_type].append(run)
            
            patterns = []
            for task_type, runs in runs_by_task.items():
                if len(runs) >= self.min_runs_for_pattern:
                    task_patterns = self._mine_patterns_for_task(task_type, runs, db)
                    patterns.extend(task_patterns)
            
            return patterns
    
    def _mine_patterns_for_task(self, task_type: str, runs: List[RunTrace], db: Session) -> List[Pattern]:
        """Mine patterns for a specific task type."""
        # Get step sequences for each run
        run_sequences = []
        
        for run in runs:
            events = db.query(RunEvent).filter(
                RunEvent.run_id == run.run_id
            ).order_by(RunEvent.timestamp).all()
            
            sequence = self._extract_step_sequence(events)
            if sequence:
                run_sequences.append({
                    'run': run,
                    'sequence': sequence,
                    'events': events
                })
        
        if not run_sequences:
            return []
        
        # Cluster similar sequences
        pattern_clusters = self._cluster_sequences(run_sequences)
        
        # Create patterns from clusters
        patterns = []
        for cluster_idx, cluster in enumerate(pattern_clusters):
            if len(cluster) >= self.min_runs_for_pattern:
                pattern = self._create_pattern_from_cluster(
                    task_type, cluster, cluster_idx
                )
                if pattern.success_rate >= self.min_success_rate:
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_step_sequence(self, events: List[RunEvent]) -> List[Dict[str, Any]]:
        """Extract step sequence from run events."""
        steps = []
        current_step = None
        
        for event in events:
            if event.event_type == 'step_start':
                current_step = {
                    'name': event.step_name,
                    'start_time': event.timestamp,
                    'data': event.event_data or {}
                }
            elif event.event_type == 'step_end' and current_step:
                if event.step_name == current_step['name']:
                    current_step['end_time'] = event.timestamp
                    current_step['duration_ms'] = int(
                        (event.timestamp - current_step['start_time']).total_seconds() * 1000
                    )
                    current_step['success'] = event.event_data.get('success', True)
                    steps.append(current_step)
                    current_step = None
            elif event.event_type == 'tool_usage':
                # Add tool usage to current step or as standalone
                tool_info = {
                    'type': 'tool_usage',
                    'tool_name': event.event_data.get('tool_name'),
                    'cost': event.event_data.get('cost', 0),
                    'timestamp': event.timestamp
                }
                
                if current_step:
                    current_step.setdefault('tools', []).append(tool_info)
                else:
                    steps.append(tool_info)
        
        return steps
    
    def _cluster_sequences(self, run_sequences: List[Dict]) -> List[List[Dict]]:
        """Cluster similar step sequences."""
        clusters = []
        
        for run_seq in run_sequences:
            # Find best matching cluster
            best_cluster_idx = None
            best_similarity = 0
            
            for idx, cluster in enumerate(clusters):
                # Calculate similarity with cluster representative
                representative = cluster[0]
                similarity = self._calculate_sequence_similarity(
                    run_seq['sequence'],
                    representative['sequence']
                )
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster_idx = idx
            
            # Add to best cluster or create new one
            if best_cluster_idx is not None:
                clusters[best_cluster_idx].append(run_seq)
            else:
                clusters.append([run_seq])
        
        return clusters
    
    def _calculate_sequence_similarity(self, seq1: List[Dict], seq2: List[Dict]) -> float:
        """Calculate similarity between two step sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        # Extract step names for comparison
        names1 = [step.get('name') for step in seq1 if step.get('name')]
        names2 = [step.get('name') for step in seq2 if step.get('name')]
        
        if not names1 or not names2:
            return 0.0
        
        # Use longest common subsequence
        lcs_length = self._longest_common_subsequence(names1, names2)
        max_length = max(len(names1), len(names2))
        
        return lcs_length / max_length if max_length > 0 else 0.0
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _create_pattern_from_cluster(self, task_type: str, cluster: List[Dict], cluster_idx: int) -> Pattern:
        """Create a pattern from a cluster of similar runs."""
        runs = [item['run'] for item in cluster]
        
        # Calculate statistics
        total_runs = len(runs)
        successful_runs = sum(1 for run in runs if run.success)
        success_rate = successful_runs / total_runs if total_runs > 0 else 0
        
        durations = [run.duration_ms for run in runs if run.duration_ms]
        avg_duration_ms = int(sum(durations) / len(durations)) if durations else 0
        
        # Extract common step pattern
        representative_sequence = cluster[0]['sequence']
        common_steps = self._extract_common_steps(cluster)
        
        # Extract tools used
        all_tools = set()
        total_cost = 0
        for item in cluster:
            for step in item['sequence']:
                if step.get('tools'):
                    for tool in step['tools']:
                        all_tools.add(tool.get('tool_name'))
                        total_cost += tool.get('cost', 0)
        
        avg_cost = total_cost / total_runs if total_runs > 0 else 0
        
        # Extract context requirements
        context_requirements = self._extract_context_requirements(runs)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            success_rate, total_runs, len(common_steps)
        )
        
        return Pattern(
            pattern_id=f"{task_type}_{cluster_idx}_{datetime.utcnow().strftime('%Y%m%d')}",
            task_type=task_type,
            steps=common_steps,
            success_rate=success_rate,
            avg_duration_ms=avg_duration_ms,
            total_runs=total_runs,
            avg_cost=avg_cost,
            context_requirements=context_requirements,
            tools_used=list(all_tools),
            confidence_score=confidence_score
        )
    
    def _extract_common_steps(self, cluster: List[Dict]) -> List[Dict[str, Any]]:
        """Extract common steps from cluster."""
        # Use the most frequent sequence as template
        step_sequences = [item['sequence'] for item in cluster]
        
        if not step_sequences:
            return []
        
        # Find most common sequence structure
        representative = step_sequences[0]
        
        common_steps = []
        for step in representative:
            if step.get('name'):
                # Calculate frequency of this step across cluster
                frequency = sum(
                    1 for seq in step_sequences
                    if any(s.get('name') == step.get('name') for s in seq)
                ) / len(step_sequences)
                
                common_steps.append({
                    'name': step.get('name'),
                    'frequency': frequency,
                    'avg_duration_ms': step.get('duration_ms', 0),
                    'tools': step.get('tools', []),
                    'data': step.get('data', {})
                })
        
        return common_steps
    
    def _extract_context_requirements(self, runs: List[RunTrace]) -> Dict[str, Any]:
        """Extract common context requirements from runs."""
        context_keys = set()
        context_types = Counter()
        
        for run in runs:
            if run.context_data:
                context_keys.update(run.context_data.keys())
                # Infer context types from data
                for key, value in run.context_data.items():
                    if isinstance(value, str) and len(value) > 100:
                        context_types[f"{key}:document"] += 1
                    elif isinstance(value, list):
                        context_types[f"{key}:list"] += 1
                    elif isinstance(value, dict):
                        context_types[f"{key}:object"] += 1
        
        return {
            'required_keys': list(context_keys),
            'common_types': dict(context_types.most_common(10))
        }
    
    def _calculate_confidence_score(self, success_rate: float, total_runs: int, num_steps: int) -> float:
        """Calculate confidence score for pattern."""
        # Factors: success rate, sample size, pattern complexity
        base_confidence = success_rate
        
        # Boost confidence for larger sample sizes
        sample_boost = min(0.2, (total_runs - self.min_runs_for_pattern) * 0.02)
        
        # Slight penalty for overly complex patterns
        complexity_penalty = max(0, (num_steps - 10) * 0.01)
        
        confidence = base_confidence + sample_boost - complexity_penalty
        return min(1.0, max(0.0, confidence))
    
    def create_playbook_from_pattern(self, pattern: Pattern, db: Session) -> Playbook:
        """Create a playbook from a discovered pattern."""
        playbook_data = {
            'name': f"Auto-generated: {pattern.task_type}",
            'description': f"Discovered pattern for {pattern.task_type} with {pattern.success_rate:.1%} success rate",
            'task_type': pattern.task_type,
            'steps': pattern.steps,
            'context_requirements': pattern.context_requirements,
            'tools_required': pattern.tools_used,
            'estimated_cost': pattern.avg_cost,
            'estimated_duration_ms': pattern.avg_duration_ms,
            'success_rate': pattern.success_rate,
            'confidence_score': pattern.confidence_score,
            'based_on_runs': pattern.total_runs,
            'created_from_pattern': pattern.pattern_id
        }
        
        playbook = Playbook(
            name=playbook_data['name'],
            description=playbook_data['description'],
            workflow_data=playbook_data,
            status='discovered',
            created_by='pattern_miner',
            created_at=datetime.utcnow()
        )
        
        db.add(playbook)
        db.commit()
        db.refresh(playbook)
        
        return playbook
    
    def get_pattern_performance(self, pattern_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Get performance metrics for a specific pattern."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with get_db_session() as db:
            # Find runs that match this pattern
            runs = db.query(RunTrace).filter(
                and_(
                    RunTrace.start_time >= cutoff_date,
                    RunTrace.metadata.contains({"pattern_id": pattern_id})
                )
            ).all()
            
            if not runs:
                return {"error": "No runs found for pattern"}
            
            total_runs = len(runs)
            successful_runs = sum(1 for run in runs if run.success)
            
            return {
                'pattern_id': pattern_id,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
                'avg_duration_ms': sum(run.duration_ms for run in runs if run.duration_ms) // total_runs if runs else 0,
                'total_cost': sum(run.metadata.get('cost', 0) for run in runs),
                'last_run': max(run.start_time for run in runs).isoformat() if runs else None
            }


# Global pattern miner instance
pattern_miner = PatternMiner()


# Convenience functions
def mine_patterns(task_type: Optional[str] = None, agent_id: Optional[str] = None, days_back: int = 30) -> List[Pattern]:
    """Mine patterns from telemetry data."""
    return pattern_miner.mine_patterns(task_type, agent_id, days_back)


def create_playbooks_from_patterns(patterns: List[Pattern]) -> List[Playbook]:
    """Create playbooks from discovered patterns."""
    playbooks = []
    
    with get_db_session() as db:
        for pattern in patterns:
            playbook = pattern_miner.create_playbook_from_pattern(pattern, db)
            playbooks.append(playbook)
    
    return playbooks
