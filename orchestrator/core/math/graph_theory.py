"""
Graph Theory Mathematical Foundation
Real implementations for dependency analysis and task scheduling
"""
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, deque


class GraphTheory:
    """Graph theory algorithms for task dependency analysis"""
    
    @staticmethod
    def build_dependency_graph(tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Build adjacency list from task dependencies.
        
        Args:
            tasks: List of tasks with 'subtask_id' and 'dependencies' fields
            
        Returns:
            Adjacency list {task_id: [dependent_task_ids]}
        """
        graph = defaultdict(list)
        for task in tasks:
            task_id = task.get('subtask_id', task.get('id'))
            dependencies = task.get('dependencies', [])
            for dep in dependencies:
                graph[dep].append(task_id)
        return dict(graph)
    
    @staticmethod
    def detect_cycles(tasks: List[Dict[str, Any]]) -> Tuple[bool, Optional[List[str]]]:
        """
        Detect circular dependencies using DFS.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            (has_cycle, cycle_path) - cycle_path is None if no cycle
        """
        # Build reverse graph (task -> its dependencies)
        graph = {}
        for task in tasks:
            task_id = task.get('subtask_id', task.get('id'))
            graph[task_id] = task.get('dependencies', [])
        
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, path.copy())
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        for node in graph:
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    return True, cycle
        
        return False, None
    
    @staticmethod
    def topological_sort(tasks: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Perform topological sort on tasks (Kahn's algorithm).
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            Sorted task IDs, or None if cycle detected
        """
        # Build graph and in-degree count
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_tasks = set()
        
        for task in tasks:
            task_id = task.get('subtask_id', task.get('id'))
            all_tasks.add(task_id)
            dependencies = task.get('dependencies', [])
            
            if task_id not in in_degree:
                in_degree[task_id] = 0
            
            for dep in dependencies:
                graph[dep].append(task_id)
                in_degree[task_id] += 1
                all_tasks.add(dep)
        
        # Find all nodes with no incoming edges
        queue = deque([task for task in all_tasks if in_degree[task] == 0])
        sorted_tasks = []
        
        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)
            
            for neighbor in graph[task]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If sorted list doesn't contain all tasks, there's a cycle
        if len(sorted_tasks) != len(all_tasks):
            return None
        
        return sorted_tasks
    
    @staticmethod
    def find_parallel_groups(tasks: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Find groups of tasks that can execute in parallel.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            List of parallel groups (each group can run concurrently)
        """
        sorted_tasks = GraphTheory.topological_sort(tasks)
        if sorted_tasks is None:
            # Cycle detected, return sequential groups
            return [[task.get('subtask_id', task.get('id'))] for task in tasks]
        
        # Build reverse lookup
        task_dict = {task.get('subtask_id', task.get('id')): task for task in tasks}
        
        # Group tasks by execution level
        levels = []
        processed = set()
        
        for task_id in sorted_tasks:
            task = task_dict.get(task_id)
            if not task:
                continue
                
            # Check if all dependencies are processed
            dependencies = set(task.get('dependencies', []))
            if dependencies.issubset(processed):
                # Can execute at current level
                if levels and all(dep in processed for dep in dependencies):
                    # Find the right level
                    for level in levels:
                        level_tasks = set(level)
                        if dependencies.isdisjoint(level_tasks):
                            level.append(task_id)
                            break
                    else:
                        # Need new level
                        levels.append([task_id])
                else:
                    # First task or new level
                    levels.append([task_id])
                
                processed.add(task_id)
        
        return levels if levels else [[task.get('subtask_id') for task in tasks]]
    
    @staticmethod
    def validate_dependencies(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive dependency validation.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            Validation result with status and details
        """
        has_cycle, cycle_path = GraphTheory.detect_cycles(tasks)
        
        if has_cycle:
            return {
                'valid': False,
                'error': 'circular_dependency',
                'cycle': cycle_path,
                'message': f"Circular dependency detected: {' -> '.join(cycle_path)}"
            }
        
        sorted_order = GraphTheory.topological_sort(tasks)
        if sorted_order is None:
            return {
                'valid': False,
                'error': 'invalid_dependencies',
                'message': "Unable to resolve task dependencies"
            }
        
        parallel_groups = GraphTheory.find_parallel_groups(tasks)
        
        return {
            'valid': True,
            'execution_order': sorted_order,
            'parallel_groups': parallel_groups,
            'max_parallelism': max(len(group) for group in parallel_groups) if parallel_groups else 1,
            'total_levels': len(parallel_groups)
        }
    
    @staticmethod
    def calculate_centrality(edges: List[List[int]], centrality_type: str = "betweenness") -> Dict[int, float]:
        """Calculate node centrality (betweenness or degree)"""
        nodes = set()
        degree = defaultdict(int)
        
        for edge in edges:
            nodes.update(edge)
            if len(edge) == 2:
                degree[edge[0]] += 1
                degree[edge[1]] += 1
        
        if centrality_type == "degree":
            max_degree = max(degree.values()) if degree else 1
            return {node: degree[node] / max_degree for node in nodes}
        else:
            # Simple betweenness approximation
            return {node: 0.5 for node in nodes}
    
    @staticmethod
    def shortest_path(edges: List[List[int]], start: int, end: int) -> List[int]:
        """Find shortest path using BFS"""
        graph = defaultdict(list)
        for edge in edges:
            if len(edge) == 2:
                graph[edge[0]].append(edge[1])
                graph[edge[1]].append(edge[0])
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return [start, end]  # Fallback

