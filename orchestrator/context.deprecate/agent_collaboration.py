
"""
Advanced Agent Collaboration System
===================================

Context-aware agent coordination with predictive collaboration and shared intelligence.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

from .vector_store import PgVectorStore
from .embeddings import EmbeddingGenerator
from .context_retriever import ContextResult

logger = logging.getLogger(__name__)

class CollaborationType(Enum):
    """Types of agent collaboration"""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"     # Simultaneous work
    HIERARCHICAL = "hierarchical"  # Manager-worker relationship
    PEER_TO_PEER = "peer_to_peer"  # Equal collaboration
    CONSULTATIVE = "consultative"  # Expert consultation

class AgentRole(Enum):
    """Specialized agent roles"""
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DEPLOYER = "deployer"
    DOCUMENTER = "documenter"
    SECURITY_EXPERT = "security_expert"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    UI_UX_DESIGNER = "ui_ux_designer"
    DATA_ANALYST = "data_analyst"

@dataclass
class AgentCapability:
    """Represents an agent's capability"""
    skill_name: str
    proficiency_level: float  # 0.0 to 1.0
    experience_count: int
    success_rate: float
    last_used: datetime
    specializations: List[str]

@dataclass
class CollaborationRequest:
    """Request for agent collaboration"""
    request_id: str
    requesting_agent: str
    task_description: str
    required_skills: List[str]
    collaboration_type: CollaborationType
    priority: int  # 1-10, 10 being highest
    deadline: Optional[datetime]
    context: Dict[str, Any]
    created_at: datetime

@dataclass
class CollaborationSession:
    """Active collaboration session"""
    session_id: str
    participants: List[str]
    collaboration_type: CollaborationType
    task_description: str
    shared_context: Dict[str, Any]
    messages: List[Dict[str, Any]]
    status: str  # 'active', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime

class AgentProfile:
    """Comprehensive agent profile with capabilities and history"""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str = None):
        self.agent_id = agent_id
        self.role = role
        self.name = name or f"{role.value}_{agent_id}"
        self.capabilities: Dict[str, AgentCapability] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'collaboration_score': 0.0
        }
        self.preferences = {
            'preferred_collaboration_types': [],
            'working_hours': None,
            'max_concurrent_tasks': 3
        }
        self.current_workload = 0
        self.status = 'available'  # 'available', 'busy', 'offline'
        self.created_at = datetime.now()
        self.last_active = datetime.now()
    
    def add_capability(self, skill_name: str, proficiency: float, specializations: List[str] = None):
        """Add or update a capability"""
        self.capabilities[skill_name] = AgentCapability(
            skill_name=skill_name,
            proficiency_level=proficiency,
            experience_count=0,
            success_rate=0.0,
            last_used=datetime.now(),
            specializations=specializations or []
        )
    
    def update_capability_performance(self, skill_name: str, success: bool):
        """Update capability performance based on task outcome"""
        if skill_name in self.capabilities:
            capability = self.capabilities[skill_name]
            capability.experience_count += 1
            
            # Update success rate using exponential moving average
            alpha = 0.1
            if capability.experience_count == 1:
                capability.success_rate = 1.0 if success else 0.0
            else:
                current_success = 1.0 if success else 0.0
                capability.success_rate = (1 - alpha) * capability.success_rate + alpha * current_success
            
            capability.last_used = datetime.now()
    
    def get_skill_match_score(self, required_skills: List[str]) -> float:
        """Calculate how well this agent matches required skills"""
        if not required_skills:
            return 0.0
        
        total_score = 0.0
        matched_skills = 0
        
        for skill in required_skills:
            if skill in self.capabilities:
                capability = self.capabilities[skill]
                # Score based on proficiency and success rate
                skill_score = capability.proficiency_level * capability.success_rate
                total_score += skill_score
                matched_skills += 1
        
        # Penalty for missing skills
        coverage = matched_skills / len(required_skills)
        return (total_score / len(required_skills)) * coverage
    
    def is_available_for_collaboration(self) -> bool:
        """Check if agent is available for collaboration"""
        return (self.status == 'available' and 
                self.current_workload < self.preferences['max_concurrent_tasks'])

class SmartAgentRegistry:
    """Registry for managing agent profiles and capabilities"""
    
    def __init__(self, vector_store: PgVectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.agents: Dict[str, AgentProfile] = {}
        self.skill_embeddings: Dict[str, List[float]] = {}
        
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent profiles"""
        
        # Software Architect
        architect = AgentProfile("arch_001", AgentRole.ARCHITECT, "Senior Architect")
        architect.add_capability("system_design", 0.95, ["microservices", "distributed_systems"])
        architect.add_capability("architecture_patterns", 0.90, ["mvc", "mvvm", "clean_architecture"])
        architect.add_capability("scalability_planning", 0.85, ["load_balancing", "caching"])
        self.agents[architect.agent_id] = architect
        
        # Senior Developer
        developer = AgentProfile("dev_001", AgentRole.DEVELOPER, "Senior Developer")
        developer.add_capability("python_development", 0.90, ["django", "fastapi", "flask"])
        developer.add_capability("javascript_development", 0.85, ["react", "node.js", "typescript"])
        developer.add_capability("database_design", 0.80, ["postgresql", "mongodb", "redis"])
        developer.add_capability("api_development", 0.88, ["rest", "graphql", "websockets"])
        self.agents[developer.agent_id] = developer
        
        # QA Tester
        tester = AgentProfile("test_001", AgentRole.TESTER, "QA Engineer")
        tester.add_capability("test_automation", 0.85, ["pytest", "selenium", "jest"])
        tester.add_capability("performance_testing", 0.75, ["load_testing", "stress_testing"])
        tester.add_capability("security_testing", 0.70, ["penetration_testing", "vulnerability_assessment"])
        self.agents[tester.agent_id] = tester
        
        # Code Reviewer
        reviewer = AgentProfile("rev_001", AgentRole.REVIEWER, "Senior Code Reviewer")
        reviewer.add_capability("code_review", 0.92, ["security_review", "performance_review"])
        reviewer.add_capability("best_practices", 0.88, ["clean_code", "solid_principles"])
        reviewer.add_capability("security_analysis", 0.80, ["owasp", "secure_coding"])
        self.agents[reviewer.agent_id] = reviewer
        
        # DevOps Engineer
        deployer = AgentProfile("dep_001", AgentRole.DEPLOYER, "DevOps Engineer")
        deployer.add_capability("containerization", 0.85, ["docker", "kubernetes"])
        deployer.add_capability("ci_cd", 0.80, ["github_actions", "jenkins", "gitlab_ci"])
        deployer.add_capability("cloud_deployment", 0.75, ["aws", "gcp", "azure"])
        self.agents[deployer.agent_id] = deployer
        
        # Technical Writer
        documenter = AgentProfile("doc_001", AgentRole.DOCUMENTER, "Technical Writer")
        documenter.add_capability("technical_writing", 0.90, ["api_docs", "user_guides"])
        documenter.add_capability("documentation_tools", 0.85, ["markdown", "sphinx", "gitbook"])
        self.agents[documenter.agent_id] = documenter
    
    async def register_agent(self, agent_profile: AgentProfile):
        """Register a new agent"""
        self.agents[agent_profile.agent_id] = agent_profile
        
        # Generate embeddings for agent skills
        skills = list(agent_profile.capabilities.keys())
        if skills:
            skill_embeddings = await self.embedding_generator.generate_embeddings(skills)
            for i, skill in enumerate(skills):
                if i < len(skill_embeddings):
                    self.skill_embeddings[f"{agent_profile.agent_id}_{skill}"] = skill_embeddings[i]['embedding']
        
        logger.info(f"Registered agent: {agent_profile.name} ({agent_profile.agent_id})")
    
    async def find_best_agents(self, required_skills: List[str], 
                             collaboration_type: CollaborationType,
                             max_agents: int = 3) -> List[Tuple[AgentProfile, float]]:
        """Find the best agents for given requirements"""
        
        candidates = []
        
        for agent in self.agents.values():
            if not agent.is_available_for_collaboration():
                continue
            
            # Calculate skill match score
            skill_score = agent.get_skill_match_score(required_skills)
            
            # Calculate collaboration compatibility
            collab_score = self._calculate_collaboration_compatibility(agent, collaboration_type)
            
            # Calculate workload factor (prefer less busy agents)
            workload_factor = 1.0 - (agent.current_workload / agent.preferences['max_concurrent_tasks'])
            
            # Calculate recency factor (prefer recently active agents)
            hours_since_active = (datetime.now() - agent.last_active).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 - (hours_since_active / 24))  # Decay over 24 hours
            
            # Combined score
            total_score = (skill_score * 0.4 + 
                          collab_score * 0.2 + 
                          workload_factor * 0.2 + 
                          recency_factor * 0.1 +
                          agent.performance_metrics['success_rate'] * 0.1)
            
            if total_score > 0.3:  # Minimum threshold
                candidates.append((agent, total_score))
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_agents]
    
    def _calculate_collaboration_compatibility(self, agent: AgentProfile, 
                                             collaboration_type: CollaborationType) -> float:
        """Calculate how compatible an agent is with a collaboration type"""
        
        # Role-based compatibility
        role_compatibility = {
            AgentRole.ARCHITECT: {
                CollaborationType.HIERARCHICAL: 0.9,
                CollaborationType.CONSULTATIVE: 0.8,
                CollaborationType.SEQUENTIAL: 0.7,
                CollaborationType.PEER_TO_PEER: 0.6,
                CollaborationType.PARALLEL: 0.5
            },
            AgentRole.DEVELOPER: {
                CollaborationType.PARALLEL: 0.9,
                CollaborationType.PEER_TO_PEER: 0.8,
                CollaborationType.SEQUENTIAL: 0.8,
                CollaborationType.HIERARCHICAL: 0.6,
                CollaborationType.CONSULTATIVE: 0.5
            },
            AgentRole.TESTER: {
                CollaborationType.SEQUENTIAL: 0.9,
                CollaborationType.PARALLEL: 0.7,
                CollaborationType.PEER_TO_PEER: 0.6,
                CollaborationType.CONSULTATIVE: 0.8,
                CollaborationType.HIERARCHICAL: 0.5
            }
        }
        
        return role_compatibility.get(agent.role, {}).get(collaboration_type, 0.5)
    
    def update_agent_performance(self, agent_id: str, task_outcome: Dict[str, Any]):
        """Update agent performance metrics"""
        
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        agent.performance_metrics['tasks_completed'] += 1
        
        # Update success rate
        success = task_outcome.get('success', False)
        alpha = 0.1
        current_success_rate = agent.performance_metrics['success_rate']
        agent.performance_metrics['success_rate'] = (
            (1 - alpha) * current_success_rate + alpha * (1.0 if success else 0.0)
        )
        
        # Update response time
        response_time = task_outcome.get('response_time', 0)
        if response_time > 0:
            current_avg = agent.performance_metrics['avg_response_time']
            agent.performance_metrics['avg_response_time'] = (
                (current_avg * (agent.performance_metrics['tasks_completed'] - 1) + response_time) /
                agent.performance_metrics['tasks_completed']
            )
        
        # Update skill-specific performance
        used_skills = task_outcome.get('skills_used', [])
        for skill in used_skills:
            agent.update_capability_performance(skill, success)
        
        agent.last_active = datetime.now()

class CollaborationOrchestrator:
    """Orchestrates agent collaborations with context awareness"""
    
    def __init__(self, agent_registry: SmartAgentRegistry, 
                 vector_store: PgVectorStore, embedding_generator: EmbeddingGenerator):
        self.agent_registry = agent_registry
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.collaboration_queue: deque = deque()
        self.collaboration_history: List[Dict[str, Any]] = []
        self.shared_memory: Dict[str, Any] = {}
    
    async def request_collaboration(self, requesting_agent: str, task_description: str,
                                  required_skills: List[str], collaboration_type: CollaborationType,
                                  priority: int = 5, deadline: datetime = None,
                                  context: Dict[str, Any] = None) -> str:
        """Request agent collaboration"""
        
        request_id = str(uuid.uuid4())
        
        collaboration_request = CollaborationRequest(
            request_id=request_id,
            requesting_agent=requesting_agent,
            task_description=task_description,
            required_skills=required_skills,
            collaboration_type=collaboration_type,
            priority=priority,
            deadline=deadline,
            context=context or {},
            created_at=datetime.now()
        )
        
        # Add to queue (sorted by priority)
        self.collaboration_queue.append(collaboration_request)
        self.collaboration_queue = deque(sorted(self.collaboration_queue, 
                                               key=lambda x: x.priority, reverse=True))
        
        logger.info(f"Collaboration requested: {request_id} for task: {task_description[:50]}...")
        
        # Try to process immediately if high priority
        if priority >= 8:
            await self._process_collaboration_request(collaboration_request)
        
        return request_id
    
    async def _process_collaboration_request(self, request: CollaborationRequest) -> Optional[str]:
        """Process a collaboration request"""
        
        try:
            # Find best agents for the task
            best_agents = await self.agent_registry.find_best_agents(
                required_skills=request.required_skills,
                collaboration_type=request.collaboration_type,
                max_agents=self._get_optimal_team_size(request.collaboration_type)
            )
            
            if not best_agents:
                logger.warning(f"No suitable agents found for request {request.request_id}")
                return None
            
            # Create collaboration session
            session_id = await self._create_collaboration_session(request, best_agents)
            
            # Initialize shared context
            await self._initialize_shared_context(session_id, request)
            
            # Start collaboration
            await self._start_collaboration(session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error processing collaboration request: {str(e)}")
            return None
    
    def _get_optimal_team_size(self, collaboration_type: CollaborationType) -> int:
        """Get optimal team size for collaboration type"""
        
        team_sizes = {
            CollaborationType.SEQUENTIAL: 3,
            CollaborationType.PARALLEL: 4,
            CollaborationType.HIERARCHICAL: 5,
            CollaborationType.PEER_TO_PEER: 3,
            CollaborationType.CONSULTATIVE: 2
        }
        
        return team_sizes.get(collaboration_type, 3)
    
    async def _create_collaboration_session(self, request: CollaborationRequest,
                                          selected_agents: List[Tuple[AgentProfile, float]]) -> str:
        """Create a new collaboration session"""
        
        session_id = str(uuid.uuid4())
        participants = [agent.agent_id for agent, _ in selected_agents]
        
        session = CollaborationSession(
            session_id=session_id,
            participants=participants,
            collaboration_type=request.collaboration_type,
            task_description=request.task_description,
            shared_context=request.context.copy(),
            messages=[],
            status='active',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        # Update agent workloads
        for agent_id in participants:
            if agent_id in self.agent_registry.agents:
                self.agent_registry.agents[agent_id].current_workload += 1
        
        logger.info(f"Created collaboration session {session_id} with agents: {participants}")
        return session_id
    
    async def _initialize_shared_context(self, session_id: str, request: CollaborationRequest):
        """Initialize shared context for the collaboration"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Retrieve relevant context from vector store
        if request.task_description:
            # Generate embedding for task
            task_embedding = await self.embedding_generator.generate_embeddings([request.task_description])
            
            if task_embedding:
                # Search for relevant context
                similar_contexts = await self.vector_store.similarity_search(
                    query_embedding=task_embedding[0]['embedding'],
                    limit=5,
                    similarity_threshold=0.3
                )
                
                # Add to shared context
                session.shared_context['relevant_documents'] = [
                    {
                        'content': result.content,
                        'source': result.source_file,
                        'relevance': result.similarity_score
                    }
                    for result in similar_contexts
                ]
        
        # Add historical collaboration patterns
        similar_collaborations = await self._find_similar_collaborations(request)
        session.shared_context['similar_patterns'] = similar_collaborations
        
        # Add agent capabilities summary
        session.shared_context['team_capabilities'] = {
            agent_id: {
                'role': self.agent_registry.agents[agent_id].role.value,
                'key_skills': list(self.agent_registry.agents[agent_id].capabilities.keys())[:5]
            }
            for agent_id in session.participants
            if agent_id in self.agent_registry.agents
        }
    
    async def _find_similar_collaborations(self, request: CollaborationRequest) -> List[Dict[str, Any]]:
        """Find similar past collaborations"""
        
        similar_collaborations = []
        
        # Simple similarity based on required skills overlap
        for collab in self.collaboration_history[-20:]:  # Check recent collaborations
            if collab.get('collaboration_type') == request.collaboration_type.value:
                # Calculate skill overlap
                past_skills = set(collab.get('required_skills', []))
                current_skills = set(request.required_skills)
                
                overlap = len(past_skills.intersection(current_skills))
                if overlap > 0:
                    similarity = overlap / len(past_skills.union(current_skills))
                    if similarity > 0.3:
                        similar_collaborations.append({
                            'similarity': similarity,
                            'outcome': collab.get('outcome'),
                            'duration': collab.get('duration'),
                            'success': collab.get('success', False)
                        })
        
        return sorted(similar_collaborations, key=lambda x: x['similarity'], reverse=True)[:3]
    
    async def _start_collaboration(self, session_id: str):
        """Start the collaboration process"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Send initial message to all participants
        initial_message = {
            'type': 'collaboration_start',
            'session_id': session_id,
            'task_description': session.task_description,
            'collaboration_type': session.collaboration_type.value,
            'shared_context': session.shared_context,
            'participants': session.participants,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, initial_message)
        
        # Set up collaboration workflow based on type
        await self._setup_collaboration_workflow(session_id)
    
    async def _setup_collaboration_workflow(self, session_id: str):
        """Set up workflow based on collaboration type"""
        
        session = self.active_sessions[session_id]
        
        if session.collaboration_type == CollaborationType.SEQUENTIAL:
            await self._setup_sequential_workflow(session_id)
        elif session.collaboration_type == CollaborationType.PARALLEL:
            await self._setup_parallel_workflow(session_id)
        elif session.collaboration_type == CollaborationType.HIERARCHICAL:
            await self._setup_hierarchical_workflow(session_id)
        elif session.collaboration_type == CollaborationType.PEER_TO_PEER:
            await self._setup_peer_to_peer_workflow(session_id)
        elif session.collaboration_type == CollaborationType.CONSULTATIVE:
            await self._setup_consultative_workflow(session_id)
    
    async def _setup_sequential_workflow(self, session_id: str):
        """Set up sequential collaboration workflow"""
        
        session = self.active_sessions[session_id]
        
        # Order agents by their role priority for sequential work
        role_priority = {
            AgentRole.ARCHITECT: 1,
            AgentRole.DEVELOPER: 2,
            AgentRole.TESTER: 3,
            AgentRole.REVIEWER: 4,
            AgentRole.DEPLOYER: 5,
            AgentRole.DOCUMENTER: 6
        }
        
        ordered_agents = []
        for agent_id in session.participants:
            if agent_id in self.agent_registry.agents:
                agent = self.agent_registry.agents[agent_id]
                priority = role_priority.get(agent.role, 10)
                ordered_agents.append((agent_id, priority))
        
        ordered_agents.sort(key=lambda x: x[1])
        
        # Set up workflow steps
        workflow_message = {
            'type': 'workflow_setup',
            'workflow_type': 'sequential',
            'agent_order': [agent_id for agent_id, _ in ordered_agents],
            'current_agent': ordered_agents[0][0] if ordered_agents else None,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, workflow_message)
    
    async def _setup_parallel_workflow(self, session_id: str):
        """Set up parallel collaboration workflow"""
        
        session = self.active_sessions[session_id]
        
        # Divide task into parallel subtasks based on agent capabilities
        subtasks = await self._divide_task_for_parallel_work(session.task_description, session.participants)
        
        workflow_message = {
            'type': 'workflow_setup',
            'workflow_type': 'parallel',
            'subtasks': subtasks,
            'coordination_agent': session.participants[0],  # First agent coordinates
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, workflow_message)
    
    async def _divide_task_for_parallel_work(self, task_description: str, 
                                           participants: List[str]) -> Dict[str, str]:
        """Divide task into parallel subtasks"""
        
        subtasks = {}
        
        # Simple heuristic based on agent roles
        for agent_id in participants:
            if agent_id in self.agent_registry.agents:
                agent = self.agent_registry.agents[agent_id]
                
                if agent.role == AgentRole.DEVELOPER:
                    subtasks[agent_id] = f"Implement core functionality for: {task_description}"
                elif agent.role == AgentRole.TESTER:
                    subtasks[agent_id] = f"Create test suite for: {task_description}"
                elif agent.role == AgentRole.DOCUMENTER:
                    subtasks[agent_id] = f"Create documentation for: {task_description}"
                elif agent.role == AgentRole.SECURITY_EXPERT:
                    subtasks[agent_id] = f"Security analysis for: {task_description}"
                else:
                    subtasks[agent_id] = f"Contribute to: {task_description}"
        
        return subtasks
    
    async def _setup_hierarchical_workflow(self, session_id: str):
        """Set up hierarchical collaboration workflow"""
        
        session = self.active_sessions[session_id]
        
        # Find the most senior agent (architect or highest success rate)
        manager_agent = None
        best_score = 0
        
        for agent_id in session.participants:
            if agent_id in self.agent_registry.agents:
                agent = self.agent_registry.agents[agent_id]
                
                # Prefer architects, then by success rate
                score = agent.performance_metrics['success_rate']
                if agent.role == AgentRole.ARCHITECT:
                    score += 0.5
                
                if score > best_score:
                    best_score = score
                    manager_agent = agent_id
        
        workers = [agent_id for agent_id in session.participants if agent_id != manager_agent]
        
        workflow_message = {
            'type': 'workflow_setup',
            'workflow_type': 'hierarchical',
            'manager': manager_agent,
            'workers': workers,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, workflow_message)
    
    async def _setup_peer_to_peer_workflow(self, session_id: str):
        """Set up peer-to-peer collaboration workflow"""
        
        workflow_message = {
            'type': 'workflow_setup',
            'workflow_type': 'peer_to_peer',
            'coordination_method': 'round_robin',
            'discussion_leader': None,  # Rotates
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, workflow_message)
    
    async def _setup_consultative_workflow(self, session_id: str):
        """Set up consultative collaboration workflow"""
        
        session = self.active_sessions[session_id]
        
        # Identify consultant (highest expertise) and client
        consultant = None
        client = None
        best_expertise = 0
        
        for agent_id in session.participants:
            if agent_id in self.agent_registry.agents:
                agent = self.agent_registry.agents[agent_id]
                
                # Calculate expertise score
                expertise = sum(cap.proficiency_level * cap.success_rate 
                              for cap in agent.capabilities.values())
                
                if expertise > best_expertise:
                    if consultant:
                        client = consultant  # Previous best becomes client
                    consultant = agent_id
                    best_expertise = expertise
                elif not client:
                    client = agent_id
        
        workflow_message = {
            'type': 'workflow_setup',
            'workflow_type': 'consultative',
            'consultant': consultant,
            'client': client,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._broadcast_message(session_id, workflow_message)
    
    async def _broadcast_message(self, session_id: str, message: Dict[str, Any]):
        """Broadcast message to all session participants"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.messages.append(message)
        session.updated_at = datetime.now()
        
        # In a real implementation, this would send messages to actual agents
        logger.info(f"Broadcasting message to session {session_id}: {message['type']}")
    
    async def handle_agent_response(self, session_id: str, agent_id: str, 
                                  response: Dict[str, Any]) -> bool:
        """Handle response from an agent in a collaboration"""
        
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if agent_id not in session.participants:
            return False
        
        # Add response to session messages
        response_message = {
            'type': 'agent_response',
            'agent_id': agent_id,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        session.messages.append(response_message)
        session.updated_at = datetime.now()
        
        # Check if collaboration is complete
        if await self._check_collaboration_complete(session_id):
            await self._complete_collaboration(session_id)
        
        return True
    
    async def _check_collaboration_complete(self, session_id: str) -> bool:
        """Check if collaboration is complete"""
        
        session = self.active_sessions[session_id]
        
        # Simple heuristic: collaboration is complete if all agents have responded
        # or if someone explicitly marks it as complete
        
        agent_responses = set()
        for message in session.messages:
            if message['type'] == 'agent_response':
                agent_responses.add(message['agent_id'])
            elif message['type'] == 'collaboration_complete':
                return True
        
        # All agents have responded at least once
        return len(agent_responses) >= len(session.participants)
    
    async def _complete_collaboration(self, session_id: str):
        """Complete a collaboration session"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.status = 'completed'
        session.updated_at = datetime.now()
        
        # Update agent workloads
        for agent_id in session.participants:
            if agent_id in self.agent_registry.agents:
                agent = self.agent_registry.agents[agent_id]
                agent.current_workload = max(0, agent.current_workload - 1)
        
        # Record collaboration history
        collaboration_record = {
            'session_id': session_id,
            'collaboration_type': session.collaboration_type.value,
            'participants': session.participants,
            'task_description': session.task_description,
            'duration': (session.updated_at - session.created_at).total_seconds(),
            'message_count': len(session.messages),
            'success': True,  # Would be determined by actual outcome
            'completed_at': datetime.now().isoformat()
        }
        
        self.collaboration_history.append(collaboration_record)
        
        # Move to completed sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Completed collaboration session {session_id}")
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        
        return {
            'active_sessions': len(self.active_sessions),
            'queued_requests': len(self.collaboration_queue),
            'completed_collaborations': len(self.collaboration_history),
            'total_agents': len(self.agent_registry.agents),
            'available_agents': len([a for a in self.agent_registry.agents.values() 
                                   if a.is_available_for_collaboration()])
        }

# Factory function for easy initialization
def create_collaboration_system(vector_store: PgVectorStore, 
                               embedding_generator: EmbeddingGenerator) -> CollaborationOrchestrator:
    """Create a complete collaboration system"""
    
    agent_registry = SmartAgentRegistry(vector_store, embedding_generator)
    orchestrator = CollaborationOrchestrator(agent_registry, vector_store, embedding_generator)
    
    return orchestrator
