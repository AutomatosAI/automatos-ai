
"""
Agent Communication System
==========================

This module provides a lightweight communication system for AI agents to coordinate
and share information during workflow execution. It includes:

- In-memory message passing between agents
- Task coordination and handoff mechanisms
- Status broadcasting and subscription
- Event-driven communication patterns
- Persistent communication logging
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid

class MessageType(Enum):
    TASK_HANDOFF = "task_handoff"
    STATUS_UPDATE = "status_update"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ERROR_REPORT = "error_report"
    COORDINATION = "coordination"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentMessage:
    message_id: str
    from_agent: str
    to_agent: str  # Can be "*" for broadcast
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentInfo:
    agent_id: str
    agent_type: str
    capabilities: List[str]
    current_status: str
    last_seen: datetime
    message_queue_size: int
    metadata: Dict[str, Any] = None

class AgentCommunicationHub:
    """Central communication hub for agent coordination"""
    
    def __init__(self, max_message_history: int = 1000):
        self.max_message_history = max_message_history
        
        # Agent registry
        self.registered_agents: Dict[str, AgentInfo] = {}
        
        # Message queues for each agent
        self.agent_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Message history for logging and debugging
        self.message_history: deque = deque(maxlen=max_message_history)
        
        # Subscription system for broadcasts
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> set of agent_ids
        
        # Response tracking
        self.pending_responses: Dict[str, AgentMessage] = {}
        
        # Event handlers
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "broadcasts_sent": 0,
            "errors_reported": 0
        }
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Register an agent with the communication hub"""
        with self._lock:
            self.registered_agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                current_status="idle",
                last_seen=datetime.now(),
                message_queue_size=0,
                metadata=metadata or {}
            )
            
            # Initialize message queue
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = deque(maxlen=100)
            
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the communication hub"""
        with self._lock:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
                
                # Clear message queue
                if agent_id in self.agent_queues:
                    del self.agent_queues[agent_id]
                
                # Remove from subscriptions
                for topic_agents in self.subscriptions.values():
                    topic_agents.discard(agent_id)
                
                return True
            return False
    
    def send_message(self, 
                    from_agent: str, 
                    to_agent: str, 
                    message_type: MessageType, 
                    content: Dict[str, Any],
                    priority: MessagePriority = MessagePriority.NORMAL,
                    requires_response: bool = False,
                    expires_in_seconds: int = None,
                    correlation_id: str = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Send a message from one agent to another"""
        
        message_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        message = AgentMessage(
            message_id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            expires_at=expires_at,
            requires_response=requires_response,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Add to message history
            self.message_history.append(message)
            
            # Handle broadcast messages
            if to_agent == "*":
                self._handle_broadcast(message)
                self.stats["broadcasts_sent"] += 1
            else:
                # Send to specific agent
                if to_agent in self.registered_agents:
                    self.agent_queues[to_agent].append(message)
                    self.registered_agents[to_agent].message_queue_size = len(self.agent_queues[to_agent])
                    self.stats["messages_delivered"] += 1
                else:
                    # Agent not found, could log error or queue for later
                    pass
            
            # Track pending responses
            if requires_response:
                self.pending_responses[message_id] = message
            
            self.stats["messages_sent"] += 1
            
            # Trigger message handlers
            self._trigger_message_handlers(message)
        
        return message_id
    
    def _handle_broadcast(self, message: AgentMessage):
        """Handle broadcast messages to subscribed agents"""
        # For now, send to all registered agents except sender
        for agent_id in self.registered_agents:
            if agent_id != message.from_agent:
                self.agent_queues[agent_id].append(message)
                self.registered_agents[agent_id].message_queue_size = len(self.agent_queues[agent_id])
    
    def get_messages(self, agent_id: str, max_messages: int = 10) -> List[AgentMessage]:
        """Get pending messages for an agent"""
        with self._lock:
            if agent_id not in self.agent_queues:
                return []
            
            messages = []
            queue = self.agent_queues[agent_id]
            
            # Get up to max_messages from the queue
            for _ in range(min(max_messages, len(queue))):
                if queue:
                    message = queue.popleft()
                    
                    # Check if message has expired
                    if message.expires_at and datetime.now() > message.expires_at:
                        continue
                    
                    messages.append(message)
            
            # Update queue size
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id].message_queue_size = len(queue)
                self.registered_agents[agent_id].last_seen = datetime.now()
            
            return messages
    
    def send_response(self, 
                     agent_id: str, 
                     original_message_id: str, 
                     response_content: Dict[str, Any],
                     metadata: Dict[str, Any] = None) -> str:
        """Send a response to a message that required a response"""
        
        with self._lock:
            if original_message_id not in self.pending_responses:
                return None
            
            original_message = self.pending_responses[original_message_id]
            
            response_id = self.send_message(
                from_agent=agent_id,
                to_agent=original_message.from_agent,
                message_type=MessageType.RESPONSE,
                content=response_content,
                correlation_id=original_message_id,
                metadata=metadata
            )
            
            # Remove from pending responses
            del self.pending_responses[original_message_id]
            
            return response_id
    
    def update_agent_status(self, agent_id: str, status: str, metadata: Dict[str, Any] = None):
        """Update an agent's status"""
        with self._lock:
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id].current_status = status
                self.registered_agents[agent_id].last_seen = datetime.now()
                if metadata:
                    self.registered_agents[agent_id].metadata.update(metadata)
                
                # Broadcast status update
                self.send_message(
                    from_agent=agent_id,
                    to_agent="*",
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        "agent_id": agent_id,
                        "status": status,
                        "metadata": metadata or {}
                    }
                )
    
    def request_task_handoff(self, 
                           from_agent: str, 
                           to_agent: str, 
                           task_description: str, 
                           task_data: Dict[str, Any],
                           priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Request a task handoff from one agent to another"""
        
        return self.send_message(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=MessageType.TASK_HANDOFF,
            content={
                "task_description": task_description,
                "task_data": task_data,
                "handoff_time": datetime.now().isoformat()
            },
            priority=priority,
            requires_response=True
        )
    
    def report_error(self, agent_id: str, error_details: Dict[str, Any]):
        """Report an error from an agent"""
        with self._lock:
            self.stats["errors_reported"] += 1
        
        return self.send_message(
            from_agent=agent_id,
            to_agent="*",
            message_type=MessageType.ERROR_REPORT,
            content={
                "error_details": error_details,
                "timestamp": datetime.now().isoformat()
            },
            priority=MessagePriority.HIGH
        )
    
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic for targeted broadcasts"""
        with self._lock:
            self.subscriptions[topic].add(agent_id)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic"""
        with self._lock:
            self.subscriptions[topic].discard(agent_id)
    
    def add_message_handler(self, message_type: str, handler: Callable[[AgentMessage], None]):
        """Add a handler for specific message types"""
        self.message_handlers[message_type].append(handler)
    
    def _trigger_message_handlers(self, message: AgentMessage):
        """Trigger registered message handlers"""
        handlers = self.message_handlers.get(message.message_type.value, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                # Log handler errors but don't let them break message flow
                pass
    
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get information about a registered agent"""
        with self._lock:
            return self.registered_agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, AgentInfo]:
        """Get information about all registered agents"""
        with self._lock:
            return dict(self.registered_agents)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        with self._lock:
            return {
                "registered_agents": len(self.registered_agents),
                "total_messages_sent": self.stats["messages_sent"],
                "total_messages_delivered": self.stats["messages_delivered"],
                "total_broadcasts_sent": self.stats["broadcasts_sent"],
                "total_errors_reported": self.stats["errors_reported"],
                "pending_responses": len(self.pending_responses),
                "message_history_size": len(self.message_history),
                "active_subscriptions": {topic: len(agents) for topic, agents in self.subscriptions.items()}
            }
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent messages for debugging/monitoring"""
        with self._lock:
            recent = list(self.message_history)[-limit:]
            return [asdict(msg) for msg in recent]
    
    def clear_expired_messages(self):
        """Clean up expired messages and pending responses"""
        now = datetime.now()
        
        with self._lock:
            # Clear expired pending responses
            expired_responses = [
                msg_id for msg_id, msg in self.pending_responses.items()
                if msg.expires_at and now > msg.expires_at
            ]
            
            for msg_id in expired_responses:
                del self.pending_responses[msg_id]
            
            # Clear expired messages from agent queues
            for agent_id, queue in self.agent_queues.items():
                # Create new queue with non-expired messages
                new_queue = deque(maxlen=queue.maxlen)
                while queue:
                    msg = queue.popleft()
                    if not msg.expires_at or now <= msg.expires_at:
                        new_queue.append(msg)
                
                self.agent_queues[agent_id] = new_queue
                
                # Update queue size
                if agent_id in self.registered_agents:
                    self.registered_agents[agent_id].message_queue_size = len(new_queue)

class AgentCommunicationClient:
    """Client interface for agents to communicate through the hub"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str], hub: AgentCommunicationHub):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.hub = hub
        
        # Register with hub
        self.hub.register_agent(agent_id, agent_type, capabilities)
    
    def send_message(self, to_agent: str, message_type: MessageType, content: Dict[str, Any], **kwargs) -> str:
        """Send a message to another agent"""
        return self.hub.send_message(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            **kwargs
        )
    
    def get_messages(self, max_messages: int = 10) -> List[AgentMessage]:
        """Get pending messages for this agent"""
        return self.hub.get_messages(self.agent_id, max_messages)
    
    def send_response(self, original_message_id: str, response_content: Dict[str, Any], **kwargs) -> str:
        """Send a response to a message"""
        return self.hub.send_response(self.agent_id, original_message_id, response_content, **kwargs)
    
    def update_status(self, status: str, metadata: Dict[str, Any] = None):
        """Update this agent's status"""
        self.hub.update_agent_status(self.agent_id, status, metadata)
    
    def request_handoff(self, to_agent: str, task_description: str, task_data: Dict[str, Any], **kwargs) -> str:
        """Request a task handoff to another agent"""
        return self.hub.request_task_handoff(self.agent_id, to_agent, task_description, task_data, **kwargs)
    
    def report_error(self, error_details: Dict[str, Any]):
        """Report an error"""
        return self.hub.report_error(self.agent_id, error_details)
    
    def broadcast(self, message_type: MessageType, content: Dict[str, Any], **kwargs) -> str:
        """Send a broadcast message to all agents"""
        return self.send_message("*", message_type, content, **kwargs)
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe to a topic"""
        self.hub.subscribe_to_topic(self.agent_id, topic)
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from a topic"""
        self.hub.unsubscribe_from_topic(self.agent_id, topic)
    
    def disconnect(self):
        """Disconnect from the communication hub"""
        self.hub.unregister_agent(self.agent_id)

# Global communication hub instance
_global_hub = None

def get_communication_hub() -> AgentCommunicationHub:
    """Get the global communication hub instance"""
    global _global_hub
    if _global_hub is None:
        _global_hub = AgentCommunicationHub()
    return _global_hub

def create_agent_client(agent_id: str, agent_type: str, capabilities: List[str]) -> AgentCommunicationClient:
    """Create an agent communication client"""
    hub = get_communication_hub()
    return AgentCommunicationClient(agent_id, agent_type, capabilities, hub)

# Example usage and testing
if __name__ == "__main__":
    # Test the communication system
    hub = AgentCommunicationHub()
    
    # Create agent clients
    orchestrator = AgentCommunicationClient("orchestrator", "main", ["coordination", "planning"], hub)
    code_gen = AgentCommunicationClient("code_generator", "specialist", ["code_generation", "python"], hub)
    git_agent = AgentCommunicationClient("git_manager", "specialist", ["git_operations", "version_control"], hub)
    
    # Test communication
    print("Testing agent communication...")
    
    # Orchestrator requests task handoff
    handoff_id = orchestrator.request_handoff(
        "code_generator",
        "Generate main application file",
        {"file_type": "python", "requirements": ["FastAPI", "async support"]}
    )
    
    # Code generator gets messages
    messages = code_gen.get_messages()
    print(f"Code generator received {len(messages)} messages")
    
    if messages:
        msg = messages[0]
        print(f"Message: {msg.content}")
        
        # Send response
        code_gen.send_response(
            msg.message_id,
            {"status": "accepted", "estimated_time": "5 minutes"}
        )
    
    # Update status
    code_gen.update_status("working", {"current_file": "main.py", "progress": 25})
    
    # Broadcast completion
    code_gen.broadcast(
        MessageType.STATUS_UPDATE,
        {"message": "Code generation completed", "files_created": ["main.py", "requirements.txt"]}
    )
    
    # Get stats
    stats = hub.get_communication_stats()
    print(f"Communication stats: {stats}")
    
    print("Communication test completed!")
