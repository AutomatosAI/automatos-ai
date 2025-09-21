"""
Agent Factory - User-Defined Agents with Flexible Metadata
===========================================================

Pure execution layer for agents. Users define their own agent types.
The orchestrator handles all prompt engineering using Context Engineering.
Multiple agents of different types can run simultaneously.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.llm_provider import (
    LLMManager, LLMConfig, LLMProvider, LLMResponse,
    create_llm_manager
)
from database.models import (
    Agent, Skill, PriorityLevel, Base
)

logger = logging.getLogger(__name__)

# Agent lifecycle states
class AgentLifecycle(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    LEARNING = "learning"
    HIBERNATING = "hibernating"
    RETIRED = "retired"

# Default LLM configuration for agents
DEFAULT_LLM_CONFIG = {
    "provider": "openai",  # Use environment default
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "context_window": 8192
}

@dataclass
class AgentMetadata:
    """User-defined agent metadata - completely flexible"""
    name: str
    agent_type: str  # User-defined type (e.g., "financial_analyst", "code_reviewer")
    description: Optional[str] = None
    skills: List[str] = field(default_factory=list)  # Semantic tags for matching
    preferred_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)  # Any user data
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration with user overrides"""
        config = DEFAULT_LLM_CONFIG.copy()
        
        if self.preferred_model:
            # Determine provider from model name
            if "gpt" in self.preferred_model.lower():
                config["provider"] = "openai"
                config["model"] = self.preferred_model
            elif "claude" in self.preferred_model.lower():
                config["provider"] = "anthropic"
                config["model"] = self.preferred_model
            else:
                # Future: HuggingFace or custom models
                config["model"] = self.preferred_model
        
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        if self.context_window is not None:
            config["context_window"] = self.context_window
            
        return config

@dataclass
class AgentRuntime:
    """Runtime representation of an agent"""
    agent_id: int
    metadata: AgentMetadata
    llm_manager: LLMManager
    lifecycle_state: AgentLifecycle
    created_at: datetime
    execution_count: int = 0
    total_tokens_used: int = 0
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)  # Short-term memory
    
    def update_metrics(self, execution_time: float, tokens_used: int, success: bool):
        """Update agent performance metrics"""
        self.execution_count += 1
        self.total_tokens_used += tokens_used
        self.last_execution = datetime.now()
        
        # Update rolling metrics
        if "avg_execution_time" not in self.performance_metrics:
            self.performance_metrics["avg_execution_time"] = execution_time
        else:
            # Rolling average
            avg = self.performance_metrics["avg_execution_time"]
            self.performance_metrics["avg_execution_time"] = (
                (avg * (self.execution_count - 1) + execution_time) / self.execution_count
            )
        
        # Success rate
        if "success_count" not in self.performance_metrics:
            self.performance_metrics["success_count"] = 0
        if success:
            self.performance_metrics["success_count"] += 1
        
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["success_count"] / self.execution_count
        )

class AgentFactory:
    """
    Creates and manages user-defined agents.
    Pure execution layer - the orchestrator handles all prompt engineering.
    Can manage multiple agents of different types simultaneously.
    """
    
    def __init__(self, db_session: Session = None):
        self.db_session = db_session
        if not self.db_session:
            engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///automatos.db"))
            Base.metadata.create_all(engine)
            SessionLocal = sessionmaker(bind=engine)
            self.db_session = SessionLocal()
        
        self.active_agents: Dict[int, AgentRuntime] = {}
        self.logger = logging.getLogger(__name__)
    
    async def create_agent(
        self,
        metadata: Union[AgentMetadata, Dict[str, Any]],
        auto_verify: bool = True
    ) -> AgentRuntime:
        """
        Create an agent from user-defined metadata.
        
        Args:
            metadata: AgentMetadata or dict with agent configuration
            auto_verify: Verify LLM connection immediately
            
        Returns:
            AgentRuntime with active LLM connection
        """
        start_time = time.time()
        
        # Convert dict to AgentMetadata if needed
        if isinstance(metadata, dict):
            metadata = AgentMetadata(
                name=metadata.get("name", "Unnamed Agent"),
                agent_type=metadata.get("type", "generic"),
                description=metadata.get("description"),
                skills=metadata.get("skills", []),
                preferred_model=metadata.get("preferred_model"),
                temperature=metadata.get("temperature"),
                max_tokens=metadata.get("max_tokens"),
                context_window=metadata.get("context_window"),
                custom_metadata=metadata.get("metadata", {})
            )
        
        # Create database record
        db_agent = Agent(
            name=metadata.name,
            description=metadata.description or f"User-defined {metadata.agent_type} agent",
            agent_type=metadata.agent_type,  # User-defined type
            status=AgentLifecycle.INITIALIZING.value,
            configuration={
                "skills": metadata.skills,
                "llm_config": metadata.get_llm_config(),
                "custom_metadata": metadata.custom_metadata
            },
            priority_level=PriorityLevel.MEDIUM.value,
            max_concurrent_tasks=5,
            auto_start=False,
            created_by="agent_factory"
        )
        
        self.db_session.add(db_agent)
        self.db_session.commit()
        
        # Initialize LLM connection
        try:
            llm_config_dict = metadata.get_llm_config()
            
            # Determine provider
            provider = LLMProvider(llm_config_dict["provider"])
            
            llm_config = LLMConfig(
                provider=provider,
                model=llm_config_dict["model"],
                temperature=llm_config_dict["temperature"],
                max_tokens=llm_config_dict["max_tokens"],
                # api_key will be loaded from environment
            )
            
            llm_manager = LLMManager(llm_config)
            
            # Verify connection if requested
            if auto_verify:
                verification_result = await self._verify_llm_connection(llm_manager)
                if not verification_result["success"]:
                    self.db_session.delete(db_agent)
                    self.db_session.commit()
                    raise Exception(f"LLM verification failed: {verification_result['error']}")
                
                self.logger.info(
                    f"Agent '{metadata.name}' LLM verified. "
                    f"Response time: {verification_result['response_time']:.2f}s"
                )
            
            # Create runtime agent
            agent_runtime = AgentRuntime(
                agent_id=db_agent.id,
                metadata=metadata,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                created_at=datetime.now()
            )
            
            # Update database status
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()
            
            # Store in active agents
            self.active_agents[db_agent.id] = agent_runtime
            
            self.logger.info(
                f"Agent '{metadata.name}' (type: {metadata.agent_type}) created in "
                f"{time.time() - start_time:.2f}s"
            )
            
            return agent_runtime
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
            if db_agent.id:
                self.db_session.delete(db_agent)
                self.db_session.commit()
            raise
    
    async def _verify_llm_connection(self, llm_manager: LLMManager) -> Dict[str, Any]:
        """Verify LLM connection with minimal test"""
        try:
            start_time = time.time()
            
            # Minimal verification - no embedded prompts
            messages = [
                {"role": "user", "content": "Respond with 'OK' to confirm connection."}
            ]
            
            response = await llm_manager.generate_response(messages)
            response_time = time.time() - start_time
            
            if response and response.content:
                return {
                    "success": True,
                    "response_time": response_time,
                    "tokens_used": response.usage.get("total_tokens", 0) if response.usage else 0
                }
            else:
                return {"success": False, "error": "No response from LLM"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_with_prompt(
        self,
        agent: Union[int, AgentRuntime],
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Execute a task with orchestrator-provided prompt.
        
        The orchestrator provides the fully engineered prompt using Context Engineering.
        
        Args:
            agent: Agent ID or runtime
            prompt: User prompt from orchestrator (may be atomic, molecular, or cellular)
            system_prompt: System prompt from orchestrator (with context, examples, etc.)
            context: Additional structured context
            use_memory: Include agent's short-term memory
            max_retries: Number of retries on failure
            
        Returns:
            Execution result with LLM response
        """
        start_time = time.time()
        
        # Get agent runtime
        if isinstance(agent, int):
            agent_runtime = self.active_agents.get(agent)
            if not agent_runtime:
                return {
                    "status": "error",
                    "error": f"Agent {agent} not found in runtime"
                }
        else:
            agent_runtime = agent
        
        # Update state
        agent_runtime.lifecycle_state = AgentLifecycle.BUSY
        
        try:
            # Build messages - orchestrator provides the engineered prompts
            messages = []
            
            # System prompt from orchestrator (contains context engineering)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add short-term memory if enabled
            if use_memory and agent_runtime.memory:
                # Only include recent relevant memory (cells concept)
                recent_memory = agent_runtime.memory[-3:]  # Last 3 interactions
                for mem in recent_memory:
                    if "user_prompt" in mem:
                        messages.append({"role": "user", "content": mem["user_prompt"]})
                    if "assistant_response" in mem:
                        messages.append({"role": "assistant", "content": mem["assistant_response"]})
            
            # Add the main prompt from orchestrator
            messages.append({"role": "user", "content": prompt})
            
            # Execute with retries
            last_error = None
            for attempt in range(max_retries):
                try:
                    # REAL LLM API CALL
                    response = await agent_runtime.llm_manager.generate_response(messages)
                    execution_time = time.time() - start_time
                    
                    if response and response.content:
                        # Success! Update metrics
                        tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0
                        agent_runtime.update_metrics(execution_time, tokens_used, True)
                        
                        # Store in memory
                        memory_entry = {
                            "task": prompt[:200],  # Truncate for storage
                            "response": response.content[:500],  # Truncate for storage
                            "summary": f"Executed: {prompt[:100]}",
                            "timestamp": datetime.now().isoformat(),
                            "tokens": tokens_used,
                            "execution_time": execution_time
                        }
                        agent_runtime.memory.append(memory_entry)
                        
                        # Keep memory size manageable
                        if len(agent_runtime.memory) > 20:
                            agent_runtime.memory = agent_runtime.memory[-20:]
                        
                        # Update lifecycle state
                        agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
                        
                        # Return successful result
                        return {
                            "status": "success",
                            "result": response.content,
                            "agent": {
                                "id": agent_runtime.agent_id,
                                "name": agent_runtime.metadata.name,
                                "type": agent_runtime.metadata.agent_type
                            },
                            "execution": {
                                "time": execution_time,
                                "tokens_used": tokens_used,
                                "model": response.model,
                                "provider": response.provider,
                                "attempt": attempt + 1
                            },
                            "metrics": {
                                "total_executions": agent_runtime.execution_count,
                                "success_rate": agent_runtime.performance_metrics.get("success_rate", 1.0),
                                "avg_execution_time": agent_runtime.performance_metrics.get("avg_execution_time", execution_time)
                            }
                        }
                    else:
                        last_error = "Empty response from LLM"
                        
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # All retries failed
            agent_runtime.update_metrics(time.time() - start_time, 0, False)
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            
            return {
                "status": "error",
                "error": f"Task execution failed after {max_retries} attempts: {last_error}",
                "agent": {
                    "id": agent_runtime.agent_id,
                    "name": agent_runtime.metadata.name,
                    "type": agent_runtime.metadata.agent_type
                }
            }
            
        except Exception as e:
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            self.logger.error(f"Task execution error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def apply_skills(
        self,
        agent: AgentRuntime,
        new_skills: List[str]
    ) -> AgentRuntime:
        """
        Apply new skills to an agent by enhancing its system prompt.
        
        Args:
            agent: Agent runtime to enhance
            new_skills: List of skill names to add
            
        Returns:
            Updated agent runtime
        """
        skill_enhancements = []
        
        for skill in new_skills:
            if skill in SKILL_PROMPTS and skill not in agent.metadata.skills:
                skill_enhancements.append(SKILL_PROMPTS[skill])
                agent.metadata.skills.append(skill)
                self.logger.info(f"Applied skill '{skill}' to agent '{agent.metadata.name}'")
        
        if skill_enhancements:
            # Append to existing system prompt
            if "\n\n## Specialized Skills:" not in agent.system_prompt:
                agent.system_prompt += "\n\n## Specialized Skills:"
            agent.system_prompt += "".join(skill_enhancements)
            
            self.logger.info(
                f"Enhanced agent '{agent.metadata.name}' with {len(skill_enhancements)} new skills"
            )
        
        return agent
    
    async def get_agent_status(self, agent_id: int) -> Dict[str, Any]:
        """
        Get detailed status of an agent.
        
        Returns real-time agent information.
        """
        agent_runtime = self.active_agents.get(agent_id)
        
        if not agent_runtime:
            # Try database
            db_agent = self.db_session.query(Agent).filter_by(id=agent_id).first()
            if db_agent:
                return {
                    "status": "inactive",
                    "agent": {
                        "id": db_agent.id,
                        "name": db_agent.metadata.name,
                        "type": db_agent.agent_type,
                        "database_status": db_agent.status,
                        "created_at": db_agent.created_at.isoformat() if db_agent.created_at else None
                    },
                    "runtime": None,
                    "message": "Agent exists in database but not in runtime. Use activate_agent() to initialize."
                }
            else:
                return {
                    "status": "not_found",
                    "error": f"Agent {agent_id} does not exist"
                }
        
        # Get provider info
        provider_info = agent_runtime.llm_manager.get_provider_info()
        
        return {
            "status": "active",
            "agent": {
                "id": agent_runtime.agent_id,
                "name": agent_runtime.metadata.name,
                "type": agent_runtime.metadata.agent_type,
                "lifecycle_state": agent_runtime.lifecycle_state.value,
                "skills": agent_runtime.metadata.skills
            },
            "runtime": {
                "created_at": agent_runtime.created_at.isoformat(),
                "last_execution": agent_runtime.last_execution.isoformat() if agent_runtime.last_execution else None,
                "execution_count": agent_runtime.execution_count,
                "total_tokens_used": agent_runtime.total_tokens_used,
                "memory_size": len(agent_runtime.memory)
            },
            "llm": provider_info,
            "metrics": agent_runtime.performance_metrics
        }
    
    async def test_agent_capabilities(self, agent: AgentRuntime) -> Dict[str, Any]:
        """
        Run comprehensive tests on agent capabilities.
        
        Returns detailed test results.
        """
        test_results = {
            "agent_id": agent.agent_id,
            "agent_name": agent.metadata.name,
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Test 1: Basic response
        test1 = await self.execute_with_prompt(
            agent,
            "What are your primary capabilities?",
            use_memory=False
        )
        test_results["tests"].append({
            "name": "basic_response",
            "success": test1["status"] == "success",
            "execution_time": test1.get("execution", {}).get("time"),
            "tokens": test1.get("execution", {}).get("tokens_used")
        })
        
        # Test 2: Skill-specific task
        if agent.metadata.skills:
            skill_task = f"Demonstrate your {agent.metadata.skills[0]} capability with a brief example."
            test2 = await self.execute_with_prompt(agent, skill_task)
            test_results["tests"].append({
                "name": f"skill_test_{agent.metadata.skills[0]}",
                "success": test2["status"] == "success",
                "execution_time": test2.get("execution", {}).get("time"),
                "tokens": test2.get("execution", {}).get("tokens_used")
            })
        
        # Test 3: Context handling
        test3 = await self.execute_with_prompt(
            agent,
            "Analyze this context and provide insights",
            context={"data": "test", "value": 42, "items": ["a", "b", "c"]}
        )
        test_results["tests"].append({
            "name": "context_handling",
            "success": test3["status"] == "success",
            "execution_time": test3.get("execution", {}).get("time"),
            "tokens": test3.get("execution", {}).get("tokens_used")
        })
        
        # Summary
        successful_tests = sum(1 for t in test_results["tests"] if t["success"])
        test_results["summary"] = {
            "total_tests": len(test_results["tests"]),
            "successful": successful_tests,
            "success_rate": successful_tests / len(test_results["tests"]) if test_results["tests"] else 0,
            "total_time": sum(t.get("execution_time", 0) for t in test_results["tests"] if t.get("execution_time")),
            "total_tokens": sum(t.get("tokens", 0) for t in test_results["tests"] if t.get("tokens"))
        }
        
        return test_results
    
    def cleanup(self):
        """Clean up resources"""
        if self.db_session:
            self.db_session.close()


# Convenience function for quick agent creation
async def create_specialized_agent(
    name: str,
    agent_type: str,
    skills: List[str] = None,
    auto_test: bool = True
) -> Dict[str, Any]:
    """
    Quick function to create and test a specialized agent.
    
    Returns agent info and test results.
    """
    factory = AgentFactory()
    
    try:
        # Create agent
        agent = await factory.create_agent(
            name=name,
            agent_type=agent_type,
            skills=skills,
            auto_verify=True
        )
        
        result = {
            "agent": {
                "id": agent.agent_id,
                "name": agent.metadata.name,
                "type": agent.agent_type.value,
                "skills": agent.metadata.skills,
                "status": "created"
            }
        }
        
        # Run tests if requested
        if auto_test:
            test_results = await factory.test_agent_capabilities(agent)
            result["tests"] = test_results
        
        return result
        
    finally:
        factory.cleanup()


# Example usage and testing
if __name__ == "__main__":
    async def test_agent_factory():
        """Test the agent factory with REAL LLM connections"""
        
        print("=" * 60)
        print("AGENT FACTORY TEST - REAL LLM CONNECTIONS")
        print("=" * 60)
        
        factory = AgentFactory()
        
        try:
            # Test 1: Create a Code Architect agent
            print("\n1. Creating Code Architect agent...")
            architect = await factory.create_agent(
                name="CodeMaster",
                agent_type=AgentType.CODE_ARCHITECT,
                skills=["code_analysis", "api_design", "system_design"],
                auto_verify=True
            )
            print(f"✓ Created: {architect.name} (ID: {architect.agent_id})")
            
            # Test 2: Execute a real task
            print("\n2. Executing code review task...")
            code_task = """
            Review this Python function and suggest improvements:
            
            def calc(x, y, op):
                if op == '+':
                    return x + y
                elif op == '-':
                    return x - y
                elif op == '*':
                    return x * y
                elif op == '/':
                    return x / y
            """
            
            result = await factory.execute_with_prompt(architect, code_task)
            
            if result["status"] == "success":
                print(f"✓ Task executed successfully")
                print(f"  Execution time: {result['execution']['time']:.2f}s")
                print(f"  Tokens used: {result['execution']['tokens_used']}")
                print(f"  Model: {result['execution']['model']}")
                print(f"  Provider: {result['execution']['provider']}")
                print("\n  Response preview:")
                print("  " + result["result"][:200] + "...")
            else:
                print(f"✗ Task failed: {result['error']}")
            
            # Test 3: Create a Security Expert
            print("\n3. Creating Security Expert agent...")
            security_expert = await factory.create_agent(
                name="SecGuardian",
                agent_type=AgentType.SECURITY_EXPERT,
                skills=["security_audit", "penetration_testing"],
                auto_verify=True
            )
            print(f"✓ Created: {security_expert.name} (ID: {security_expert.agent_id})")
            
            # Test 4: Security analysis task
            print("\n4. Executing security analysis...")
            security_task = "What are the top 3 security vulnerabilities in web applications?"
            
            sec_result = await factory.execute_with_prompt(security_expert, security_task)
            
            if sec_result["status"] == "success":
                print(f"✓ Security analysis completed")
                print(f"  Tokens used: {sec_result['execution']['tokens_used']}")
            
            # Test 5: Get agent status
            print("\n5. Checking agent status...")
            status = await factory.get_agent_status(architect.agent_id)
            print(f"✓ Agent status retrieved")
            print(f"  Lifecycle: {status['agent']['lifecycle_state']}")
            print(f"  Executions: {status['runtime']['execution_count']}")
            print(f"  Total tokens: {status['runtime']['total_tokens_used']}")
            
            # Test 6: Run capability tests
            print("\n6. Running capability tests...")
            test_results = await factory.test_agent_capabilities(architect)
            print(f"✓ Tests completed")
            print(f"  Success rate: {test_results['summary']['success_rate']*100:.0f}%")
            print(f"  Total time: {test_results['summary']['total_time']:.2f}s")
            print(f"  Total tokens: {test_results['summary']['total_tokens']}")
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED SUCCESSFULLY")
            print("Agents are connected to REAL LLM services")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            factory.cleanup()
    
    # Run the test
    asyncio.run(test_agent_factory())
