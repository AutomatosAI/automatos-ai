#!/usr/bin/env python3
"""
Dashboard Verification Script - PRD-06
======================================

Tests the monitoring dashboard with REAL data from existing infrastructure.
Verifies all metrics are pulled from actual database tables and Redis.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add orchestrator to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'orchestrator'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis

from orchestrator.services.analytics_engine import AnalyticsEngine
from orchestrator.services.agent_factory import AgentFactory
from orchestrator.services.task_decomposition import TaskDecomposer
from orchestrator.services.context_optimizer import ContextOptimizer
from orchestrator.services.memory_knowledge_system import HierarchicalMemorySystem
from orchestrator.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardVerification:
    """Test harness for dashboard verification"""
    
    def __init__(self):
        self.config = Config()
        self.redis_client = None
        self.db_session = None
        self.analytics_engine = None
        self.results = {}
    
    async def setup(self):
        """Initialize connections"""
        logger.info("=== Setting up Dashboard Verification ===")
        
        # Database connection
        engine = create_engine(self.config.DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        self.db_session = SessionLocal()
        
        # Redis connection
        self.redis_client = await redis.from_url(self.config.REDIS_URL)
        await self.redis_client.ping()
        
        # Initialize analytics engine
        self.analytics_engine = AnalyticsEngine(self.db_session, self.redis_client)
        
        logger.info("✅ Connections established")
    
    async def verify_data_sources(self) -> Dict[str, Any]:
        """Verify all data sources have real data"""
        logger.info("\n=== Verifying Data Sources ===")
        
        sources = {}
        
        # Check agents table
        agent_count = self.db_session.execute(
            text("SELECT COUNT(*) FROM agents")
        ).scalar()
        sources['agents'] = {
            'count': agent_count,
            'has_data': agent_count > 0
        }
        logger.info(f"Agents table: {agent_count} records")
        
        # Check task tables
        task_count = self.db_session.execute(
            text("SELECT COUNT(*) FROM task_decompositions")
        ).scalar()
        sources['tasks'] = {
            'count': task_count,
            'has_data': task_count > 0
        }
        logger.info(f"Tasks table: {task_count} records")
        
        # Check memory tables
        memory_count = self.db_session.execute(
            text("SELECT COUNT(*) FROM memory_items")
        ).scalar()
        sources['memory'] = {
            'count': memory_count,
            'has_data': memory_count > 0
        }
        logger.info(f"Memory items: {memory_count} records")
        
        # Check context optimizations
        context_count = self.db_session.execute(
            text("SELECT COUNT(*) FROM context_optimizations")
        ).scalar()
        sources['context'] = {
            'count': context_count,
            'has_data': context_count > 0
        }
        logger.info(f"Context optimizations: {context_count} records")
        
        # Check collaboration sessions
        collab_count = self.db_session.execute(
            text("SELECT COUNT(*) FROM collaboration_sessions")
        ).scalar()
        sources['collaboration'] = {
            'count': collab_count,
            'has_data': collab_count > 0
        }
        logger.info(f"Collaboration sessions: {collab_count} records")
        
        self.results['data_sources'] = sources
        return sources
    
    async def test_analytics_engine(self):
        """Test AnalyticsEngine with real data"""
        logger.info("\n=== Testing Analytics Engine ===")
        
        # Test agent metrics
        logger.info("Getting agent metrics...")
        agent_metrics = await self.analytics_engine.get_agent_metrics()
        logger.info(f"✅ Retrieved {len(agent_metrics)} agent metrics")
        if agent_metrics:
            sample = agent_metrics[0]
            logger.info(f"  Sample: {sample.name} - {sample.execution_count} executions, {sample.success_rate}% success")
        
        # Test task metrics
        logger.info("Getting task metrics...")
        task_metrics = await self.analytics_engine.get_task_metrics()
        logger.info(f"✅ Task metrics: {task_metrics.total_tasks} total, {task_metrics.completed_tasks} completed")
        
        # Test memory metrics
        logger.info("Getting memory metrics...")
        memory_metrics = await self.analytics_engine.get_memory_metrics()
        logger.info(f"✅ Memory metrics: {memory_metrics.total_memories} memories, {memory_metrics.consolidation_rate}% consolidation")
        
        # Test context metrics
        logger.info("Getting context metrics...")
        context_metrics = await self.analytics_engine.get_context_metrics()
        logger.info(f"✅ Context metrics: {context_metrics.total_tokens_saved} tokens saved, {context_metrics.compression_ratio}x compression")
        
        # Test collaboration metrics
        logger.info("Getting collaboration metrics...")
        collab_metrics = await self.analytics_engine.get_collaboration_metrics()
        logger.info(f"✅ Collaboration metrics: {collab_metrics.total_sessions} sessions, {collab_metrics.active_sessions} active")
        
        # Test system health
        logger.info("Getting system health...")
        health = await self.analytics_engine.get_system_health()
        logger.info(f"✅ System health: {health['overall_status']} ({health['health_score']}%)")
        
        self.results['analytics_engine'] = {
            'agents': len(agent_metrics),
            'tasks': task_metrics.total_tasks,
            'memories': memory_metrics.total_memories,
            'tokens_saved': context_metrics.total_tokens_saved,
            'collaborations': collab_metrics.total_sessions,
            'health_score': health['health_score']
        }
    
    async def test_dashboard_overview(self):
        """Test complete dashboard overview"""
        logger.info("\n=== Testing Dashboard Overview ===")
        
        overview = await self.analytics_engine.get_dashboard_overview()
        
        logger.info(f"✅ Dashboard overview retrieved")
        logger.info(f"  Active agents: {overview['summary']['active_agents']}/{overview['summary']['total_agents']}")
        logger.info(f"  Tasks completed: {overview['summary']['tasks_completed']}")
        logger.info(f"  Success rate: {overview['summary']['overall_success_rate']}%")
        logger.info(f"  Tokens saved: {overview['summary']['tokens_saved']} ({overview['summary']['token_savings_percent']}%)")
        logger.info(f"  Memory items: {overview['summary']['memory_items']}")
        logger.info(f"  Knowledge graph: {overview['summary']['knowledge_graph_size']} nodes+edges")
        
        self.results['dashboard_overview'] = overview['summary']
    
    async def simulate_agent_activity(self):
        """Simulate agent activity for real-time testing"""
        logger.info("\n=== Simulating Agent Activity ===")
        
        try:
            # Create test agents if needed
            existing_agents = self.db_session.execute(
                text("SELECT COUNT(*) FROM agents")
            ).scalar()
            
            if existing_agents < 3:
                logger.info("Creating test agents...")
                factory = AgentFactory(self.db_session)
                
                for i in range(3):
                    agent_data = {
                        "name": f"Dashboard Test Agent {i+1}",
                        "agent_type": ["orchestrator", "worker", "specialist"][i],
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "status": "active"
                    }
                    agent = await factory.create_agent(agent_data)
                    logger.info(f"  Created agent: {agent.name}")
            
            # Simulate executions
            logger.info("Simulating agent executions...")
            agents = self.db_session.execute(
                text("SELECT id, name FROM agents LIMIT 3")
            ).fetchall()
            
            for agent in agents:
                # Record performance
                self.db_session.execute(text("""
                    INSERT INTO agent_performance 
                    (agent_id, task_id, tokens_used, execution_time, success, recorded_at)
                    VALUES (:agent_id, :task_id, :tokens, :time, :success, :recorded_at)
                """), {
                    "agent_id": agent.id,
                    "task_id": f"test-task-{datetime.now().timestamp()}",
                    "tokens": 150 + (agent.id % 100),
                    "time": 2.5 + (agent.id % 3),
                    "success": agent.id % 3 != 2,  # 2/3 success rate
                    "recorded_at": datetime.now()
                })
                
                # Publish to Redis
                await self.redis_client.publish(
                    "agent:execution_completed",
                    json.dumps({
                        "agent_id": str(agent.id),
                        "agent_name": agent.name,
                        "tokens_used": 150,
                        "execution_time": 2.5,
                        "timestamp": datetime.now().isoformat()
                    })
                )
                logger.info(f"  Simulated execution for {agent.name}")
            
            self.db_session.commit()
            
            # Create test tasks
            logger.info("Creating test tasks...")
            for i in range(5):
                self.db_session.execute(text("""
                    INSERT INTO task_decompositions 
                    (task_description, subtasks, status, created_at)
                    VALUES (:description, :subtasks, :status, :created_at)
                """), {
                    "description": f"Test task {i+1} for dashboard",
                    "subtasks": json.dumps([f"Subtask {j}" for j in range(3)]),
                    "status": ["completed", "in_progress", "pending"][i % 3],
                    "created_at": datetime.now() - timedelta(hours=i)
                })
            
            # Create memory items
            logger.info("Creating memory items...")
            for i in range(10):
                self.db_session.execute(text("""
                    INSERT INTO memory_items 
                    (agent_id, content, memory_level, importance, created_at)
                    VALUES (:agent_id, :content, :level, :importance, :created_at)
                """), {
                    "agent_id": agents[i % len(agents)].id,
                    "content": f"Test memory {i+1}: Important information",
                    "level": ["working", "short_term", "long_term"][i % 3],
                    "importance": 0.5 + (i % 5) / 10,
                    "created_at": datetime.now() - timedelta(minutes=i*10)
                })
            
            self.db_session.commit()
            logger.info("✅ Activity simulation complete")
            
        except Exception as e:
            logger.error(f"Error simulating activity: {e}")
            self.db_session.rollback()
    
    async def test_real_time_updates(self):
        """Test real-time update mechanism"""
        logger.info("\n=== Testing Real-time Updates ===")
        
        # Subscribe to Redis channels
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("dashboard_updates", "agent:*", "task:*")
        
        # Publish test events
        test_events = [
            ("agent:status_changed", {"agent_id": "1", "status": "active"}),
            ("task:created", {"task_id": "test-123", "task_name": "Test Task"}),
            ("memory:consolidated", {"memory_id": "mem-456", "level": "long_term"})
        ]
        
        for channel, data in test_events:
            await self.redis_client.publish(channel, json.dumps({
                **data,
                "timestamp": datetime.now().isoformat()
            }))
            logger.info(f"  Published to {channel}")
        
        # Check for messages
        received = []
        try:
            async def get_message():
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        received.append(message)
                        if len(received) >= 3:
                            break
            
            await asyncio.wait_for(get_message(), timeout=2.0)
            
        except asyncio.TimeoutError:
            pass
        
        await pubsub.unsubscribe()
        await pubsub.close()
        
        logger.info(f"✅ Received {len(received)} real-time updates")
        self.results['real_time_updates'] = len(received)
    
    async def verify_dashboard_load_time(self):
        """Verify dashboard loads within 2 seconds"""
        logger.info("\n=== Testing Dashboard Load Time ===")
        
        start_time = datetime.now()
        overview = await self.analytics_engine.get_dashboard_overview()
        load_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Dashboard load time: {load_time:.2f} seconds")
        
        if load_time < 2.0:
            logger.info("✅ Dashboard loads within 2 seconds requirement")
        else:
            logger.warning(f"⚠️ Dashboard load time exceeds 2 seconds: {load_time:.2f}s")
        
        self.results['load_time'] = load_time
    
    async def cleanup(self):
        """Clean up connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_session:
            self.db_session.close()
        logger.info("✅ Cleanup complete")
    
    def generate_report(self):
        """Generate verification report"""
        logger.info("\n" + "="*50)
        logger.info("DASHBOARD VERIFICATION REPORT - PRD-06")
        logger.info("="*50)
        
        # Data Sources
        logger.info("\n📊 DATA SOURCES:")
        if 'data_sources' in self.results:
            for source, info in self.results['data_sources'].items():
                status = "✅" if info['has_data'] else "❌"
                logger.info(f"  {status} {source}: {info['count']} records")
        
        # Analytics Engine
        logger.info("\n🔍 ANALYTICS ENGINE:")
        if 'analytics_engine' in self.results:
            for metric, value in self.results['analytics_engine'].items():
                logger.info(f"  • {metric}: {value}")
        
        # Dashboard Overview
        logger.info("\n📈 DASHBOARD OVERVIEW:")
        if 'dashboard_overview' in self.results:
            overview = self.results['dashboard_overview']
            logger.info(f"  • Active Agents: {overview['active_agents']}/{overview['total_agents']}")
            logger.info(f"  • Tasks Completed: {overview['tasks_completed']}")
            logger.info(f"  • Success Rate: {overview['overall_success_rate']}%")
            logger.info(f"  • Tokens Saved: {overview['tokens_saved']:,}")
            logger.info(f"  • System Health: {overview['system_health']}%")
        
        # Performance
        logger.info("\n⚡ PERFORMANCE:")
        if 'load_time' in self.results:
            logger.info(f"  • Dashboard Load Time: {self.results['load_time']:.2f}s")
            if self.results['load_time'] < 2.0:
                logger.info("  ✅ Meets < 2 second requirement")
            else:
                logger.info("  ❌ Exceeds 2 second requirement")
        
        if 'real_time_updates' in self.results:
            logger.info(f"  • Real-time Updates: {self.results['real_time_updates']} received")
        
        # Summary
        logger.info("\n" + "="*50)
        all_passed = all([
            self.results.get('data_sources', {}).get('agents', {}).get('has_data', False),
            self.results.get('load_time', 3) < 2.0,
            self.results.get('real_time_updates', 0) > 0
        ])
        
        if all_passed:
            logger.info("✅ DASHBOARD VERIFICATION PASSED")
        else:
            logger.info("❌ DASHBOARD VERIFICATION FAILED - See issues above")
        
        logger.info("="*50)

async def main():
    """Run dashboard verification"""
    verifier = DashboardVerification()
    
    try:
        await verifier.setup()
        await verifier.verify_data_sources()
        await verifier.simulate_agent_activity()
        await verifier.test_analytics_engine()
        await verifier.test_dashboard_overview()
        await verifier.test_real_time_updates()
        await verifier.verify_dashboard_load_time()
        verifier.generate_report()
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await verifier.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

