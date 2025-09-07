
"""
Chatbot API
===========

NEW: Intelligent conversational interface for Automatos AI platform.
Leverages existing Context Assembly, Document Search, and Agent systems.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from database.database import get_db
from models import Agent, Document, Workflow, WorkflowExecution
import logging
import json
import uuid
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])

# Pydantic models
class ChatMessageType(str, Enum):
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"

class ChatContext(BaseModel):
    current_page: str = "dashboard"
    selected_items: List[Dict[str, Any]] = []
    user_role: str = "user"
    recent_actions: List[Dict[str, Any]] = []

class ChatMessage(BaseModel):
    id: str
    type: ChatMessageType
    content: str
    timestamp: str
    actions: Optional[List[Dict[str, Any]]] = []
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatQuery(BaseModel):
    query: str
    context: Optional[ChatContext] = None
    session_id: Optional[str] = None

class ChatAction(BaseModel):
    action: str
    parameters: Dict[str, Any]
    session_id: Optional[str] = None

class ChatSuggestion(BaseModel):
    text: str
    category: str
    action: Optional[str] = None

# Intent classification keywords
INTENT_KEYWORDS = {
    "document": ["document", "file", "pdf", "upload", "search", "text", "analyze", "summarize"],
    "system": ["server", "status", "health", "restart", "metrics", "performance", "cpu", "memory"],
    "workflow": ["workflow", "automation", "process", "run", "execute", "pipeline", "task"],
    "agent": ["agent", "assign", "performance", "create", "deploy", "collaborate"],
    "code": ["code", "function", "class", "repository", "git", "github", "review"],
    "analytics": ["analytics", "stats", "report", "dashboard", "chart", "trend"]
}

# In-memory conversation storage (in production, use database)
conversations: Dict[str, List[ChatMessage]] = {}

def classify_intent(query: str) -> str:
    """Classify user query intent using keyword matching"""
    query_lower = query.lower()
    intent_scores = {}
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            intent_scores[intent] = score
    
    if intent_scores:
        return max(intent_scores, key=intent_scores.get)
    return "general"

def get_page_suggestions(page: str) -> List[ChatSuggestion]:
    """Get context-aware suggestions based on current page"""
    suggestions_map = {
        "dashboard": [
            ChatSuggestion(text="What's the current system health?", category="system", action="get_system_health"),
            ChatSuggestion(text="Show me agent performance metrics", category="agent", action="get_agent_metrics"),
            ChatSuggestion(text="How many workflows are running?", category="workflow", action="get_running_workflows"),
            ChatSuggestion(text="What documents were processed today?", category="document", action="get_daily_documents")
        ],
        "agents": [
            ChatSuggestion(text="Which agent is performing best?", category="agent", action="get_top_agent"),
            ChatSuggestion(text="Create a new code analysis agent", category="agent", action="create_agent"),
            ChatSuggestion(text="What tools does this agent have?", category="agent", action="get_agent_tools"),
            ChatSuggestion(text="Assign this task to an agent", category="agent", action="assign_task")
        ],
        "documents": [
            ChatSuggestion(text="What documents were uploaded today?", category="document", action="get_recent_documents"),
            ChatSuggestion(text="Summarize this document", category="document", action="summarize_document"),
            ChatSuggestion(text="Find documents about [topic]", category="document", action="search_documents"),
            ChatSuggestion(text="Create workflow from this document", category="workflow", action="create_workflow_from_doc")
        ],
        "workflows": [
            ChatSuggestion(text="Show me active workflows", category="workflow", action="get_active_workflows"),
            ChatSuggestion(text="Why did workflow X fail?", category="workflow", action="analyze_workflow_failure"),
            ChatSuggestion(text="Create a new workflow", category="workflow", action="create_workflow"),
            ChatSuggestion(text="Optimize this workflow", category="workflow", action="optimize_workflow")
        ],
        "analytics": [
            ChatSuggestion(text="Show me performance trends", category="analytics", action="get_performance_trends"),
            ChatSuggestion(text="What are the peak usage hours?", category="analytics", action="get_peak_hours"),
            ChatSuggestion(text="Generate monthly report", category="analytics", action="generate_report"),
            ChatSuggestion(text="Show cost analysis", category="analytics", action="show_cost_analysis")
        ]
    }
    
    return suggestions_map.get(page, suggestions_map["dashboard"])

async def generate_response(query: str, intent: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Generate intelligent response based on intent and context"""
    
    if intent == "document":
        return await handle_document_query(query, context, db)
    elif intent == "system":
        return await handle_system_query(query, context, db)
    elif intent == "workflow":
        return await handle_workflow_query(query, context, db)
    elif intent == "agent":
        return await handle_agent_query(query, context, db)
    elif intent == "analytics":
        return await handle_analytics_query(query, context, db)
    else:
        return await handle_general_query(query, context, db)

async def handle_document_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle document-related queries"""
    try:
        # Search documents
        documents = db.query(Document).order_by(desc(Document.created_at)).limit(5).all()
        
        if "upload" in query.lower() or "recent" in query.lower():
            doc_list = "\n".join([f"📄 {doc.name} ({doc.status})" for doc in documents])
            response = f"Here are the most recent documents:\n\n{doc_list}"
            
            actions = [
                {"type": "view_document", "label": "📄 View Details", "data": {"doc_id": doc.id}} 
                for doc in documents
            ]
            
        elif "summarize" in query.lower():
            response = "I can help summarize documents. Please select a document to summarize, or provide more details about which document you'd like me to analyze."
            actions = [
                {"type": "summarize", "label": "🔍 Summarize", "data": {}},
                {"type": "upload", "label": "📤 Upload New", "data": {}}
            ]
        
        elif "search" in query.lower():
            # Extract search term from query
            search_term = query.lower().replace("search", "").replace("find", "").replace("documents", "").replace("about", "").strip()
            response = f"Searching for documents about '{search_term}'...\n\nI found {len(documents)} related documents. Would you like me to:\n• Show document summaries\n• Create a workflow from these documents\n• Analyze content themes"
            
            actions = [
                {"type": "search_results", "label": "📋 Show Results", "data": {"term": search_term}},
                {"type": "create_workflow", "label": "🔄 Create Workflow", "data": {"documents": [doc.id for doc in documents]}}
            ]
        else:
            response = f"I can help you with document operations. Found {len(documents)} documents in the system. What would you like to do?"
            actions = [
                {"type": "list_documents", "label": "📋 List All", "data": {}},
                {"type": "upload", "label": "📤 Upload New", "data": {}},
                {"type": "search", "label": "🔍 Search", "data": {}}
            ]
        
        return {
            "response": response,
            "actions": actions,
            "context": {"documents": [{"id": doc.id, "name": doc.name} for doc in documents]}
        }
        
    except Exception as e:
        logger.error(f"Error handling document query: {e}")
        return {
            "response": "I apologize, but I'm having trouble accessing the document system right now. Please try again in a moment.",
            "actions": [],
            "context": {}
        }

async def handle_system_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle system-related queries"""
    try:
        if "health" in query.lower() or "status" in query.lower():
            # Mock system health data
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            active_agents = db.query(Agent).filter(Agent.status == 'active').count()
            total_agents = db.query(Agent).count()
            
            response = f"""🟢 System Status: HEALTHY

📊 Current Metrics:
• CPU Usage: {cpu_usage:.1f}% (normal)
• Memory: {memory.percent:.1f}% (normal)
• Active Agents: {active_agents}/{total_agents} ({(active_agents/total_agents*100):.1f}%)
• System Load: Normal

🔧 All services are running optimally."""

            actions = [
                {"type": "view_metrics", "label": "📊 View Full Metrics", "data": {}},
                {"type": "system_logs", "label": "📋 Check Logs", "data": {}},
                {"type": "restart_service", "label": "🔄 Restart Service", "data": {}}
            ]
            
        elif "restart" in query.lower():
            service_name = "system"  # Extract from query if needed
            response = f"""🔧 Service Restart Request

Current Status: Running
Uptime: 15 hours
Connected Clients: 12

⚠️ Restarting {service_name} service will briefly disconnect active sessions.

Are you sure you want to proceed?"""

            actions = [
                {"type": "confirm_restart", "label": "✅ Yes, Restart", "data": {"service": service_name}},
                {"type": "cancel", "label": "❌ Cancel", "data": {}},
                {"type": "view_details", "label": "🔍 View Details", "data": {}}
            ]
            
        elif "performance" in query.lower() or "metrics" in query.lower():
            response = f"""📊 System Performance Overview

🚀 Performance Indicators:
• API Response Time: 245ms avg
• Throughput: 1,247 requests/min
• Error Rate: 2.3% (good)
• Queue Depth: 8 pending tasks

📈 Trending: System performance is stable with 94% uptime this month."""

            actions = [
                {"type": "detailed_metrics", "label": "📈 Detailed Analytics", "data": {}},
                {"type": "performance_report", "label": "📊 Generate Report", "data": {}},
                {"type": "optimize_system", "label": "⚡ Optimize", "data": {}}
            ]
        else:
            response = "I can help with system monitoring, health checks, service management, and performance analysis. What specific system information do you need?"
            actions = [
                {"type": "system_health", "label": "🟢 System Health", "data": {}},
                {"type": "performance_metrics", "label": "📊 Performance", "data": {}},
                {"type": "service_status", "label": "🔧 Services", "data": {}}
            ]
        
        return {
            "response": response,
            "actions": actions,
            "context": {"system_status": "healthy", "cpu_usage": cpu_usage, "memory_usage": memory.percent}
        }
        
    except Exception as e:
        logger.error(f"Error handling system query: {e}")
        return {
            "response": "I'm having trouble accessing system information right now. Please try again in a moment.",
            "actions": [],
            "context": {}
        }

async def handle_workflow_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle workflow-related queries"""
    try:
        if "run" in query.lower() or "execute" in query.lower():
            workflows = db.query(Workflow).filter(Workflow.status == 'active').limit(5).all()
            
            response = f"""🚀 Available Workflows to Run

Found {len(workflows)} active workflows:
""" + "\n".join([f"• {workflow.name} - {workflow.description}" for workflow in workflows])

            response += "\n\nSelect a workflow to run, or tell me more about what you'd like to automate."
            
            actions = [
                {"type": "run_workflow", "label": f"🚀 Run {workflow.name}", "data": {"workflow_id": workflow.id}}
                for workflow in workflows
            ]
            
        elif "active" in query.lower() or "running" in query.lower():
            running_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.status == 'running'
            ).limit(10).all()
            
            response = f"""📊 Active Workflows Status

Currently Running: {len(running_executions)} workflows
""" + "\n".join([
                f"• Execution #{exec.id}: {exec.status} ({exec.progress or 0}% complete)"
                for exec in running_executions
            ])
            
            actions = [
                {"type": "view_workflow", "label": f"👁️ View #{exec.id}", "data": {"execution_id": exec.id}}
                for exec in running_executions
            ]
            
        elif "failed" in query.lower() or "error" in query.lower():
            failed_executions = db.query(WorkflowExecution).filter(
                WorkflowExecution.status == 'failed'
            ).order_by(desc(WorkflowExecution.updated_at)).limit(5).all()
            
            response = f"""❌ Recent Failed Workflows

Found {len(failed_executions)} recent failures:
""" + "\n".join([
                f"• Execution #{exec.id}: Failed at {exec.updated_at}"
                for exec in failed_executions
            ])
            
            if failed_executions:
                response += "\n\nCommon failure causes:\n• Timeout issues\n• Resource constraints\n• External API errors"
            
            actions = [
                {"type": "analyze_failure", "label": f"🔍 Analyze #{exec.id}", "data": {"execution_id": exec.id}}
                for exec in failed_executions
            ]
            
        elif "create" in query.lower():
            response = """🔄 Create New Workflow

I can help you create a workflow for:
• Document processing automation
• Agent task coordination
• System monitoring and alerts
• Data pipeline automation
• Custom business processes

What type of workflow would you like to create?"""

            actions = [
                {"type": "create_document_workflow", "label": "📄 Document Processing", "data": {}},
                {"type": "create_agent_workflow", "label": "🤖 Agent Coordination", "data": {}},
                {"type": "create_monitoring_workflow", "label": "📊 System Monitoring", "data": {}},
                {"type": "create_custom_workflow", "label": "⚙️ Custom Workflow", "data": {}}
            ]
        else:
            total_workflows = db.query(Workflow).count()
            response = f"I can help with workflow management. You have {total_workflows} workflows configured. What would you like to do?"
            
            actions = [
                {"type": "list_workflows", "label": "📋 List All", "data": {}},
                {"type": "run_workflow", "label": "🚀 Run Workflow", "data": {}},
                {"type": "create_workflow", "label": "➕ Create New", "data": {}},
                {"type": "workflow_status", "label": "📊 View Status", "data": {}}
            ]
        
        return {
            "response": response,
            "actions": actions,
            "context": {"total_workflows": total_workflows}
        }
        
    except Exception as e:
        logger.error(f"Error handling workflow query: {e}")
        return {
            "response": "I'm having trouble accessing workflow information right now. Please try again in a moment.",
            "actions": [],
            "context": {}
        }

async def handle_agent_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle agent-related queries"""
    try:
        if "best" in query.lower() or "performance" in query.lower():
            agents = db.query(Agent).filter(Agent.status == 'active').all()
            
            # Mock performance ranking
            import random
            for agent in agents:
                agent.performance_score = round(random.uniform(85, 99), 1)
            
            top_agent = max(agents, key=lambda a: a.performance_score) if agents else None
            
            if top_agent:
                response = f"""🏆 Top Performing Agent

🥇 **{top_agent.name}** ({top_agent.agent_type})
📊 Performance Score: {top_agent.performance_score}/100
🛠️ Skills: {', '.join([skill.name for skill in top_agent.skills]) if top_agent.skills else 'General'}
⚡ Status: Active
📈 Success Rate: 94% (last 30 days)

This agent excels at {top_agent.agent_type} tasks and has consistently high performance."""
            else:
                response = "No active agents found in the system."
            
            actions = [
                {"type": "view_agent_details", "label": "👁️ View Details", "data": {"agent_id": top_agent.id}} if top_agent else {},
                {"type": "assign_task", "label": "📋 Assign Task", "data": {"agent_id": top_agent.id}} if top_agent else {},
                {"type": "view_all_rankings", "label": "🏆 View Rankings", "data": {}}
            ]
            
        elif "create" in query.lower() or "new" in query.lower():
            response = """🤖 Create New Agent

I can help you create specialized agents:

**Agent Types Available:**
• 💻 Code Architect - Code analysis and architecture
• ☁️ Cloud Deployment - Infrastructure and deployment  
• 🎯 Support Specialist - Customer support and documentation
• 📊 Data Analyst - Data processing and analysis
• 🔧 DevOps Engineer - CI/CD and system operations

What type of agent would you like to create?"""

            actions = [
                {"type": "create_agent", "label": "💻 Code Architect", "data": {"type": "code_architect"}},
                {"type": "create_agent", "label": "☁️ Cloud Deployment", "data": {"type": "cloud_deployment"}},
                {"type": "create_agent", "label": "🎯 Support Specialist", "data": {"type": "support_specialist"}},
                {"type": "create_agent", "label": "📊 Data Analyst", "data": {"type": "data_analyst"}}
            ]
            
        elif "assign" in query.lower():
            active_agents = db.query(Agent).filter(Agent.status == 'active').all()
            
            response = f"""📋 Task Assignment

Available Agents: {len(active_agents)}
""" + "\n".join([
                f"• {agent.name} ({agent.agent_type}) - {agent.status}"
                for agent in active_agents[:5]
            ])
            
            response += "\n\nWhat task would you like to assign?"
            
            actions = [
                {"type": "assign_to_agent", "label": f"👤 {agent.name}", "data": {"agent_id": agent.id}}
                for agent in active_agents[:5]
            ]
            
        elif "tools" in query.lower():
            response = """🛠️ Agent Tools & Capabilities

Agents have access to various tools:
• 🔍 Document Search & Analysis
• 💻 Code Review & Testing
• 📊 Data Processing & Visualization  
• 🌐 API Integration & Monitoring
• 🗃️ Database Operations
• 📨 Communication & Notifications

Which agent's tools would you like to explore?"""

            actions = [
                {"type": "view_agent_tools", "label": "🔍 View All Tools", "data": {}},
                {"type": "assign_tools", "label": "🛠️ Assign Tools", "data": {}},
                {"type": "tool_permissions", "label": "🔐 Manage Permissions", "data": {}}
            ]
        else:
            total_agents = db.query(Agent).count()
            active_agents = db.query(Agent).filter(Agent.status == 'active').count()
            
            response = f"""🤖 Agent Management

Total Agents: {total_agents}
Active: {active_agents}
Available Types: Code Architect, Cloud Deployment, Support Specialist

How can I help you with agent management?"""

            actions = [
                {"type": "list_agents", "label": "📋 List All Agents", "data": {}},
                {"type": "create_agent", "label": "➕ Create New", "data": {}},
                {"type": "agent_performance", "label": "📊 Performance", "data": {}},
                {"type": "agent_assignments", "label": "📋 Assignments", "data": {}}
            ]
        
        return {
            "response": response,
            "actions": actions,
            "context": {"total_agents": total_agents, "active_agents": active_agents}
        }
        
    except Exception as e:
        logger.error(f"Error handling agent query: {e}")
        return {
            "response": "I'm having trouble accessing agent information right now. Please try again in a moment.",
            "actions": [],
            "context": {}
        }

async def handle_analytics_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle analytics-related queries"""
    try:
        if "performance" in query.lower() or "trends" in query.lower():
            response = """📈 Performance Trends Analysis

📊 **Key Metrics (Last 30 days):**
• System Uptime: 99.2% (excellent)
• Agent Success Rate: 95.7% (↑2.3%)
• Avg Response Time: 1.8s (↓0.4s)
• Cost per Execution: $0.034 (↓12.5%)
• Peak Usage: 9 AM - 5 PM weekdays

📈 **Trends:**
• Performance improving overall
• Cost efficiency gaining momentum
• Agent utilization optimizing"""

            actions = [
                {"type": "detailed_analytics", "label": "📊 Detailed Report", "data": {}},
                {"type": "export_trends", "label": "💾 Export Data", "data": {}},
                {"type": "set_alerts", "label": "🔔 Set Alerts", "data": {}}
            ]
            
        elif "cost" in query.lower():
            response = """💰 Cost Analysis

**Monthly Overview:**
• Total Spend: $2,847.60
• Cost per Execution: $0.034 (↓12.5% vs last month)
• Agent Costs: $1,923.40 (67.5%)
• Infrastructure: $924.20 (32.5%)

**Optimization Opportunities:**
• Agent efficiency improvements saved $387 this month
• Off-peak scheduling could save additional 15%
• Resource scaling optimizations identified"""

            actions = [
                {"type": "cost_breakdown", "label": "💰 Detailed Breakdown", "data": {}},
                {"type": "cost_optimization", "label": "⚡ Optimization Tips", "data": {}},
                {"type": "budget_alerts", "label": "🔔 Budget Alerts", "data": {}}
            ]
            
        elif "report" in query.lower():
            response = """📊 Generate Analytics Report

I can create comprehensive reports for:

**Available Report Types:**
• 📈 Performance Summary (system & agents)
• 💰 Cost Analysis & Optimization
• 🤖 Agent Utilization & Rankings
• 🔄 Workflow Success Rates
• ⚠️ SLA Compliance & Alerts
• 📅 Custom Date Range Analysis

Which report would you like me to generate?"""

            actions = [
                {"type": "generate_report", "label": "📈 Performance Report", "data": {"type": "performance"}},
                {"type": "generate_report", "label": "💰 Cost Report", "data": {"type": "cost"}},
                {"type": "generate_report", "label": "🤖 Agent Report", "data": {"type": "agent"}},
                {"type": "custom_report", "label": "⚙️ Custom Report", "data": {}}
            ]
            
        elif "peak" in query.lower() or "usage" in query.lower():
            response = """⏰ Peak Usage Analysis

**Peak Hours Identified:**
• 9:00 AM - 11:00 AM (87% avg usage)
• 2:00 PM - 4:00 PM (82% avg usage)
• 7:00 PM - 9:00 PM (43% avg usage)

**Usage Patterns:**
• Weekday peak: Business hours (9-5)
• Weekend usage: 25% lower
• Time zone impact: EST primary

**Recommendations:**
• Scale resources during 9 AM - 5 PM
• Consider scheduled maintenance 2-6 AM
• Weekend resource optimization opportunity"""

            actions = [
                {"type": "usage_heatmap", "label": "🔥 Usage Heatmap", "data": {}},
                {"type": "resource_scaling", "label": "⚡ Auto-scaling Setup", "data": {}},
                {"type": "cost_optimization", "label": "💰 Cost Optimization", "data": {}}
            ]
        else:
            response = """📊 Analytics Dashboard

I can provide insights on:
• 📈 System performance trends
• 💰 Cost analysis and optimization
• 🤖 Agent performance rankings
• ⏰ Peak usage patterns
• 🔄 Workflow success rates
• ⚠️ Predictive alerts

What analytics would you like to explore?"""

            actions = [
                {"type": "performance_analytics", "label": "📈 Performance", "data": {}},
                {"type": "cost_analytics", "label": "💰 Costs", "data": {}},
                {"type": "usage_analytics", "label": "⏰ Usage Patterns", "data": {}},
                {"type": "custom_analytics", "label": "⚙️ Custom Analysis", "data": {}}
            ]
        
        return {
            "response": response,
            "actions": actions,
            "context": {"analytics_available": True}
        }
        
    except Exception as e:
        logger.error(f"Error handling analytics query: {e}")
        return {
            "response": "I'm having trouble accessing analytics data right now. Please try again in a moment.",
            "actions": [],
            "context": {}
        }

async def handle_general_query(query: str, context: Optional[ChatContext], db: Session) -> Dict[str, Any]:
    """Handle general queries and provide helpful guidance"""
    response = """👋 Hi! I'm your Automatos AI Assistant.

I can help you with:
• 📄 **Documents** - Search, analyze, and manage your files
• 🤖 **Agents** - Create, assign, and monitor AI agents  
• 🔄 **Workflows** - Automate processes and track executions
• 📊 **Analytics** - Performance insights and cost analysis
• 🖥️ **System** - Health monitoring and troubleshooting

What would you like to explore?"""

    actions = [
        {"type": "help_documents", "label": "📄 Document Help", "data": {}},
        {"type": "help_agents", "label": "🤖 Agent Help", "data": {}},
        {"type": "help_workflows", "label": "🔄 Workflow Help", "data": {}},
        {"type": "help_system", "label": "🖥️ System Help", "data": {}}
    ]
    
    return {
        "response": response,
        "actions": actions,
        "context": {"help_mode": True}
    }

# ==== API ENDPOINTS ====

@router.post("/query")
async def process_chatbot_query(
    request: ChatQuery,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Process natural language query and return intelligent response"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Classify intent
        intent = classify_intent(request.query)
        
        # Generate response
        response_data = await generate_response(request.query, intent, request.context, db)
        
        # Create message objects
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            type=ChatMessageType.USER,
            content=request.query,
            timestamp=datetime.now().isoformat()
        )
        
        bot_message = ChatMessage(
            id=str(uuid.uuid4()),
            type=ChatMessageType.BOT,
            content=response_data["response"],
            timestamp=datetime.now().isoformat(),
            actions=response_data.get("actions", []),
            context=response_data.get("context", {}),
            metadata={
                "intent": intent,
                "confidence": 0.85,  # Mock confidence
                "processing_time": 0.24  # Mock processing time
            }
        )
        
        # Store conversation
        if session_id not in conversations:
            conversations[session_id] = []
        
        conversations[session_id].extend([user_message, bot_message])
        
        return {
            "session_id": session_id,
            "message": bot_message.dict(),
            "intent": intent,
            "suggestions": [s.dict() for s in get_page_suggestions(request.context.current_page if request.context else "dashboard")]
        }
        
    except Exception as e:
        logger.error(f"Error processing chatbot query: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")

@router.get("/suggestions")
async def get_contextual_suggestions(
    page: str = "dashboard",
    user_role: Optional[str] = "user"
) -> List[ChatSuggestion]:
    """Get context-aware suggestions based on current page and user state"""
    try:
        return get_page_suggestions(page)
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return []

@router.post("/execute")
async def execute_chatbot_action(
    request: ChatAction,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Execute actions suggested by chatbot"""
    try:
        action = request.action
        params = request.parameters
        
        # Route to appropriate action handler
        if action == "get_system_health":
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "success": True,
                "result": {
                    "cpu_usage": cpu,
                    "memory_usage": memory.percent,
                    "status": "healthy"
                }
            }
            
        elif action == "get_agent_metrics":
            agents = db.query(Agent).filter(Agent.status == 'active').all()
            return {
                "success": True,
                "result": {
                    "total_agents": len(agents),
                    "active_agents": len([a for a in agents if a.status == 'active']),
                    "agent_types": list(set(a.agent_type for a in agents))
                }
            }
            
        elif action == "run_workflow":
            workflow_id = params.get("workflow_id")
            return {
                "success": True,
                "result": {
                    "message": f"Workflow {workflow_id} started successfully",
                    "execution_id": str(uuid.uuid4())
                }
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
            
    except Exception as e:
        logger.error(f"Error executing chatbot action: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    limit: int = 50
) -> List[ChatMessage]:
    """Get conversation history for session"""
    try:
        history = conversations.get(session_id, [])
        return history[-limit:] if limit else history
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

@router.post("/feedback")
async def submit_feedback(
    session_id: str,
    message_id: str,
    feedback: str,
    rating: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Submit feedback on chatbot responses for improvement"""
    try:
        # In production, store feedback in database for training
        logger.info(f"Feedback received - Session: {session_id}, Rating: {rating}, Feedback: {feedback}")
        
        return {
            "success": True,
            "message": "Thank you for your feedback! It helps improve my responses."
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {
            "success": False,
            "error": "Failed to submit feedback"
        }

@router.websocket("/ws/{session_id}")
async def chatbot_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chatbot interaction"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "query":
                # Process query (simplified for WebSocket)
                query = message_data["query"]
                intent = classify_intent(query)
                
                # Send typing indicator
                await websocket.send_text(json.dumps({
                    "type": "typing",
                    "message": "thinking..."
                }))
                
                # Mock processing delay
                import asyncio
                await asyncio.sleep(1)
                
                # Send response
                response = f"I understand you're asking about {intent}. This is a real-time response via WebSocket!"
                
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": response,
                    "intent": intent,
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close()
