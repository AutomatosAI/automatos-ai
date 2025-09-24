
import { NextRequest, NextResponse } from 'next/server'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const page = searchParams.get('page') || 'dashboard'
    const userRole = searchParams.get('user_role') || 'user'

    // Try to get suggestions from the orchestrator backend
    const response = await fetch(
      `${process.env.API_BASE_URL || 'http://localhost:8002'}/api/chat/suggestions?page=${page}&user_role=${userRole}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      }
    )

    if (response.ok) {
      const data = await response.json()
      return NextResponse.json(data.suggestions || data)
    } else {
      throw new Error(`Backend responded with status ${response.status}`)
    }
  } catch (error) {
    console.error('Chatbot suggestions error:', error)
    
    // Fallback contextual suggestions based on page
    const { searchParams } = new URL(request.url)
    const page = searchParams.get('page') || 'dashboard'
    
    const suggestionMap: Record<string, any[]> = {
      dashboard: [
        { text: "What's the current system health?", category: "system", action: "system_health" },
        { text: "Show me today's agent performance", category: "agent", action: "agent_status" },
        { text: "How many workflows are running?", category: "workflow", action: "workflow_status" }
      ],
      agents: [
        { text: "Which agent performs best?", category: "agent", action: "top_agents" },
        { text: "Show me agent utilization", category: "agent", action: "agent_metrics" },
        { text: "How can I optimize agent performance?", category: "agent", action: "agent_optimization" }
      ],
      documents: [
        { text: "What documents were processed today?", category: "document", action: "recent_documents" },
        { text: "Analyze my latest uploads", category: "document", action: "document_analysis" },
        { text: "Find documents with errors", category: "document", action: "document_errors" }
      ],
      workflows: [
        { text: "Show active workflow status", category: "workflow", action: "active_workflows" },
        { text: "Why did my workflow fail?", category: "workflow", action: "workflow_debugging" },
        { text: "Optimize workflow performance", category: "workflow", action: "workflow_optimization" }
      ],
      analytics: [
        { text: "Show performance trends", category: "analytics", action: "performance_trends" },
        { text: "Generate cost analysis report", category: "analytics", action: "cost_analysis" },
        { text: "What are peak usage hours?", category: "analytics", action: "usage_patterns" }
      ]
    }
    
    const suggestions = suggestionMap[page] || suggestionMap.dashboard
    return NextResponse.json(suggestions)
  }
}
