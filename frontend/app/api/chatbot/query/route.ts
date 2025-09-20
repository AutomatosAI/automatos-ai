
import { NextRequest, NextResponse } from 'next/server'

// Helper function to determine intent from query
function analyzeIntent(query: string, context: any) {
  const lowerQuery = query.toLowerCase()
  
  // System health related
  if (lowerQuery.includes('health') || lowerQuery.includes('status') || lowerQuery.includes('system')) {
    return { intent: 'system_health', confidence: 0.9 }
  }
  
  // Agent related
  if (lowerQuery.includes('agent') || lowerQuery.includes('bot')) {
    return { intent: 'agent_management', confidence: 0.9 }
  }
  
  // Document related
  if (lowerQuery.includes('document') || lowerQuery.includes('file') || lowerQuery.includes('upload')) {
    return { intent: 'document_management', confidence: 0.9 }
  }
  
  // Workflow related
  if (lowerQuery.includes('workflow') || lowerQuery.includes('process') || lowerQuery.includes('automation')) {
    return { intent: 'workflow_management', confidence: 0.9 }
  }
  
  // Context based intent
  if (context.currentPage === 'agents') {
    return { intent: 'agent_management', confidence: 0.7 }
  }
  if (context.currentPage === 'documents') {
    return { intent: 'document_management', confidence: 0.7 }
  }
  if (context.currentPage === 'workflows') {
    return { intent: 'workflow_management', confidence: 0.7 }
  }
  
  return { intent: 'general', confidence: 0.5 }
}

// Helper function to handle different intents
async function handleIntent(intent: string, query: string, context: any) {
  const baseUrl = process.env.API_BASE_URL || 'http://localhost:8002'
  
  try {
    switch (intent) {
      case 'system_health':
        const healthResponse = await fetch(`${baseUrl}/health`)
        if (healthResponse.ok) {
          const healthData = await healthResponse.json()
          return {
            content: `🟢 **System Health Status**\n\nOverall Status: ${healthData.status || 'Operational'}\n\nServices:\n${healthData.services ? Object.entries(healthData.services).map(([service, status]) => `• ${service}: ${status}`).join('\n') : '• All services operational'}\n\nSystem is running smoothly!`,
            actions: [
              { type: 'system_metrics', label: '📊 View Detailed Metrics', data: {} },
              { type: 'agent_status', label: '🤖 Check Agent Status', data: {} }
            ]
          }
        }
        break
        
      case 'agent_management':
        const agentsResponse = await fetch(`${baseUrl}/api/agents`)
        if (agentsResponse.ok) {
          const agentsData = await agentsResponse.json()
          const agentCount = agentsData.length || 0
          return {
            content: `🤖 **Agent Management**\n\nActive Agents: ${agentCount}\n\nI can help you with:\n• Creating new agents\n• Managing agent skills\n• Monitoring agent performance\n• Assigning tasks to agents\n\nWhat would you like to do with your agents?`,
            actions: [
              { type: 'create_agent', label: '➕ Create New Agent', data: {} },
              { type: 'view_agents', label: '👀 View All Agents', data: {} },
              { type: 'agent_metrics', label: '📈 Agent Performance', data: {} }
            ]
          }
        }
        break
        
      case 'document_management':
        const documentsResponse = await fetch(`${baseUrl}/api/documents`)
        if (documentsResponse.ok) {
          const documentsData = await documentsResponse.json()
          const docCount = documentsData.length || 0
          return {
            content: `📄 **Document Management**\n\nTotal Documents: ${docCount}\n\nI can assist you with:\n• Document analysis and processing\n• Knowledge extraction\n• Search and retrieval\n• Document organization\n\nWhat document task can I help with?`,
            actions: [
              { type: 'upload_document', label: '📤 Upload Document', data: {} },
              { type: 'search_documents', label: '🔍 Search Documents', data: {} },
              { type: 'document_analytics', label: '📊 Document Analytics', data: {} }
            ]
          }
        }
        break
        
      case 'workflow_management':
        const workflowsResponse = await fetch(`${baseUrl}/api/workflows`)
        if (workflowsResponse.ok) {
          const workflowsData = await workflowsResponse.json()
          const workflowCount = workflowsData.length || 0
          return {
            content: `🔄 **Workflow Management**\n\nActive Workflows: ${workflowCount}\n\nI can help you:\n• Create automated workflows\n• Monitor workflow execution\n• Optimize process efficiency\n• Debug workflow issues\n\nWhat workflow assistance do you need?`,
            actions: [
              { type: 'create_workflow', label: '🆕 Create Workflow', data: {} },
              { type: 'workflow_status', label: '📋 Workflow Status', data: {} },
              { type: 'workflow_analytics', label: '📈 Workflow Analytics', data: {} }
            ]
          }
        }
        break
    }
  } catch (error) {
    console.error(`Error handling ${intent}:`, error)
  }
  
  return null
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  
  try {
    const body = await request.json()
    const { query, context, session_id } = body

    // Analyze the user's intent
    const { intent, confidence } = analyzeIntent(query, context)
    
    // Try to handle the intent with real backend data
    const intentResponse = await handleIntent(intent, query, context)
    
    if (intentResponse) {
      return NextResponse.json({
        success: true,
        message: {
          id: `msg-${Date.now()}`,
          content: intentResponse.content,
          timestamp: new Date().toISOString(),
          actions: intentResponse.actions || [],
          context: { intent, query, currentPage: context.currentPage },
          metadata: {
            intent,
            confidence,
            processing_time: (Date.now() - startTime) / 1000
          }
        }
      })
    }
    
    // Fallback for unhandled intents
    throw new Error('Intent not handled')
    
  } catch (error) {
    console.error('Chatbot query error:', error)
    
    // Return a helpful fallback response
    return NextResponse.json({
      success: false,
      message: {
        id: `fallback-${Date.now()}`,
        content: `I understand you're asking about that topic. I'm currently connecting to our AI orchestrator to provide you with more detailed assistance. 

In the meantime, here are some things I can help you with:
• 📊 System health and monitoring
• 🤖 Agent performance and management  
• 📄 Document processing and analysis
• 🔄 Workflow automation and tracking
• 📈 Analytics and reporting

Please try rephrasing your question or select one of the suggested actions below.`,
        timestamp: new Date().toISOString(),
        actions: [
          { type: 'system_health', label: '🟢 Check System Health', data: {} },
          { type: 'agent_status', label: '🤖 View Agent Status', data: {} },
          { type: 'workflow_status', label: '🔄 Workflow Status', data: {} },
          { type: 'help', label: '❓ Get Help', data: {} }
        ],
        context: {},
        metadata: {
          intent: 'fallback',
          confidence: 0.5,
          processing_time: 0.1
        }
      }
    }, { status: 200 })
  }
}
