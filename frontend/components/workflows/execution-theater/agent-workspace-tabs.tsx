'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Bot,
  Brain,
  Zap,
  Clock,
  Activity,
  MessageSquare,
  FileText,
  Cpu
} from 'lucide-react'

interface AgentWorkspaceTabsProps {
  workflow: any
  selectedAgent: number | null
  onAgentSelect: (agentId: number) => void
  isExecuting: boolean
}

export function AgentWorkspaceTabs({
  workflow,
  selectedAgent,
  onAgentSelect,
  isExecuting
}: AgentWorkspaceTabsProps) {
  // Extract real agents from workflow execution subtasks
  const execution = workflow?.execution
  const subtasks = execution?.output_data?.subtasks || []
  
  // Build agent list from subtasks (deduplicate by agent_id)
  const agentMap = new Map()
  subtasks.forEach((subtask: any, index: number) => {
    const agentId = subtask.selected_agent?.agent_id
    const agentName = subtask.selected_agent?.agent_name
    
    if (agentId && !agentMap.has(agentId)) {
      agentMap.set(agentId, {
        id: agentId,
        name: agentName,
        agent_type: subtask.agent_type || 'agent',
        status: execution?.status === 'running' && index === 0 ? 'executing' : 
                subtask.execution_result?.status || 'pending',
        currentTask: subtask.description,
        progress: subtask.execution_result?.status === 'completed' ? 100 : 
                 subtask.execution_result?.status === 'running' ? 50 : 0,
        tokensUsed: subtask.execution_result?.tokens_used || 0
      })
    }
  })
  
  const agents = Array.from(agentMap.values())
  const liveProgress = workflow?.liveProgress || {}
  
  // Select first agent by default
  useEffect(() => {
    if (!selectedAgent && agents.length > 0) {
      onAgentSelect(agents[0].id)
    }
  }, [agents, selectedAgent, onAgentSelect])

  const activeAgent = agents.find((a: any) => a.id === selectedAgent) || agents[0]
  const agentProgress = liveProgress.agents?.[selectedAgent || agents[0]?.id] || 
                       (activeAgent ? {
                         status: activeAgent.status,
                         current_task: activeAgent.currentTask,
                         progress: activeAgent.progress,
                         tokensUsed: activeAgent.tokensUsed
                       } : {})

  // Show waiting state when no agents yet
  if (!execution || agents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <Bot className="w-12 h-12 mb-3 opacity-50 animate-pulse" />
        <p className="text-sm font-medium">Waiting for Orchestrator...</p>
        <p className="text-xs mt-1">Agents will appear as tasks are assigned</p>
      </div>
    )
  }

  if (!activeAgent) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        No agent selected
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Agent Tabs */}
      <Tabs
        value={`agent-${selectedAgent}`}
        onValueChange={(value) => onAgentSelect(parseInt(value.replace('agent-', '')))}
        className="flex-1 flex flex-col"
      >
        <TabsList className="w-full justify-start border-b rounded-none bg-transparent">
          {agents.map((agent: any) => {
            const progress = liveProgress.agents?.[agent.id]
            return (
              <TabsTrigger
                key={agent.id}
                value={`agent-${agent.id}`}
                className="relative"
              >
                <Bot className="w-4 h-4 mr-2" />
                {agent.name}
                {progress?.status === 'executing' && (
                  <div className="absolute top-1 right-1 w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                )}
              </TabsTrigger>
            )
          })}
        </TabsList>

        {agents.map((agent: any) => {
          const agentProgress = liveProgress.agents?.[agent.id] || {
            status: agent.status,
            current_task: agent.currentTask,
            progress: agent.progress,
            tokensUsed: agent.tokensUsed
          }
          
          return (
            <TabsContent
              key={agent.id}
              value={`agent-${agent.id}`}
              className="flex-1 m-0 p-4 overflow-hidden"
            >
              <AgentWorkspace
                agent={agent}
                progress={agentProgress}
                isExecuting={isExecuting}
                workflow={workflow}
              />
            </TabsContent>
          )
        })}
      </Tabs>
    </div>
  )
}

function AgentWorkspace({
  agent,
  progress,
  isExecuting,
  workflow
}: {
  agent: any
  progress: any
  isExecuting: boolean
  workflow: any
}) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'executing': return 'text-blue-400'
      case 'completed': return 'text-green-400'
      case 'waiting': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <ScrollArea className="h-full pr-4">
      <div className="space-y-4">
        {/* Current Task - Compact */}
        {progress?.current_task && (
          <Card className="glass-card">
            <CardContent className="pt-4">
              <p className="text-sm text-primary font-medium mb-3">
                {progress.current_task}
              </p>
              
              {progress.progress !== undefined && (
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Progress</span>
                    <span className="font-medium">{Math.round(progress.progress)}%</span>
                  </div>
                  <Progress value={progress.progress} className="h-2" />
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* LLM Call Status */}
        {isExecuting && progress?.status === 'executing' && (
          <Card className="glass-card border-blue-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center text-blue-400">
                <Brain className="w-4 h-4 mr-2 animate-pulse" />
                LLM Call in Progress
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Model:</span>
                <span className="font-medium">GPT-4</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Tokens:</span>
                <span className="font-medium">
                  {progress.tokens_used || 0} / 8,000
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Elapsed:</span>
                <span className="font-medium">{progress.elapsed_time || '0'}s</span>
              </div>
              <div className="h-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-pulse" />
            </CardContent>
          </Card>
        )}

        {/* Tools Used */}
        {progress?.tools_used && progress.tools_used.length > 0 && (
          <Card className="glass-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center">
                <Zap className="w-4 h-4 mr-2" />
                Tools Used
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {progress.tools_used.map((tool: any, index: number) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span>{tool.name}</span>
                  <Badge variant="outline" className="text-xs">
                    {tool.status || 'completed'}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Memory Operations */}
        {progress?.memory_operations && progress.memory_operations.length > 0 && (
          <Card className="glass-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                Memory Operations
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {progress.memory_operations.map((op: any, index: number) => (
                <div key={index} className="text-sm space-y-1">
                  <div className="flex items-center justify-between">
                    <span className={getStatusColor(op.status)}>
                      {op.type === 'store' ? '✓' : '→'} {op.type.toUpperCase()}
                    </span>
                    <span className="text-xs text-muted-foreground">{op.key}</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {/* Context Engineering */}
        {progress?.prompt_construction && (
          <Card className="glass-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center">
                <FileText className="w-4 h-4 mr-2" />
                Context Engineering
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Atomic:</span>
                  <span>{progress.prompt_construction.atomic_tokens || 0} tokens</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Molecular:</span>
                  <span>{progress.prompt_construction.molecular_tokens || 0} tokens</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Cellular:</span>
                  <span>{progress.prompt_construction.cellular_tokens || 0} tokens</span>
                </div>
                <div className="border-t border-border/50 pt-2 flex justify-between font-medium">
                  <span>Total:</span>
                  <span className="text-primary">
                    {progress.prompt_construction.total_tokens || 0} tokens
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Info Density:</span>
                  <span className="text-green-400">
                    {progress.prompt_construction.info_density || 0.85}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Live Logs */}
        <Card className="glass-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center">
              <MessageSquare className="w-4 h-4 mr-2" />
              Live Logs
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[200px]">
              <div className="space-y-1 font-mono text-xs">
                {(() => {
                  // Generate real logs from subtask execution data
                  const execution = workflow?.execution
                  const subtasks = execution?.output_data?.subtasks || []
                  
                  // Find subtasks assigned to this agent
                  const agentSubtasks = subtasks.filter((st: any) => 
                    st.selected_agent?.agent_id === agent.id ||
                    st.selected_agent?.agent_name === agent.name
                  )
                  
                  if (agentSubtasks.length === 0) {
                    return <div className="text-muted-foreground">No logs available yet</div>
                  }
                  
                  const logs: any[] = []
                  
                  agentSubtasks.forEach((subtask: any, idx: number) => {
                    // Task received log
                    logs.push({
                      timestamp: new Date(execution.started_at).toLocaleTimeString(),
                      message: `📥 Task received: ${subtask.description.substring(0, 60)}...`,
                      type: 'info'
                    })
                    
                    // Execution start
                    if (subtask.execution_result) {
                      logs.push({
                        timestamp: new Date(execution.started_at).toLocaleTimeString(),
                        message: `▶️  Executing with ${subtask.selected_agent.llm_config?.model || 'default model'}`,
                        type: 'info'
                      })
                      
                      // Show tokens if available
                      if (subtask.execution_result.tokens_used > 0) {
                        logs.push({
                          timestamp: new Date(execution.started_at).toLocaleTimeString(),
                          message: `🔢 Processed ${subtask.execution_result.tokens_used} tokens`,
                          type: 'success'
                        })
                      }
                      
                      // Show result
                      if (subtask.execution_result.status === 'completed') {
                        const response = subtask.execution_result.llm_response || subtask.execution_result.response || ''
                        const preview = response.substring(0, 80)
                        logs.push({
                          timestamp: new Date(execution.started_at).toLocaleTimeString(),
                          message: `✅ Completed: ${preview}${response.length > 80 ? '...' : ''}`,
                          type: 'success'
                        })
                      } else if (subtask.execution_result.status === 'failed') {
                        logs.push({
                          timestamp: new Date(execution.started_at).toLocaleTimeString(),
                          message: `❌ Failed: ${subtask.execution_result.error_message || 'Unknown error'}`,
                          type: 'error'
                        })
                      } else if (subtask.execution_result.status === 'running') {
                        logs.push({
                          timestamp: new Date().toLocaleTimeString(),
                          message: `⏳ In progress...`,
                          type: 'info'
                        })
                      }
                    }
                  })
                  
                  return logs.map((log, index) => (
                    <div key={index} className={`${
                      log.type === 'error' ? 'text-red-400' : 
                      log.type === 'success' ? 'text-green-400' : 
                      'text-muted-foreground'
                    }`}>
                      <span className="text-xs opacity-50">{log.timestamp}</span>{' '}
                      <span>{log.message}</span>
                    </div>
                  ))
                })()}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  )
}

