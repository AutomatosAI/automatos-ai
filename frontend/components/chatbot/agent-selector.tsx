'use client'

import { Check, ChevronDown, Loader2, Bot } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu'
import { useEffect, useState } from 'react'
import { apiClient } from '@/lib/api-client'

import { useWorkspace } from '@/components/workspace-provider'

export interface Agent {
  id: number
  name: string
  agent_type: string
  description: string
  status: string
  skills: string[]
  model_config: any
  is_default: boolean
  tags: string[]
}

export interface AgentSelectorProps {
  selectedAgentId?: number | null
  onAgentChange: (agentId: number | null) => void
}

export function AgentSelector({ selectedAgentId, onAgentChange }: AgentSelectorProps) {
  const [agents, setAgents] = useState<Agent[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const { workspace, isLoading: isWorkspaceLoading } = useWorkspace()

  useEffect(() => {
    if (isWorkspaceLoading) return

    // Even if workspace is null (e.g. auth failed), we try to fetch active agents
    // The apiClient will pick up the header if it exists in localStorage (set by WorkspaceProvider)

    apiClient.request('/api/agents/?status=active')
      .then((data: any) => {
        const agentList = Array.isArray(data) ? data : (data.agents || [])
        setAgents(agentList)
        setIsLoading(false)
      })
      .catch((error) => {
        console.error('Failed to load agents:', error)
        setIsLoading(false)
      })
  }, [isWorkspaceLoading, workspace])

  const selectedAgent = agents.find(a => a.id === selectedAgentId)

  const agentsByType = agents.reduce((acc: Record<string, Agent[]>, agent) => {
    const type = agent.agent_type || 'custom'
    if (!acc[type]) acc[type] = []
    acc[type].push(agent)
    return acc
  }, {})

  const typeLabels: Record<string, string> = {
    system: '⚙️ System',
    code_architect: '💻 Code Experts',
    security_expert: '🔒 Security',
    technical_writer: '📝 Writers',
    data_analyst: '📊 Data',
    operations: '🚀 DevOps',
    custom: '🤖 Custom',
    analysis: '🔍 Analysis',
    specialized: '⭐ Specialized',
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 px-3 rounded-full border-2 border-orange-500/20 bg-black/20 hover:bg-orange-500/5 hover:border-orange-500/40 text-foreground/90 text-xs gap-2 shadow-[0_0_18px_rgba(249,115,22,0.10)]"
        >
          {isLoading ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <>
              <Bot className="w-3 h-3 text-orange-400" />
              <span className="truncate max-w-[160px]">
                {selectedAgent?.name || 'Auto (Model)'}
              </span>
            </>
          )}
          <ChevronDown className="w-3 h-3 text-orange-400" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="start"
        className="w-[380px] max-h-[500px] overflow-y-auto bg-background border-orange-500/20 rounded-2xl"
        style={{ maxHeight: '70vh' }}
      >
        <DropdownMenuItem
          onClick={() => onAgentChange(null)}
          className="text-foreground hover:bg-orange-500/10 cursor-pointer rounded-lg py-2"
        >
          <div className="flex items-start justify-between w-full">
            <div className="flex-1 min-w-0">
              <div className="font-medium text-sm">�� Auto (Model Selection)</div>
              <div className="text-xs text-muted-foreground truncate">
                Use model selector (legacy mode)
              </div>
            </div>
            {!selectedAgentId && (
              <Check className="w-4 h-4 text-orange-400 ml-2 flex-shrink-0" />
            )}
          </div>
        </DropdownMenuItem>
        <DropdownMenuSeparator />

        {Object.entries(agentsByType).map(([type, typeAgents]) => (
          <div key={type}>
            <DropdownMenuLabel className="text-xs text-muted-foreground sticky top-0 bg-background z-10">
              {typeLabels[type] || type}
            </DropdownMenuLabel>
            {typeAgents.map((agent) => {
              const isSelected = selectedAgentId === agent.id
              const modelName = agent.model_config?.model_id || 'Unknown'

              return (
                <DropdownMenuItem
                  key={agent.id}
                  onClick={() => onAgentChange(agent.id)}
                  className="text-foreground hover:bg-orange-500/10 cursor-pointer rounded-lg py-2"
                >
                  <div className="flex items-start justify-between w-full">
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm">{agent.name}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {agent.description}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-muted-foreground/70">
                          {modelName}
                        </span>
                        {agent.skills.length > 0 && (
                          <span className="text-xs text-orange-400/70">
                            • {agent.skills.length} skill{agent.skills.length !== 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    </div>
                    {isSelected && (
                      <Check className="w-4 h-4 text-orange-400 ml-2 flex-shrink-0" />
                    )}
                  </div>
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
          </div>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
