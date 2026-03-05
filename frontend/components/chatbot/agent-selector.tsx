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

import { ToolLogo } from '@/components/ui/tool-logo'
import { Badge } from '@/components/ui/badge'

export interface Agent {
  id: number
  name: string
  agent_type: string
  description: string
  status: string
  skills?: Array<{ id: string; name: string }>
  agent_model_config?: {
    model_id?: string
    provider?: string
  }
  model_config?: any // Legacy fallback
  is_default: boolean
  tags: string[]
  tools?: Array<{
    id: number
    name: string
    icon?: string
  }>
  // PRD-67: System agent fields
  is_system_agent?: boolean
  slug?: string
  required_role?: string
}

export interface AgentSelectorProps {
  selectedAgentId?: number | null
  onAgentChange: (agentId: number | null) => void
  onAgentData?: (agent: Agent | null) => void
}

// Helper for model display name
const getModelDisplayName = (modelId?: string): string => {
  if (!modelId) return 'GPT-4'

  const modelNames: Record<string, string> = {
    'gpt-4': 'GPT-4',
    'gpt-4-turbo': 'GPT-4 Turbo',
    'gpt-4-turbo-preview': 'GPT-4 Turbo',
    'gpt-3.5-turbo': 'GPT-3.5',
    'claude-3-opus-20240229': 'Claude 3 Opus',
    'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
    'claude-3-haiku-20240307': 'Claude 3 Haiku',
  }

  return modelNames[modelId] || modelId.split('-')[0].toUpperCase()
}

export function AgentSelector({ selectedAgentId, onAgentChange, onAgentData }: AgentSelectorProps) {
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

  // PRD-67: Separate CTO/system agents from workspace agents
  const ctoAgent = agents.find(a => a.slug === 'auto-cto' && a.is_system_agent)
  const workspaceAgents = agents.filter(a => !a.is_system_agent)

  // Bubble up selected agent data
  useEffect(() => {
    if (onAgentData) {
      onAgentData(selectedAgent || null)
    }
  }, [selectedAgent, onAgentData])





  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          data-tour="chat-agent-selector"
          className="h-9 px-3 rounded-full border-2 border-orange-500/20 bg-black/20 hover:bg-orange-500/5 hover:border-orange-500/40 text-foreground/90 text-xs gap-2 shadow-[0_0_18px_rgba(249,115,22,0.10)]"
        >
          {isLoading ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <>
              {selectedAgent?.slug === 'auto-cto' ? (
                <>
                  <Bot className="w-3 h-3 text-emerald-400" />
                  <span className="truncate max-w-[120px] font-medium text-emerald-400">
                    Auto CTO
                  </span>
                </>
              ) : selectedAgent ? (
                <>
                  <Bot className="w-3 h-3 text-orange-400" />
                  <span className="truncate max-w-[120px] font-medium">
                    {selectedAgent.name}
                  </span>
                </>
              ) : (
                <>
                  <Bot className="w-3 h-3 text-orange-400" />
                  <span className="truncate max-w-[160px]">
                    Auto
                  </span>
                </>
              )}
            </>
          )}
          <ChevronDown className="w-3 h-3 text-orange-400 opacity-50" />
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
              <div className="font-medium text-sm">Auto</div>
              <div className="text-xs text-muted-foreground truncate">
                Router picks the best agent for each message
              </div>
            </div>
            {!selectedAgentId && (
              <Check className="w-4 h-4 text-orange-400 ml-2 flex-shrink-0" />
            )}
          </div>
        </DropdownMenuItem>
        {/* PRD-67: CTO Agent pinned at top for admins */}
        {ctoAgent && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuLabel className="text-[10px] text-muted-foreground/60 uppercase tracking-wider px-2">
              Platform Builder
            </DropdownMenuLabel>
            <DropdownMenuItem
              onClick={() => onAgentChange(ctoAgent.id)}
              className="text-foreground hover:bg-emerald-500/10 cursor-pointer rounded-lg py-2"
            >
              <div className="flex items-start justify-between w-full">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <div className="font-medium text-sm text-emerald-400">{ctoAgent.name}</div>
                    <span className="text-[10px] text-emerald-400/80 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                      CTO
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {ctoAgent.description}
                  </div>
                </div>
                {selectedAgentId === ctoAgent.id && (
                  <Check className="w-4 h-4 text-emerald-400 ml-2 flex-shrink-0" />
                )}
              </div>
            </DropdownMenuItem>
          </>
        )}

        <DropdownMenuSeparator />
        {workspaceAgents.length > 0 && (
          <DropdownMenuLabel className="text-[10px] text-muted-foreground/60 uppercase tracking-wider px-2">
            Workspace Agents
          </DropdownMenuLabel>
        )}

        {workspaceAgents.map((agent) => {
          const isSelected = selectedAgentId === agent.id

          return (
            <DropdownMenuItem
              key={agent.id}
              onClick={() => onAgentChange(agent.id)}
              className="text-foreground hover:bg-orange-500/10 cursor-pointer rounded-lg py-2"
            >
              <div className="flex items-start justify-between w-full">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <div className="font-medium text-sm">{agent.name}</div>
                    <span className="text-[10px] text-muted-foreground/70 bg-secondary/30 px-1.5 py-0.5 rounded border border-white/5">
                      {getModelDisplayName(agent.agent_model_config?.model_id || agent.model_config?.model_id)}
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {agent.description}
                  </div>

                  {agent.skills && agent.skills.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {agent.skills.slice(0, 3).map((skill: any) => (
                        <Badge key={skill.id} variant="secondary" className="text-[10px] px-1 h-5 font-normal bg-orange-500/5 text-orange-300 border-orange-500/10">
                          {skill.name ? skill.name.replace('_', ' ') : 'Unknown'}
                        </Badge>
                      ))}
                      {agent.skills.length > 3 && (
                        <Badge variant="secondary" className="text-[10px] px-1 h-5 font-normal bg-secondary/30 text-muted-foreground">
                          +{agent.skills.length - 3}
                        </Badge>
                      )}
                    </div>
                  )}

                  {/* Tools Preview */}
                  {agent.tools && agent.tools.length > 0 && (
                    <div className="flex items-center gap-1 mt-2 pt-2 border-t border-white/5">
                      {agent.tools.slice(0, 5).map((tool) => (
                        <ToolLogo
                          key={tool.id}
                          name={tool.name}
                          logo={tool.icon}
                          size={14}
                          showBackground={true}
                          className="bg-secondary/30 border border-white/5 opacity-70 hover:opacity-100 transition-opacity"
                        />
                      ))}
                      {agent.tools.length > 5 && (
                        <span className="text-[10px] text-muted-foreground ml-1">
                          +{agent.tools.length - 5}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                {isSelected && (
                  <Check className="w-4 h-4 text-orange-400 ml-2 flex-shrink-0" />
                )}
              </div>
            </DropdownMenuItem>
          )
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
