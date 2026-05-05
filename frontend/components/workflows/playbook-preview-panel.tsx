'use client'

import * as React from 'react'
import { useFormContext } from 'react-hook-form'
import {
  Bot,
  Zap,
  ArrowDown,
  GitBranch,
  Clock,
  Play,
  Webhook,
  Wrench,
  BarChart3,
} from 'lucide-react'
import { useAgents } from '@/hooks/use-agent-api'
import type { PlaybookFormValues } from './create-playbook-modal'

interface Agent {
  id: number
  name: string
  agent_model_config?: {
    model_id?: string
    provider?: string
  }
  model_config?: any
  skills?: Array<{ id: string; name: string }>
  status: string
}

const SCHEDULE_LABELS: Record<string, { label: string; icon: typeof Play }> = {
  manual: { label: 'Manual', icon: Play },
  cron: { label: 'Scheduled', icon: Clock },
  trigger: { label: 'Triggered', icon: Webhook },
}

function calculateComplexity(
  steps: PlaybookFormValues['steps'],
  config: PlaybookFormValues['execution_config'],
  schedule: PlaybookFormValues['schedule_config']
): { score: number; label: string; color: string } {
  let score = 0

  // Step count contribution (0-40)
  score += Math.min(steps.length * 8, 40)

  // Parallel mode adds complexity
  if (config.mode === 'parallel') score += 10

  // High retry count adds complexity
  if (config.max_retries > 5) score += 10
  else if (config.max_retries > 2) score += 5

  // Scheduled/triggered adds complexity vs manual
  if (schedule.type === 'cron') score += 5
  if (schedule.type === 'trigger') score += 10

  // Steps with custom pass_to (non-linear flow)
  const customRouting = steps.filter((s) => s.pass_to).length
  if (customRouting > 0) score += customRouting * 5

  score = Math.min(score, 100)

  if (score <= 25) return { score, label: 'Simple', color: 'text-[hsl(var(--success))]' }
  if (score <= 50) return { score, label: 'Moderate', color: 'text-[hsl(var(--warning))]' }
  if (score <= 75) return { score, label: 'Complex', color: 'text-primary' }
  return { score, label: 'Advanced', color: 'text-[hsl(var(--destructive))]' }
}

export function PlaybookPreviewPanel() {
  const methods = useFormContext<PlaybookFormValues>()
  const name = methods.watch('name')
  const description = methods.watch('description')
  const steps = methods.watch('steps')
  const executionConfig = methods.watch('execution_config')
  const scheduleConfig = methods.watch('schedule_config')

  const { data: agentsData } = useAgents()

  const agents: Agent[] = React.useMemo(() => {
    if (!agentsData) return []
    const list = Array.isArray(agentsData) ? agentsData : (agentsData as any).agents || []
    return list.filter((a: Agent) => a.status === 'active')
  }, [agentsData])

  // Map agent IDs to agent objects for the selected steps
  const stepAgents = React.useMemo(() => {
    const map = new Map<string, Agent>()
    for (const step of steps) {
      if (step.agent_id) {
        const agentIdStr = String(step.agent_id)
        const agent = agents.find((a) => String(a.id) === agentIdStr)
        if (agent) map.set(agentIdStr, agent)
      }
    }
    return map
  }, [steps, agents])

  // Aggregate tools from all selected agents
  const allTools = React.useMemo(() => {
    const toolSet = new Map<string, string>()
    for (const agent of stepAgents.values()) {
      if (agent.skills) {
        for (const skill of agent.skills) {
          toolSet.set(skill.id, skill.name)
        }
      }
    }
    return Array.from(toolSet.values())
  }, [stepAgents])

  const complexity = React.useMemo(
    () => calculateComplexity(steps, executionConfig, scheduleConfig),
    [steps, executionConfig, scheduleConfig]
  )

  const scheduleInfo = SCHEDULE_LABELS[scheduleConfig.type] || SCHEDULE_LABELS.manual
  const ScheduleIcon = scheduleInfo.icon

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Preview
      </h3>

      {/* Recipe Name & Description */}
      <div className="glass-card rounded-2xl p-4 border border-border/20 space-y-2">
        <div className="text-sm font-medium text-foreground truncate">
          {name || <span className="text-muted-foreground/50 italic">Untitled Playbook</span>}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground line-clamp-3">{description}</p>
        )}
      </div>

      {/* Badges Row */}
      <div className="flex flex-wrap gap-2">
        {/* Execution Mode Badge */}
        <div
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
            executionConfig.mode === 'parallel'
              ? 'border-[hsl(var(--agent))]/40 bg-[hsl(var(--agent))]/10 text-[hsl(var(--agent))]'
              : 'border-[hsl(var(--info))]/40 bg-[hsl(var(--info))]/10 text-[hsl(var(--info))]'
          }`}
        >
          {executionConfig.mode === 'parallel' ? (
            <GitBranch className="w-3 h-3" />
          ) : (
            <ArrowDown className="w-3 h-3" />
          )}
          {executionConfig.mode === 'parallel' ? 'Parallel' : 'Sequential'}
        </div>

        {/* Schedule Type Badge */}
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border border-primary/40 bg-primary/10 text-primary">
          <ScheduleIcon className="w-3 h-3" />
          {scheduleInfo.label}
        </div>

        {/* Step Count Badge */}
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border border-border/40 bg-secondary/50 text-muted-foreground">
          {steps.length} step{steps.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Visual Flow Diagram */}
      {steps.length > 0 && (
        <div className="glass-card rounded-2xl p-4 border border-border/20 space-y-1">
          <div className="text-xs font-medium text-muted-foreground mb-3">Workflow Flow</div>
          <div className="space-y-0">
            {steps.map((step, index) => {
              const agent = step.agent_id ? stepAgents.get(String(step.agent_id)) : null
              const isLast = index === steps.length - 1
              const hasCustomRoute = Boolean(step.pass_to)

              return (
                <div key={step.step_id} className="relative">
                  {/* Step Box */}
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 flex items-center justify-center rounded bg-primary/20 text-primary text-[10px] font-bold flex-shrink-0">
                      {step.order}
                    </div>
                    <div className="flex-1 min-w-0 px-2.5 py-1.5 rounded-lg bg-secondary/40 border border-border/20 flex items-center gap-2">
                      {agent ? (
                        <>
                          <Bot className="w-3 h-3 text-primary flex-shrink-0" />
                          <span className="text-xs text-foreground/80 truncate">
                            {agent.name}
                          </span>
                        </>
                      ) : (
                        <span className="text-xs text-muted-foreground/50 italic">
                          No agent
                        </span>
                      )}
                    </div>
                    {hasCustomRoute && (
                      <div className="text-[10px] text-[hsl(var(--agent))] flex-shrink-0 flex items-center gap-0.5">
                        <GitBranch className="w-2.5 h-2.5" />
                        {(() => {
                          const target = steps.find((s) => s.step_id === step.pass_to)
                          return target ? `→${target.order}` : ''
                        })()}
                      </div>
                    )}
                  </div>

                  {/* Arrow between steps */}
                  {!isLast && (
                    <div className="flex justify-center py-0.5">
                      <ArrowDown className="w-3 h-3 text-muted-foreground/30" />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Selected Agents */}
      {stepAgents.size > 0 && (
        <div className="glass-card rounded-2xl p-4 border border-border/20 space-y-2">
          <div className="text-xs font-medium text-muted-foreground">
            Agents ({stepAgents.size})
          </div>
          <div className="space-y-2">
            {Array.from(stepAgents.values()).map((agent) => (
              <div key={agent.id} className="flex items-center gap-2">
                <Bot className="w-3 h-3 text-primary flex-shrink-0" />
                <span className="text-xs text-foreground/80 truncate">{agent.name}</span>
                <span className="text-[10px] text-muted-foreground/50 ml-auto flex-shrink-0">
                  {agent.skills?.length || 0} tools
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Required Tools */}
      {allTools.length > 0 && (
        <div className="glass-card rounded-2xl p-4 border border-border/20 space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Wrench className="w-3 h-3" />
            Tools ({allTools.length})
          </div>
          <div className="flex flex-wrap gap-1.5">
            {allTools.slice(0, 12).map((tool) => (
              <span
                key={tool}
                className="px-2 py-0.5 rounded-md bg-secondary/50 border border-border/20 text-[10px] text-muted-foreground"
              >
                {tool}
              </span>
            ))}
            {allTools.length > 12 && (
              <span className="px-2 py-0.5 rounded-md bg-secondary/50 border border-border/20 text-[10px] text-muted-foreground/50">
                +{allTools.length - 12} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Complexity Score */}
      <div className="glass-card rounded-2xl p-4 border border-border/20 space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          <BarChart3 className="w-3 h-3" />
          Complexity
        </div>
        <div className="flex items-center gap-3">
          <div className="flex-1 h-2 rounded-full bg-secondary/50 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${complexity.score}%`,
                background: `linear-gradient(to right, hsl(var(--success)), hsl(var(--primary)), hsl(var(--destructive)))`,
                backgroundSize: '300% 100%',
                backgroundPosition: `${complexity.score}% 0`,
              }}
            />
          </div>
          <span className={`text-xs font-medium ${complexity.color}`}>
            {complexity.label}
          </span>
        </div>
        <div className="text-[10px] text-muted-foreground/50">
          Score: {complexity.score}/100
        </div>
      </div>
    </div>
  )
}
