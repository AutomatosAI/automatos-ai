'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import {
  Bot,
  Pin,
  PinOff,
  Settings2,
  CheckCircle2,
  XCircle,
  Clock,
  ExternalLink,
  Loader2,
} from 'lucide-react'
import { PremiumIcon } from '@/components/shared'
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { useAgentReports } from '@/hooks/use-activity-api'
import type { AgentReport } from '@/hooks/use-activity-api'
import { useAgents } from '@/hooks/use-agent-api'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { formatDistanceToNow } from 'date-fns'
import { cn } from '@/lib/utils'

const MAX_PINNED = 12

/** F157: pins are kept per workspace. An agent id belongs to one workspace, so
 *  a single browser-wide list carried pins that name no agent here. */
export function pinnedAgentsKey(workspaceId: string): string {
  return `automatos:pinned-agents:${workspaceId}`
}

function loadPinnedAgents(workspaceId: string): number[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(pinnedAgentsKey(workspaceId))
    const ids: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(ids) ? ids.filter((id): id is number => Number.isInteger(id)) : []
  } catch {
    return []
  }
}

function savePinnedAgents(workspaceId: string, ids: number[]) {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(pinnedAgentsKey(workspaceId), JSON.stringify(ids))
  } catch {
    // Storage full or blocked: the pins last for this visit only.
  }
}

/** F157: the pins that are still this workspace's agents, in pin order. */
export function livePins(pinnedIds: number[], agentIds: number[]): number[] {
  const here = new Set(agentIds)
  return pinnedIds.filter((id) => here.has(id))
}

interface SimpleAgent {
  id: number
  name: string
  premium_icon?: string | null
}

const STATUS_CONFIG: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
  completed: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]', label: 'Done' },
  failed: { icon: XCircle, color: 'text-destructive', label: 'Failed' },
  no_data: { icon: Clock, color: 'text-muted-foreground', label: 'No reports' },
  error: { icon: XCircle, color: 'text-destructive', label: 'Error' },
}

function ReportCard({ report, premiumIconName }: { report: AgentReport; premiumIconName?: string | null }) {
  const router = useRouter()
  const statusConf = STATUS_CONFIG[report.status] || STATUS_CONFIG.no_data
  const StatusIcon = statusConf.icon
  const lastRunLabel = report.last_run
    ? formatDistanceToNow(new Date(report.last_run), { addSuffix: true })
    : 'Never'

  const handleViewReport = () => {
    if (report.latest_file_path) {
      router.push(`/deliverables/explorer?path=${encodeURIComponent(report.latest_file_path)}`)
    } else if (report.agent_id) {
      router.replace(`/command-center?tab=board&agent_id=${report.agent_id}`, { scroll: false })
    }
  }

  return (
    <div className="flex flex-col p-3 rounded-lg bg-secondary/30 border border-border/30 hover:border-border/60 transition-colors min-w-[200px]">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-8 h-8 flex items-center justify-center shrink-0">
          {premiumIconName ? (
            <PremiumIcon name={premiumIconName} size={28} />
          ) : (
            <Bot className="w-5 h-5 text-primary" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium truncate">{report.agent_name}</p>
          <div className="flex items-center gap-1">
            <StatusIcon className={cn('w-3 h-3', statusConf.color)} />
            <span className="text-[10px] text-muted-foreground">{lastRunLabel}</span>
          </div>
        </div>
      </div>

      {report.summary ? (
        <p className="text-xs text-muted-foreground line-clamp-3 flex-1 mb-2">
          &ldquo;{report.summary}&rdquo;
        </p>
      ) : (
        <p className="text-xs text-muted-foreground/50 italic flex-1 mb-2">
          No reports yet
        </p>
      )}

      <Button
        variant="ghost"
        size="sm"
        className="text-xs h-7 w-full justify-center"
        onClick={handleViewReport}
      >
        <ExternalLink className="w-3 h-3 mr-1" />
        View Report
      </Button>
    </div>
  )
}

interface PinSelectorProps {
  agents: SimpleAgent[]
  pinnedIds: number[]
  onToggle: (id: number) => void
  isLoading?: boolean
}

function PinSelector({ agents, pinnedIds, onToggle, isLoading }: PinSelectorProps) {
  const pinnedSet = new Set(pinnedIds)

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (agents.length === 0) {
    return (
      <div className="py-4 text-center text-xs text-muted-foreground">
        No agents found. Create agents first.
      </div>
    )
  }

  return (
    <div className="space-y-1 max-h-60 overflow-y-auto">
      <p className="text-xs text-muted-foreground px-2 pb-1">
        Pin up to {MAX_PINNED} agents ({pinnedIds.length}/{MAX_PINNED})
      </p>
      {agents.map((agent) => {
        const isPinned = pinnedSet.has(agent.id)
        const isDisabled = !isPinned && pinnedIds.length >= MAX_PINNED
        return (
          <button
            key={agent.id}
            onClick={() => !isDisabled && onToggle(agent.id)}
            disabled={isDisabled}
            className={cn(
              'flex items-center gap-2 w-full px-2 py-1.5 rounded text-xs transition-colors',
              isPinned ? 'bg-primary/10' : 'hover:bg-secondary/60',
              isDisabled && 'opacity-40 cursor-not-allowed'
            )}
          >
            {agent.premium_icon ? (
              <PremiumIcon name={agent.premium_icon} size={14} className="shrink-0" />
            ) : isPinned ? (
              <Pin className="w-3 h-3 text-primary shrink-0" />
            ) : (
              <PinOff className="w-3 h-3 text-muted-foreground shrink-0" />
            )}
            <span className="truncate">{agent.name}</span>
            {isPinned && (
              <span className="ml-auto text-[10px] text-primary">pinned</span>
            )}
          </button>
        )
      })}
    </div>
  )
}

interface AgentReportsWidgetProps {
  className?: string
}

export function AgentReportsWidget({ className }: AgentReportsWidgetProps) {
  const workspaceId = useWorkspaceOptional()?.workspace?.id?.toString() ?? ''
  const [pinnedIds, setPinnedIds] = useState<number[]>([])

  // Load this workspace's pins from localStorage
  useEffect(() => {
    setPinnedIds(workspaceId ? loadPinnedAgents(workspaceId) : [])
  }, [workspaceId])

  const { data: rawAgents, isLoading: agentsLoading } = useAgents()
  // Map to SimpleAgent shape — useAgents returns full AgentResponse objects
  const allAgents: SimpleAgent[] | undefined = rawAgents
    ? (rawAgents as any[]).map((a: any) => ({ id: a.id, name: a.name, premium_icon: a.premium_icon ?? null }))
    : undefined

  const hasPinned = pinnedIds.length > 0
  // Pinning is a FILTER, not a precondition. With nothing pinned the endpoint
  // returns the agents that reported most recently, and the panel shows them.
  const { data: reportsData, isLoading: reportsLoading } = useAgentReports(pinnedIds)
  const reports = reportsData?.reports ?? []
  // A pinned agent that never reported still comes back, as status no_data.
  const hasReports = reports.some((report) => report.status !== 'no_data')

  // F157: every pinned agent of this workspace comes back with an entry; a pin
  // with none names no agent here any more (deleted, or recreated with a new id)
  // and is dropped. It used to stay, and the panel showed nothing at all.
  useEffect(() => {
    if (!workspaceId || pinnedIds.length === 0 || !reportsData) return
    const live = livePins(pinnedIds, reportsData.reports.map((report) => report.agent_id))
    if (live.length === pinnedIds.length) return
    setPinnedIds(live)
    savePinnedAgents(workspaceId, live)
  }, [workspaceId, pinnedIds, reportsData])

  const togglePin = useCallback((id: number) => {
    setPinnedIds((prev) => {
      const next = prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < MAX_PINNED
          ? [...prev, id]
          : prev
      if (workspaceId) savePinnedAgents(workspaceId, next)
      return next
    })
  }, [workspaceId])

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Agent Reports</h3>
          {/* Say what the number IS. "2 of 12" was pinned agents over all
              agents, and read as "2 of my 12 reports". */}
          <span className="text-[10px] text-muted-foreground">
            {hasPinned
              ? `${reports.length} pinned of ${allAgents?.length ?? 0} agents`
              : `${reports.length} most recent`}
          </span>
        </div>
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7">
              <Settings2 className="w-3.5 h-3.5" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-56 p-2" align="end">
            <PinSelector
              agents={allAgents ?? []}
              pinnedIds={pinnedIds}
              onToggle={togglePin}
              isLoading={agentsLoading}
            />
          </PopoverContent>
        </Popover>
      </div>

      <div className="flex-1 overflow-x-auto px-4 py-3">
        {agentsLoading || reportsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : hasPinned && !hasReports ? (
          /* F157: pinned agents that have never reported — say so, not a row of empty cards */
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <Bot className="w-8 h-8 mb-3 opacity-30" />
            <p className="text-sm font-medium mb-1">Your pinned agents have no reports</p>
            <p className="text-xs text-center max-w-[260px]">
              Change the pins with the settings button, or unpin them all to see the
              workspace&apos;s most recent reports.
            </p>
          </div>
        ) : !hasReports ? (
          /* Nothing pinned and nobody has reported yet — offer the pins */
          <div className="flex flex-col items-center justify-center py-6 text-muted-foreground">
            <Bot className="w-8 h-8 mb-3 opacity-30" />
            <p className="text-sm font-medium mb-1">No agent reports yet</p>
            <p className="text-xs mb-4 text-center max-w-[250px]">
              Agents&apos; routine reports appear here. Pin agents to follow them.
            </p>
            {allAgents && allAgents.length > 0 ? (
              <div className="w-full max-w-[280px] border border-border/50 rounded-lg p-2 bg-secondary/20">
                <PinSelector
                  agents={allAgents}
                  pinnedIds={pinnedIds}
                  onToggle={togglePin}
                />
              </div>
            ) : (
              <p className="text-xs text-muted-foreground/60">
                Create agents first to see reports
              </p>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {reports.map((report) => {
              const agent = allAgents?.find(a => a.id === report.agent_id)
              return (
                <ReportCard key={report.agent_id} report={report} premiumIconName={agent?.premium_icon} />
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
