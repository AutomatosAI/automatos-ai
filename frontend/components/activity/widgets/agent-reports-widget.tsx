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
import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { useAgentReports } from '@/hooks/use-activity-api'
import type { AgentReport } from '@/hooks/use-activity-api'
import { apiClient } from '@/lib/api-client'
import { useQuery } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import { cn } from '@/lib/utils'

const PINNED_AGENTS_KEY = 'automatos:pinned-agents'

function loadPinnedAgents(): number[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(PINNED_AGENTS_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function savePinnedAgents(ids: number[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(PINNED_AGENTS_KEY, JSON.stringify(ids))
}

interface SimpleAgent {
  id: number
  name: string
  marketplace_icon?: string | null
}

function useWorkspaceAgents() {
  return useQuery<SimpleAgent[]>({
    queryKey: ['agents', 'simple-list'],
    queryFn: async () => {
      const res = await apiClient.request<{ agents: SimpleAgent[] }>('/api/agents')
      return res.agents ?? res as unknown as SimpleAgent[]
    },
    staleTime: 60000,
  })
}

const STATUS_CONFIG: Record<string, { icon: typeof CheckCircle2; color: string; label: string }> = {
  completed: { icon: CheckCircle2, color: 'text-[hsl(var(--success))]', label: 'Done' },
  failed: { icon: XCircle, color: 'text-destructive', label: 'Failed' },
  no_data: { icon: Clock, color: 'text-muted-foreground', label: 'No reports' },
  error: { icon: XCircle, color: 'text-destructive', label: 'Error' },
}

function ReportCard({ report }: { report: AgentReport }) {
  const router = useRouter()
  const statusConf = STATUS_CONFIG[report.status] || STATUS_CONFIG.no_data
  const StatusIcon = statusConf.icon
  const lastRunLabel = report.last_run
    ? formatDistanceToNow(new Date(report.last_run), { addSuffix: true })
    : 'Never'

  const handleViewReport = () => {
    if (report.agent_id) {
      router.push(`/agents?agent=${report.agent_id}`)
    }
  }

  return (
    <div className="flex flex-col p-3 rounded-lg bg-secondary/30 border border-border/30 hover:border-border/60 transition-colors min-w-[200px]">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
          {report.agent_icon ? (
            <span className="text-lg">{report.agent_icon}</span>
          ) : (
            <Bot className="w-4 h-4 text-primary" />
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
}

function PinSelector({ agents, pinnedIds, onToggle }: PinSelectorProps) {
  const pinnedSet = new Set(pinnedIds)

  return (
    <div className="space-y-1 max-h-60 overflow-y-auto">
      <p className="text-xs text-muted-foreground px-2 pb-1">
        Pin up to 4 agents
      </p>
      {agents.map((agent) => {
        const isPinned = pinnedSet.has(agent.id)
        const isDisabled = !isPinned && pinnedIds.length >= 4
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
            {isPinned ? (
              <Pin className="w-3 h-3 text-primary shrink-0" />
            ) : (
              <PinOff className="w-3 h-3 text-muted-foreground shrink-0" />
            )}
            <span className="truncate">{agent.name}</span>
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
  const [pinnedIds, setPinnedIds] = useState<number[]>([])

  // Load pinned IDs from localStorage
  useEffect(() => {
    setPinnedIds(loadPinnedAgents())
  }, [])

  const { data: allAgents } = useWorkspaceAgents()
  const { data: reportsData, isLoading } = useAgentReports(pinnedIds)
  const reports = reportsData?.reports ?? []

  // Auto-pin first 3 agents if nothing pinned
  useEffect(() => {
    if (pinnedIds.length === 0 && allAgents && allAgents.length > 0) {
      const autoPinned = allAgents.slice(0, 3).map((a) => a.id)
      setPinnedIds(autoPinned)
      savePinnedAgents(autoPinned)
    }
  }, [allAgents, pinnedIds.length])

  const togglePin = useCallback((id: number) => {
    setPinnedIds((prev) => {
      const next = prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 4
          ? [...prev, id]
          : prev
      savePinnedAgents(next)
      return next
    })
  }, [])

  return (
    <div className={cn('h-full flex flex-col', className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Agent Reports</h3>
          <span className="text-[10px] text-muted-foreground">
            {pinnedIds.length} of {allAgents?.length ?? 0}
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
            />
          </PopoverContent>
        </Popover>
      </div>

      <div className="flex-1 overflow-x-auto px-4 py-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : reports.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
            <Bot className="w-8 h-8 mb-2 opacity-30" />
            <p className="text-xs">Pin agents to see their reports</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {reports.map((report) => (
              <ReportCard key={report.agent_id} report={report} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
