'use client'

/**
 * PRD-228 US-004 — the fleet view.
 *
 * A live "what is everyone doing right now" list inside the Agents surface.
 * Each row shows the agent, its live line (working: <task> / idle / blocked:
 * awaiting answer), queue depth, rolling-24h cost, and a watch badge. Clicking
 * a row opens the EXISTING agent details modal via the shared onViewDetails
 * handler — no new modal. Data comes from the useFleetState hook (10s poll +
 * board-event refetch); this component owns none of that plumbing.
 */
import { AlertTriangle, Eye, Loader2 } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { useFleetState } from '@/hooks/use-agent-api'
import type { FleetHost, FleetAgentRow } from '@/lib/api-client'

interface FleetTabProps {
  /** Opens the shared AgentDetailsModal (owned by AgentManagement). */
  onViewDetails: (agentId: string | null) => void
}

type LiveTone = 'working' | 'idle' | 'blocked'

function liveLine(agent: FleetAgentRow): { text: string; tone: LiveTone } {
  // Blocked > working > idle. An agent is blocked either because it has an open
  // question ask, or because one of its board tasks is flagged blocked with no
  // outstanding question (approval-pending / manually blocked) — the latter uses
  // the already-emitted blocked.count so it never reads as 'idle' (P228-RVW-5).
  if (agent.blocked.open_asks.length > 0) {
    return { text: 'blocked: awaiting answer', tone: 'blocked' }
  }
  if (agent.blocked.count > 0) {
    return { text: 'blocked', tone: 'blocked' }
  }
  if (agent.current) {
    const s = agent.session
    if (s) {
      // PRD-234 S3: the user's own Claude Code session — say what it is doing.
      const doing = s.live_tool || s.last_event || 'running'
      return { text: `working: ${agent.current.title} · Claude Code (${s.model || 'claude'}) · ${doing}`, tone: 'working' }
    }
    return { text: `working: ${agent.current.title}`, tone: 'working' }
  }
  return { text: 'idle', tone: 'idle' }
}

function HostCard({ hosts }: { hosts: FleetHost[] }) {
  // PRD-234 S3: the CLI host explains every idle Claude Code agent below it.
  if (hosts.length === 0) {
    return (
      <div data-testid="fleet-host-card" className="rounded-xl border border-border/60 bg-card/40 px-4 py-3 text-sm">
        <span className="font-semibold">CLI host</span>
        <span className="text-muted-foreground"> · none paired — Claude Code agents cannot run. Pair one in Settings → Session mode.</span>
      </div>
    )
  }
  return (
    <div data-testid="fleet-host-card" className="space-y-2">
      {hosts.map((h) => (
        <div key={h.id} className="flex items-center gap-3 rounded-xl border border-border/60 bg-card/40 px-4 py-3 text-sm">
          <span
            aria-hidden
            className={'inline-block h-2 w-2 rounded-full ' + (h.online ? 'bg-[hsl(var(--success))]' : 'bg-[hsl(var(--warning))]')}
          />
          <span className="font-semibold">CLI host {h.name}</span>
          <span className="text-muted-foreground">
            {h.online ? 'online' : 'offline — start it with make cli-host'}
            {h.capabilities?.claude?.version ? ` · Claude Code ${h.capabilities.claude.version}` : ''}
          </span>
        </div>
      ))}
    </div>
  )
}

const TONE_CLASS: Record<LiveTone, string> = {
  working: 'text-[hsl(var(--success))]',
  idle: 'text-muted-foreground',
  blocked: 'text-[hsl(var(--destructive))]',
}

function formatCost(agent: FleetAgentRow): string {
  if (!agent.cost_24h) return 'cost n/a'
  const { tokens, usd } = agent.cost_24h
  return `last 24h · ${tokens.toLocaleString()} tok · $${usd.toFixed(2)}`
}

function FleetRow({
  agent,
  onViewDetails,
}: {
  agent: FleetAgentRow
  onViewDetails: (agentId: string | null) => void
}) {
  const line = liveLine(agent)
  const watches = agent.watches
  return (
    <button
      type="button"
      data-testid="fleet-row"
      onClick={() => onViewDetails(String(agent.agent_id))}
      className="flex w-full items-center gap-4 rounded-xl border border-border/60 bg-card/40 px-4 py-3 text-left transition-colors hover:bg-card/70"
    >
      <div className="min-w-0 flex-1">
        <div className="truncate font-semibold text-sm">{agent.name}</div>
        <div className={`truncate text-sm ${TONE_CLASS[line.tone]}`}>{line.text}</div>
      </div>

      {agent.queue_depth > 0 && (
        <Badge variant="outline" className="shrink-0">
          queue {agent.queue_depth}
        </Badge>
      )}

      {watches.active > 0 && (
        <Badge
          variant="outline"
          className={`shrink-0 gap-1 ${
            watches.needs_attention > 0
              ? 'border-[hsl(var(--destructive))]/40 text-[hsl(var(--destructive))]'
              : ''
          }`}
        >
          {watches.needs_attention > 0 ? (
            <AlertTriangle className="h-3 w-3" />
          ) : (
            <Eye className="h-3 w-3" />
          )}
          {watches.active}
        </Badge>
      )}

      <div className="shrink-0 text-right text-xs text-muted-foreground tabular-nums">
        {formatCost(agent)}
      </div>
    </button>
  )
}

export function FleetTab({ onViewDetails }: FleetTabProps) {
  const { data, isLoading, isError, refetch } = useFleetState()

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 px-1 py-8 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading fleet…
      </div>
    )
  }

  if (isError) {
    return (
      <div className="rounded-xl border border-[hsl(var(--destructive))]/20 bg-[hsl(var(--destructive))]/5 p-4">
        <div className="text-sm font-semibold text-[hsl(var(--destructive))]">
          Fleet state failed to load
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          className="mt-2 text-sm text-muted-foreground underline"
        >
          Retry
        </button>
      </div>
    )
  }

  const agents = data?.agents ?? []
  if (agents.length === 0) {
    return (
      <div className="px-1 py-8 text-sm text-muted-foreground">
        No agents in this workspace yet.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2 px-1">
        <p className="text-sm text-muted-foreground">
          What each agent is doing right now. Cost is the rolling last 24h.
        </p>
        {/*
          Source-degradation indicators. A degraded watches/asks source defaults
          its fields to zero (fail-soft); flag it so those zeros are not read as
          a clean bill of health (P228-RVW-6). Destructive tone = uncounted data,
          not a real zero; muted tone for the always-omittable cost.
        */}
        <div className="flex shrink-0 items-center gap-2">
          {data && !data.cost_available && (
            <Badge variant="outline" className="text-muted-foreground">
              cost unavailable
            </Badge>
          )}
          {data && !data.watches_available && (
            <Badge variant="outline" className="text-[hsl(var(--destructive))]">
              watches unavailable
            </Badge>
          )}
          {data && !data.asks_available && (
            <Badge variant="outline" className="text-[hsl(var(--destructive))]">
              asks unavailable
            </Badge>
          )}
        </div>
      </div>
      <div className="space-y-2">
        {data?.hosts !== undefined && <HostCard hosts={data.hosts} />}
        {agents.map((agent) => (
          <FleetRow key={agent.agent_id} agent={agent} onViewDetails={onViewDetails} />
        ))}
      </div>
    </div>
  )
}
