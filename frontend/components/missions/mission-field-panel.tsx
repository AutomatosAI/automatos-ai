'use client'

import { useMemo, useState } from 'react'
import { Brain, Zap, TrendingDown, Eye, Users, Activity, Clock, BarChart3, AlertTriangle, ServerCrash } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMissionField } from '@/hooks/use-missions-api'
import type { FieldPattern, FieldScope } from '@/hooks/use-missions-api'
import dynamic from 'next/dynamic'

const MissionFieldViz = dynamic(
  () => import('./mission-field-viz').then(mod => ({ default: mod.MissionFieldViz })),
  { ssr: false, loading: () => <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground animate-pulse">Loading 3D field...</div> }
)

interface MissionFieldPanelProps {
  missionId: string
  className?: string
}

// Agent colors for visual differentiation
const AGENT_COLORS = [
  'bg-info/20 border-info/40 text-info',
  'bg-success/20 border-success/40 text-success',
  'bg-agent/20 border-agent/40 text-agent',
  'bg-warning/20 border-warning/40 text-warning',
  'bg-rose-500/20 border-rose-500/40 text-rose-400',
  'bg-cyan-500/20 border-cyan-500/40 text-cyan-400',
  'bg-warning/20 border-warning/40 text-warning',
]

const AGENT_DOT_COLORS = [
  'bg-blue-500',
  'bg-success',
  'bg-agent',
  'bg-warning',
  'bg-rose-500',
  'bg-cyan-500',
  'bg-warning',
]

function getAgentColor(agentId: number, index: number) {
  return AGENT_COLORS[index % AGENT_COLORS.length]
}

function getAgentDotColor(index: number) {
  return AGENT_DOT_COLORS[index % AGENT_DOT_COLORS.length]
}

function StrengthBar({ strength, maxStrength }: { strength: number; maxStrength: number }) {
  const pct = maxStrength > 0 ? (strength / maxStrength) * 100 : 0
  return (
    <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
      <div
        className={cn(
          'h-full rounded-full transition-all duration-300',
          strength > 0.7 ? 'bg-success' :
          strength > 0.3 ? 'bg-warning' :
          strength > 0.05 ? 'bg-warning' :
          'bg-destructive/50'
        )}
        style={{ width: `${Math.max(pct, 2)}%` }}
      />
    </div>
  )
}

function PatternCard({ pattern, agentIndex, maxStrength }: {
  pattern: FieldPattern
  agentIndex: number
  maxStrength: number
}) {
  const ageMs = Date.now() - new Date(pattern.last_accessed).getTime()
  const ageMinutes = Math.floor(ageMs / 60_000)
  const ageStr = ageMinutes < 60 ? `${ageMinutes}m ago` : `${Math.floor(ageMinutes / 60)}h ${ageMinutes % 60}m ago`

  return (
    <div className={cn(
      'rounded-lg border p-3 space-y-2 transition-all',
      pattern.is_archived ? 'opacity-40' : '',
      getAgentColor(pattern.agent_id, agentIndex),
    )}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className={cn('w-2 h-2 rounded-full shrink-0', getAgentDotColor(agentIndex))} />
          <span className="text-xs font-medium truncate">{pattern.key}</span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {pattern.access_count > 0 && (
            <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
              <Eye className="w-2.5 h-2.5" />
              {pattern.access_count}
            </span>
          )}
          <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
            <Clock className="w-2.5 h-2.5" />
            {ageStr}
          </span>
        </div>
      </div>

      <p className="text-[11px] text-muted-foreground line-clamp-2 leading-relaxed">
        {pattern.value}
      </p>

      <div className="flex items-center gap-2">
        <StrengthBar strength={pattern.decayed_strength} maxStrength={maxStrength} />
        <span className="text-[10px] text-muted-foreground tabular-nums shrink-0">
          {(pattern.decayed_strength * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

export function MissionFieldPanel({ missionId, className }: MissionFieldPanelProps) {
  const [scope, setScope] = useState<FieldScope>('mission')
  const { data, isLoading } = useMissionField(missionId, true, scope)

  const agentMap = useMemo(() => {
    if (!data?.patterns) return new Map<number, number>()
    const uniqueAgents = [...new Set(data.patterns.map(p => p.agent_id))]
    return new Map(uniqueAgents.map((id, i) => [id, i]))
  }, [data?.patterns])

  const maxStrength = useMemo(() => {
    if (!data?.patterns.length) return 1
    return Math.max(...data.patterns.map(p => p.decayed_strength))
  }, [data?.patterns])

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center h-full', className)}>
        <div className="text-sm text-muted-foreground animate-pulse">Loading field...</div>
      </div>
    )
  }

  // Derive a fallback status if the API didn't return one (back-compat)
  const status = data?.status
    ?? (!data?.field_id ? 'not_created' : (data.patterns?.length ? 'active' : 'empty'))

  if (status === 'not_created') {
    return (
      <div className={cn('flex flex-col items-center justify-center h-full gap-3 p-6', className)}>
        <Brain className="w-10 h-10 text-muted-foreground/30" />
        <div className="text-sm text-muted-foreground text-center">
          No shared field active for this mission.
          <br />
          <span className="text-xs">The field is created when the mission starts running.</span>
        </div>
      </div>
    )
  }

  if (status === 'missing') {
    return (
      <div className={cn('flex flex-col items-center justify-center h-full gap-3 p-6', className)}>
        <AlertTriangle className="w-10 h-10 text-warning/60" />
        <div className="text-sm text-foreground text-center font-medium">
          Field collection missing
        </div>
        <div className="text-xs text-muted-foreground text-center max-w-xs leading-relaxed">
          The mission references field <code className="text-[10px] font-mono">{data?.field_id?.slice(0, 8)}…</code> but the underlying collection was destroyed.
          <br />
          <span className="text-warning/80">The coordinator will recreate it on the next tick.</span>
        </div>
      </div>
    )
  }

  if (status === 'unavailable') {
    return (
      <div className={cn('flex flex-col items-center justify-center h-full gap-3 p-6', className)}>
        <ServerCrash className="w-10 h-10 text-destructive/60" />
        <div className="text-sm text-foreground text-center font-medium">
          Field backend unavailable
        </div>
        <div className="text-xs text-muted-foreground text-center max-w-xs">
          The shared-context backend (Qdrant) is unreachable.
        </div>
      </div>
    )
  }

  if (!data) {
    return null
  }

  const { patterns, stability, metrics } = data
  const activePatterns = patterns.filter(p => !p.is_archived)
  const archivedPatterns = patterns.filter(p => p.is_archived)

  return (
    <div className={cn('h-full flex flex-col', className)}>
      {/* Header */}
      <div className="p-3 border-b border-border space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5" />
            {scope === 'workspace' ? 'Workspace Field' : 'Shared Field'}
          </h3>
          {/* PRD-166 S1/S4: mission ↔ workspace-persistent field scope */}
          <div className="flex items-center gap-1 rounded-md border border-border p-0.5">
            {(['mission', 'workspace'] as FieldScope[]).map((s) => (
              <button
                key={s}
                onClick={() => setScope(s)}
                className={cn(
                  'px-1.5 py-0.5 text-[10px] rounded transition-colors capitalize',
                  scope === s ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Stability gauge */}
        <div className="grid grid-cols-4 gap-2">
          <div className="text-center">
            <div className="text-lg font-bold tabular-nums text-foreground">
              {(stability.stability * 100).toFixed(0)}%
            </div>
            <div className="text-[10px] text-muted-foreground">Stability</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold tabular-nums text-foreground">
              {stability.pattern_count}
            </div>
            <div className="text-[10px] text-muted-foreground">Patterns</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold tabular-nums text-foreground">
              {stability.active_patterns ?? activePatterns.length}
            </div>
            <div className="text-[10px] text-muted-foreground">Active</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold tabular-nums text-foreground">
              {agentMap.size}
            </div>
            <div className="text-[10px] text-muted-foreground">Agents</div>
          </div>
        </div>

        {/* Agent legend */}
        {agentMap.size > 0 && (
          <div className="flex flex-wrap gap-2">
            {[...agentMap.entries()].map(([agentId, index]) => {
              const count = patterns.filter(p => p.agent_id === agentId).length
              return (
                <div key={agentId} className="flex items-center gap-1.5">
                  <div className={cn('w-2 h-2 rounded-full', getAgentDotColor(index))} />
                  <span className="text-[10px] text-muted-foreground">
                    {agentId === 0 ? 'System' : `Agent ${agentId}`} ({count})
                  </span>
                </div>
              )
            })}
          </div>
        )}

        {/* Metrics row */}
        {metrics && (
          <div className="flex gap-3 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1">
              <Zap className="w-2.5 h-2.5" />
              {metrics.total_injections} injections
            </span>
            <span className="flex items-center gap-1">
              <Eye className="w-2.5 h-2.5" />
              {metrics.total_queries} queries
            </span>
            {metrics.avg_query_latency_ms > 0 && (
              <span className="flex items-center gap-1">
                <Activity className="w-2.5 h-2.5" />
                {metrics.avg_query_latency_ms.toFixed(0)}ms avg
              </span>
            )}
          </div>
        )}
      </div>

      {/* Neural field visualization */}
      <div className="relative" style={{ height: '480px' }}>
        {patterns.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Brain className="w-8 h-8 mb-2 opacity-30 animate-pulse" />
            <span className="text-xs">Field is empty. Waiting for agent activity...</span>
          </div>
        ) : (
          <MissionFieldViz missionId={missionId} patterns={patterns} className="absolute inset-0" />
        )}
      </div>
    </div>
  )
}
