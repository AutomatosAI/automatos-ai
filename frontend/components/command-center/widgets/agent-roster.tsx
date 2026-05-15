'use client'

/**
 * AgentRoster — workforce panel for the Summary tab.
 * Live data via `useAgents()`. Shows up to `limit` agents sorted with
 * active ones first, then by most recent run.
 */

import { useMemo } from 'react'
import { useAgents } from '@/hooks/use-agent-api'
import { toneFor } from '../agent-tones'

interface RosterAgent {
  id: string | number
  name: string
  description: string | null
  active: boolean
  runs: number
  spend: string
}

function isActive(agent: any): boolean {
  const status = (agent.status || '').toLowerCase()
  return status === 'active' || status === 'working' || status === 'running'
}

function spendLabel(agent: any): string {
  const cost = agent.model_usage_stats?.total_cost
  if (typeof cost !== 'number' || cost === 0) return '—'
  if (cost >= 1) return `$${cost.toFixed(2)}`
  return `$${cost.toFixed(2)}`
}

function runsCount(agent: any): number {
  return (
    agent.model_usage_stats?.total_requests ??
    agent.performance_metrics?.total_tasks_completed ??
    0
  )
}

export function AgentRoster({ limit = 8 }: { limit?: number }) {
  const { data, isLoading } = useAgents()

  const roster = useMemo<RosterAgent[]>(() => {
    if (!data) return []
    const rows = (data as any[]).map((a) => ({
      id: a.id,
      name: a.name,
      description:
        a.role ||
        a.description ||
        a.configuration?.role ||
        a.agent_type ||
        null,
      active: isActive(a),
      runs: runsCount(a),
      spend: spendLabel(a),
    }))
    // Active first, then by runs descending
    return rows
      .sort((a, b) => {
        if (a.active !== b.active) return a.active ? -1 : 1
        return b.runs - a.runs
      })
      .slice(0, limit)
  }, [data, limit])

  if (isLoading) {
    return <div className="cc-panel-empty">Loading roster…</div>
  }
  if (roster.length === 0) {
    return (
      <div className="cc-panel-empty">
        No agents in this workspace yet. Add one from Agent Management.
      </div>
    )
  }

  return (
    <div className="cc-roster">
      {roster.map((a) => {
        const tone = toneFor(a.name)
        return (
          <div key={a.id} className="cc-roster-row">
            <div style={{ minWidth: 0 }}>
              <div className="nm">
                <span
                  aria-hidden
                  style={{
                    display: 'inline-block',
                    width: 8,
                    height: 8,
                    borderRadius: 2,
                    background: tone.bg,
                    marginRight: 8,
                    verticalAlign: 'middle',
                  }}
                />
                {a.name}
              </div>
              {a.description && <div className="det">{a.description}</div>}
            </div>
            <div>
              {a.active ? (
                <span className="cc-pill-ok">● ACTIVE</span>
              ) : (
                <span
                  className="cc-pill-ok"
                  style={{ background: 'hsl(var(--muted))', color: 'hsl(var(--muted-foreground))' }}
                >
                  IDLE
                </span>
              )}
            </div>
            <div className="v">{a.runs} runs</div>
            <div className={`v${a.spend === '—' ? '' : ' ok'}`}>{a.spend}</div>
          </div>
        )
      })}
    </div>
  )
}
