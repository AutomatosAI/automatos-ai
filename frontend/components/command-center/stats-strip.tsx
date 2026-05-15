'use client'

/**
 * StatsStrip — Four prominent stat cards (Working / Agents / Queue /
 * Attention), Studio-styled on cream paper. Same four counts as the
 * classic StatsBar so nothing is lost; new aesthetic.
 *
 * Auto-swaps to a single editorial sentence when all four are zero so the
 * page doesn't shout "0/0/0/0" on a quiet morning. The prose variant has
 * NO action buttons — Command Centre is a monitoring surface, not a
 * creation surface (those CTAs belong on Assignments / Chat).
 */

import { Activity, Users, ListTodo, AlertTriangle } from 'lucide-react'
import { useActivityStats } from '@/hooks/use-activity-api'

type Tone = 'ok' | 'err' | 'warn' | 'info' | 'muted'

interface StatCard {
  label: string
  value: number
  tone: Tone
  delta: string
  Icon: React.ComponentType<{ style?: React.CSSProperties }>
}

export function StatsStrip() {
  const { data: stats } = useActivityStats('1d')

  const working = stats?.working_now ?? 0
  const agents = stats?.agents_active ?? 0
  const queue = stats?.tasks_in_queue ?? 0
  const attn = stats?.needs_attention ?? 0

  const isQuiet = working === 0 && attn === 0 && queue === 0

  if (isQuiet) {
    return (
      <div className="cc-stats-prose">
        <div className="pulse">
          <MoonStar />
        </div>
        <div className="phrase">
          A quiet hour. <span className="num">{agents}</span>{' '}
          agent{agents === 1 ? '' : 's'} working,{' '}
          <span className="num">{queue}</span> in queue. Routines keep the
          lights on while you focus on something else.
        </div>
      </div>
    )
  }

  const cards: StatCard[] = [
    {
      label: 'Working Now',
      value: working,
      tone: working > 0 ? 'ok' : 'muted',
      delta: working > 0 ? 'running' : 'idle',
      Icon: Activity,
    },
    {
      label: 'Agents Active',
      value: agents,
      tone: 'info',
      delta: agents > 0 ? `${agents} on duty` : 'workforce idle',
      Icon: Users,
    },
    {
      label: 'Tasks in Queue',
      value: queue,
      tone: queue > 0 ? 'warn' : 'muted',
      delta: queue > 0 ? `${queue} waiting` : 'all clear',
      Icon: ListTodo,
    },
    {
      label: 'Needs Attention',
      value: attn,
      tone: attn > 0 ? 'err' : 'muted',
      delta: attn > 0 ? 'needs you' : '0 open',
      Icon: AlertTriangle,
    },
  ]

  return (
    <div className="cc-stat-grid">
      {cards.map((c) => {
        const Icon = c.Icon
        return (
          <div key={c.label} className="cc-stat-card">
            <div className="cc-stat-icon" data-tone={c.tone}>
              <Icon style={{ width: 18, height: 18 }} />
            </div>
            <div className="cc-stat-body">
              <div className={`cc-stat-val ${c.tone}`}>{c.value}</div>
              <div className="cc-stat-label">{c.label}</div>
              <div className="cc-stat-delta">{c.delta}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function MoonStar() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      <path d="M19 3v4" />
      <path d="M21 5h-4" />
    </svg>
  )
}
