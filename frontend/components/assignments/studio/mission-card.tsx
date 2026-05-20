'use client'

import { ArrowRight } from 'lucide-react'
import { formatDistanceToNowStrict } from 'date-fns'
import type { MissionResponse, RunState } from '@/types/missions'

type Tone = 'ok' | 'err' | 'warn' | 'info' | 'run' | 'muted'

interface MissionCardProps {
  mission: MissionResponse
  onOpen: (id: string) => void
}

const STATE_TO_TONE: Record<RunState, Tone> = {
  pending: 'muted',
  planning: 'info',
  awaiting_approval: 'warn',
  running: 'run',
  paused: 'info',
  verifying: 'info',
  awaiting_human: 'warn',
  completed: 'ok',
  failed: 'err',
  cancelled: 'muted',
}

const STATE_LABEL: Record<RunState, string> = {
  pending: 'PENDING',
  planning: 'PLANNING',
  awaiting_approval: 'NEEDS APPROVAL',
  running: 'RUNNING',
  paused: 'PAUSED',
  verifying: 'VERIFYING',
  awaiting_human: 'NEEDS YOU',
  completed: 'COMPLETED',
  failed: 'FAILED',
  cancelled: 'CANCELLED',
}

const STATE_CTA: Record<RunState, string> = {
  pending: 'Open',
  planning: 'Watch',
  awaiting_approval: 'Approve',
  running: 'Watch live',
  paused: 'Resume',
  verifying: 'Watch',
  awaiting_human: 'Take call',
  completed: 'Review',
  failed: 'Re-plan',
  cancelled: 'Open',
}

function shortId(id: string) {
  // M-2027 style: take first 4 alphanumerics after a dash, or trim 8 chars.
  const t = id.replace(/-/g, '').toUpperCase()
  return t.length > 6 ? t.slice(0, 6) : t
}

function age(ts: string | null) {
  if (!ts) return ''
  try {
    return formatDistanceToNowStrict(new Date(ts), { addSuffix: false })
  } catch {
    return ''
  }
}

function fmtTokens(n: number) {
  if (!n) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

export function MissionCard({ mission, onOpen }: MissionCardProps) {
  const tone = STATE_TO_TONE[mission.state]
  const label = STATE_LABEL[mission.state]
  const cta = STATE_CTA[mission.state]

  const stateClass =
    mission.state === 'awaiting_approval' || mission.state === 'awaiting_human'
      ? 'is-needsapproval'
      : mission.state === 'failed'
      ? 'is-failed'
      : ''

  const stateDot =
    tone === 'err' ? 'hsl(var(--accent))'
    : tone === 'ok' ? 'hsl(82 50% 22%)'
    : tone === 'warn' ? 'hsl(38 78% 50%)'
    : tone === 'info' ? 'hsl(var(--info))'
    : tone === 'run' ? 'hsl(var(--foreground))'
    : 'hsl(40 30% 70%)'

  return (
    <button
      type="button"
      className={`mis-card ${stateClass}`}
      onClick={() => onOpen(mission.id)}
    >
      <div className="row1">
        <span className="id">{shortId(mission.id)}</span>
        <span
          className={`cc-status-pill ${tone === 'run' ? 'run' : tone}`}
          style={{ padding: '2px 7px', fontSize: 9.5 }}
        >
          ● {label}
        </span>
      </div>
      <div className="nm">{mission.goal}</div>

      <div className="row-meta">
        <div>
          <div className="l">STATE</div>
          <div className="v">{mission.state_type}</div>
        </div>
        <div>
          <div className="l">TOKENS</div>
          <div className="v">{fmtTokens(mission.tokens_used)}</div>
        </div>
        <div>
          <div className="l">PARALLEL</div>
          <div className="v">{mission.max_concurrent || 1}</div>
        </div>
        <div>
          <div className="l">TIER</div>
          <div className="v">{mission.complexity_tier || '—'}</div>
        </div>
      </div>

      <div className="last">
        <span className="pip" style={{ background: stateDot }} />
        <span
          style={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
            minWidth: 0,
          }}
        >
          {mission.state === 'failed' && mission.output_summary
            ? 'Failed — see summary'
            : mission.has_synthesis_tasks
            ? 'Includes synthesis step'
            : `${mission.parallel_groups.length} parallel group${
                mission.parallel_groups.length === 1 ? '' : 's'
              }`}
        </span>
      </div>

      <div className="row-actions-bottom">
        <span className="age">
          {age(mission.updated_at || mission.created_at)} ago
        </span>
        <span className="spacer" />
        <span
          className="l-btn"
          style={{
            background:
              mission.state === 'awaiting_approval' ||
              mission.state === 'awaiting_human'
                ? 'hsl(38 78% 50%)'
                : mission.state === 'failed'
                ? 'hsl(var(--accent))'
                : undefined,
            color:
              mission.state === 'awaiting_approval' ||
              mission.state === 'awaiting_human' ||
              mission.state === 'failed'
                ? '#fff'
                : undefined,
            borderColor:
              mission.state === 'awaiting_approval' ||
              mission.state === 'awaiting_human'
                ? 'hsl(38 78% 50%)'
                : mission.state === 'failed'
                ? 'hsl(var(--accent))'
                : undefined,
            pointerEvents: 'none',
          }}
        >
          {cta}
          <ArrowRight style={{ width: 11, height: 11 }} />
        </span>
      </div>
    </button>
  )
}
