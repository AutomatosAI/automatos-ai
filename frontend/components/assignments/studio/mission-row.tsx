'use client'

import { formatDistanceToNowStrict } from 'date-fns'
import type { KeyboardEvent } from 'react'
import { RowStatusLabel } from '@/components/proactive/row-status'
import { missionRetryTarget } from '@/components/proactive/use-row-retry'
import type { MissionResponse, RunState } from '@/types/missions'

type Tone = 'ok' | 'err' | 'warn' | 'info' | 'run' | 'muted'

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

const PIP_CHAR: Record<Tone, string> = {
  run: '↻',
  ok: '✓',
  warn: '!',
  err: '!',
  info: '◷',
  muted: '·',
}

function shortId(id: string) {
  const t = id.replace(/-/g, '').toUpperCase()
  return t.length > 6 ? t.slice(0, 6) : t
}

function fmtTokens(n: number) {
  if (!n) return '0'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function age(ts: string | null) {
  if (!ts) return '—'
  try {
    return formatDistanceToNowStrict(new Date(ts), { addSuffix: false })
  } catch {
    return '—'
  }
}

interface MissionRowProps {
  mission: MissionResponse
  onOpen: (id: string) => void
}

/**
 * One Mission in the grouped list. PRD-268 US-W04: under its goal, where it stands, what happens next and Try
 * again (RowStatusLabel). Try again is a button of its own, so the row is a button by role (as a board card is,
 * issue #1090): a click opens it, and Enter or Space opens it only when the row itself has focus.
 */
export function MissionRow({ mission, onOpen }: MissionRowProps) {
  const tone = STATE_TO_TONE[mission.state]
  const pip = PIP_CHAR[tone]
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) return
    event.preventDefault()
    onOpen(mission.id)
  }
  return (
    <div
      role="button"
      tabIndex={0}
      className="mis-row w-full cursor-pointer"
      onClick={() => onOpen(mission.id)}
      onKeyDown={onKeyDown}
    >
      <span className={`pip ${tone}`}>{pip}</span>
      <span className="id">{shortId(mission.id)}</span>
      <div style={{ minWidth: 0 }}>
        <div className="nm">{mission.goal}</div>
        <div className="sub">
          {mission.parallel_groups.length} parallel group
          {mission.parallel_groups.length === 1 ? '' : 's'}
          {mission.complexity_tier ? ` · ${mission.complexity_tier}` : ''}
        </div>
        <RowStatusLabel row={mission} target={missionRetryTarget(mission)} title={mission.goal} className="mt-1" />
      </div>
      <span style={{ fontFamily: 'var(--serif)', fontSize: 13 }}>
        {mission.created_by || '—'}
      </span>
      <span className="num">{mission.parallel_groups.length || 1}</span>
      <span className="num">{fmtTokens(mission.tokens_used)}</span>
      <span className="age">{age(mission.updated_at || mission.created_at)}</span>
    </div>
  )
}
