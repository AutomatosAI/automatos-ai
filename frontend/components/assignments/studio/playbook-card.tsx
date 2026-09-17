'use client'

import { Clock, Hand, Zap as TriggerZap, BookMarked } from 'lucide-react'

export interface PlaybookLike {
  id?: string | number
  template_id?: string
  name?: string
  description?: string | null
  icon?: string | null
  category?: string | null
  use_count?: number | null
  execution_count?: number | null
  success_rate?: number | null
  is_featured?: boolean
  steps?: unknown[] | null
  /**
   * Backend shape (orchestrator core.models.WorkflowTemplate):
   *   { type: 'manual' | 'cron' | 'trigger',
   *     cron_expression?: string,
   *     trigger_config?: { event?: string; webhook_id?: string } }
   */
  schedule_config?: {
    type?: string
    cron_expression?: string
    trigger_config?: { event?: string; webhook_id?: string } | null
  } | null
  trigger_type?: string | null
}

interface PlaybookCardProps {
  playbook: PlaybookLike
  onRun: (p: PlaybookLike) => void
}

type TriggerKind = 'cron' | 'trigger' | 'manual'

/** Humanize common cron patterns; fall back to the raw expression. */
function humanizeCron(cron: string): string {
  const c = cron.trim()
  if (c === '@daily' || c === '0 0 * * *') return 'daily · midnight'
  if (c === '@hourly' || c === '0 * * * *') return 'hourly'
  if (c === '@weekly' || c === '0 0 * * 0') return 'weekly'
  if (c === '@monthly' || c === '0 0 1 * *') return 'monthly'
  const m = c.match(/^0\s+(\d{1,2})\s+\*\s+\*\s+\*$/)
  if (m) {
    const h = Number(m[1])
    const tag = h === 0 ? 'midnight' : h === 12 ? 'noon' : `${h.toString().padStart(2, '0')}:00`
    return `daily · ${tag}`
  }
  return c
}

function describeTrigger(p: PlaybookLike): { label: string; kind: TriggerKind } {
  const sched = p.schedule_config ?? null
  const type = (sched?.type || p.trigger_type || 'manual').toLowerCase()
  if (type === 'cron' && sched?.cron_expression) {
    return { label: humanizeCron(sched.cron_expression), kind: 'cron' }
  }
  if (type === 'cron') return { label: 'scheduled', kind: 'cron' }
  if (type === 'trigger' || type === 'webhook' || type === 'event') {
    const evt = sched?.trigger_config?.event
    return { label: evt || 'triggered', kind: 'trigger' }
  }
  return { label: 'manual', kind: 'manual' }
}

export function PlaybookCard({ playbook, onRun }: PlaybookCardProps) {
  const trig = describeTrigger(playbook)
  const TrigIcon = trig.kind === 'cron' ? Clock : trig.kind === 'trigger' ? TriggerZap : Hand

  const runs = playbook.use_count ?? playbook.execution_count ?? 0
  // Only show success when there's data behind it. The backend defaults
  // success_rate to 0.0, so an unrun playbook always reads "0%" — that's
  // noise, not signal. Require at least one run.
  const showSuccess = runs > 0 && playbook.success_rate != null
  const success = playbook.success_rate ?? 0
  const steps = playbook.steps?.length ?? 0
  const featured = !!playbook.is_featured

  return (
    <button
      type="button"
      className={`pb-card ${featured ? 'featured' : ''}`}
      onClick={() => onRun(playbook)}
    >
      <div className="head">
        <span className="ic">
          <BookMarked style={{ width: 16, height: 16, strokeWidth: 1.7 }} />
        </span>
        <span className="nm">{playbook.name || 'Untitled playbook'}</span>
      </div>
      {featured && <span className="badge-feat">★ Featured</span>}
      {playbook.description && <div className="desc">{playbook.description}</div>}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className="trigger">
          <TrigIcon style={{ width: 10, height: 10 }} />
          {trig.label}
        </span>
        {playbook.category && (
          <span
            style={{
              marginLeft: 'auto',
              fontFamily: 'var(--font-geist-mono, monospace)',
              fontSize: 9.5,
              color: 'var(--fg-3)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            {playbook.category}
          </span>
        )}
      </div>
      <div className="stats">
        <div>
          <span className="l">Steps</span>
          <span className="v">{steps}</span>
        </div>
        <div>
          <span className="l">Runs</span>
          <span className="v">{runs}</span>
        </div>
        {showSuccess && (
          <div style={{ marginLeft: 'auto', alignItems: 'flex-end' }}>
            <span className="l">Success</span>
            <span
              className="v"
              style={{
                color:
                  success >= 99
                    ? 'hsl(82 50% 22%)'
                    : success >= 95
                    ? 'hsl(38 78% 27%)'
                    : 'hsl(var(--accent))',
              }}
            >
              {Math.round(success)}%
            </span>
          </div>
        )}
      </div>
    </button>
  )
}
