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
  schedule_config?: { cron?: string; interval_seconds?: number } | null
  trigger_type?: string | null
}

interface PlaybookCardProps {
  playbook: PlaybookLike
  onRun: (p: PlaybookLike) => void
}

function describeTrigger(p: PlaybookLike): { label: string; kind: 'cron' | 'webhook' | 'manual' } {
  const t = (p.trigger_type || '').toLowerCase()
  if (t === 'cron' || p.schedule_config?.cron) {
    return { label: `cron · ${p.schedule_config?.cron ?? 'scheduled'}`, kind: 'cron' }
  }
  if (t === 'webhook') return { label: 'webhook', kind: 'webhook' }
  return { label: 'manual', kind: 'manual' }
}

export function PlaybookCard({ playbook, onRun }: PlaybookCardProps) {
  const trig = describeTrigger(playbook)
  const TrigIcon = trig.kind === 'cron' ? Clock : trig.kind === 'webhook' ? TriggerZap : Hand

  const runs = playbook.use_count ?? playbook.execution_count ?? 0
  const success = playbook.success_rate ?? null
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
        {success != null && (
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
