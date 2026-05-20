'use client'

import {
  Orbit,
  BookMarked,
  Lightbulb,
  Zap,
  Sparkles,
  ArrowRight,
} from 'lucide-react'

export type EntryType = 'mission' | 'playbook' | 'plan' | 'task'

interface EntryDef {
  id: EntryType
  label: string
  desc: string
  when: string
  Icon: typeof Orbit
  tone: 'mission' | 'playbook' | 'plan' | 'task'
}

export const ENTRY_TYPES: ReadonlyArray<EntryDef> = [
  {
    id: 'mission',
    label: 'Mission',
    desc: 'Big, complex work. 6–9 agents, field memory, parallel reasoning.',
    when: 'When the goal needs more than one head.',
    Icon: Orbit,
    tone: 'mission',
  },
  {
    id: 'playbook',
    label: 'Playbook',
    desc: 'Reusable routine. Schedulable, triggerable, version-controlled.',
    when: 'When you’ve done this before — and will again.',
    Icon: BookMarked,
    tone: 'playbook',
  },
  {
    id: 'plan',
    label: 'Plan',
    desc: 'Iterate with Auto. Refine the steps before launching the work.',
    when: 'When the shape isn’t clear yet.',
    Icon: Lightbulb,
    tone: 'plan',
  },
  {
    id: 'task',
    label: 'Task',
    desc: 'Quick single action. One agent, one shot.',
    when: 'When you just need it done.',
    Icon: Zap,
    tone: 'task',
  },
]

interface EntryGridProps {
  recommended?: EntryType
  onPick: (id: EntryType) => void
}

export function EntryGrid({ recommended = 'mission', onPick }: EntryGridProps) {
  return (
    <div className="entry-grid">
      {ENTRY_TYPES.map((e) => (
        <button
          key={e.id}
          type="button"
          className="entry-card"
          onClick={() => onPick(e.id)}
        >
          {e.id === recommended && (
            <span className="new-pill">
              <Sparkles style={{ width: 10, height: 10 }} />
              RECOMMENDED
            </span>
          )}
          <span className={`em ${e.tone}`}>
            <e.Icon style={{ width: 18, height: 18, strokeWidth: 1.7 }} />
          </span>
          <div className="nm">{e.label}</div>
          <div className="desc">{e.desc}</div>
          <div className="when">
            <ArrowRight style={{ width: 11, height: 11 }} />
            <span>{e.when}</span>
          </div>
        </button>
      ))}
    </div>
  )
}
