'use client'

/**
 * PRD-251 S2.1 — List | Board, in the Socials tab's header. It looks like the
 * shared ViewToggle (components/shared/view-toggle.tsx), which is fixed to grid
 * and list; both views read the same posts. The campaigns (S2.4) moved to the
 * Studio's Plans view (PRD-251B US-B107).
 */
import { Kanban, List, type LucideIcon } from 'lucide-react'

import { Button } from '@/components/ui/button'

export type SocialsView = 'list' | 'board'

const VIEWS: ReadonlyArray<{ value: SocialsView; label: string; icon: LucideIcon }> = [
  { value: 'list', label: 'List', icon: List },
  { value: 'board', label: 'Board', icon: Kanban },
]

interface SocialsViewToggleProps {
  value: SocialsView
  onChange: (view: SocialsView) => void
}

export function SocialsViewToggle({ value, onChange }: SocialsViewToggleProps) {
  return (
    <div
      role="group"
      aria-label="Show posts as"
      className="socials-view-toggle flex items-center gap-1 rounded-lg bg-secondary/30 p-1"
    >
      {VIEWS.map(({ value: view, label, icon: Icon }) => (
        <Button
          key={view}
          type="button"
          size="sm"
          variant={value === view ? 'default' : 'ghost'}
          aria-pressed={value === view}
          onClick={() => onChange(view)}
          className="h-7 px-2.5"
        >
          <Icon className="mr-1.5 h-4 w-4" aria-hidden />
          {label}
        </Button>
      ))}
    </div>
  )
}
