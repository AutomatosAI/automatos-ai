'use client'

/** A calendar event's menu of actions (split out of calendar-tab.tsx, PRD-251 US-307). */
import type { ReactNode } from 'react'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import type { EventAction } from './calendar-actions'

export interface EventMenuGroup {
  label: string
  actions: EventAction[]
}

function MenuItems({ actions }: { actions: EventAction[] }) {
  return (
    <>
      {actions.map((a) => (
        <DropdownMenuItem
          key={a.label}
          onSelect={() => a.run()}
          className={a.tone === 'danger' ? 'text-destructive focus:text-destructive' : undefined}
        >
          {a.label}
        </DropdownMenuItem>
      ))}
    </>
  )
}

/** One event's actions, or — for a stacked card — a submenu per member. */
export function EventMenu({
  actions,
  groups,
  children,
}: {
  actions?: EventAction[]
  groups?: EventMenuGroup[]
  children: ReactNode
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[180px]">
        {actions && <MenuItems actions={actions} />}
        {(groups ?? []).map((g, i) => (
          <DropdownMenuSub key={`${g.label}-${i}`}>
            <DropdownMenuSubTrigger>{g.label}</DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <MenuItems actions={g.actions} />
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
