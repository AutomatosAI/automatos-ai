'use client'

/**
 * PRD-251B US-B208 — a plan's slot that is not made yet, on the Socials calendar: dashed, with
 * the time and format ("09:00 · Video 0:30"), its topic once one is pinned to the day (else the
 * plan's name), the channel badges and the word Planned. Dragging it moves the slot.
 */
import type { CSSProperties, HTMLAttributes } from 'react'

import { cn } from '@/lib/utils'
import { wallInZone } from '@/lib/social-time'
import { CHANNEL_BADGE, CHIP_KEY, TONES } from './socials-calendar-chip'
import { channelBadge, formatLabel } from './socials-calendar-model'
import { plannedTitle, type PlannedSlot } from './plan-calendar-model'

interface PlannedChipProps {
  planned: PlannedSlot
  onOpen: () => void
  dragProps?: HTMLAttributes<HTMLElement>
  style?: CSSProperties
}

export function SocialsPlannedChip({ planned, onOpen, dragProps, style }: PlannedChipProps) {
  const { slot } = planned
  const time = wallInZone(slot.at, planned.timezone).slice(11, 16)
  const format = formatLabel({ format: slot.format, length_seconds: slot.length_seconds })
  const badges = Array.from(new Set(slot.channels.map(channelBadge)))
  const title = plannedTitle(planned)
  const label = [title, `${time} ${format}`, ...badges, 'Planned'].join(', ')
  return (
    <button type="button" {...dragProps} onClick={onOpen} aria-label={label} title={label} style={style} data-status="planned"
      className={cn('flex w-full min-w-0 flex-col gap-[3px] rounded-lg border px-2 py-1.5 text-left text-xs leading-[1.3] text-foreground', TONES.planned.chip)}>
      <span className={cn(CHIP_KEY, 'truncate')}>{time} · {format}</span>
      <span className="truncate">{title}</span>
      <span className="flex items-center justify-between gap-1">
        <span className="flex min-w-0 gap-[3px]">
          {badges.map((badge) => <span key={badge} className={CHANNEL_BADGE}>{badge}</span>)}
        </span>
        <span className={cn(CHIP_KEY, TONES.planned.key)}>Planned</span>
      </span>
    </button>
  )
}
