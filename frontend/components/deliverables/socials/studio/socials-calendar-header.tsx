'use client'

/**
 * PRD-251B US-B108 — the Socials calendar's header (Main.dc.html): the month (or week)
 * title, Previous / Next / Today, the View group Month | Week | List, and the Show
 * channels chips: All channels, one per connected channel, Video.
 */
import { ChevronLeft, ChevronRight } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { SocialChannel } from '@/lib/api-client'
import { ALL_CHANNELS, VIDEO_ONLY, type ChannelFilter } from './socials-calendar-model'
import type { CalendarLayout } from './studio-route'

const LAYOUTS: ReadonlyArray<{ value: CalendarLayout; label: string }> = [
  { value: 'month', label: 'Month' },
  { value: 'week', label: 'Week' },
  { value: 'list', label: 'List' },
]

const SEGMENT = 'h-[38px] rounded-lg px-3.5 text-sm font-medium text-muted-foreground transition-colors'
const SEGMENT_ON = 'bg-secondary text-foreground ring-1 ring-inset ring-border'
const FILTER_CHIP = 'h-[38px] rounded-full border border-border px-3.5 text-sm text-muted-foreground transition-colors'
const FILTER_ON = 'border-accent bg-accent/15 text-foreground'

interface SocialsCalendarHeaderProps {
  title: string
  layout: CalendarLayout
  channels: ReadonlyArray<SocialChannel>
  filter: ChannelFilter
  onLayout: (layout: CalendarLayout) => void
  onShift: (direction: -1 | 0 | 1) => void
  onFilter: (filter: ChannelFilter) => void
}

export function SocialsCalendarHeader(props: SocialsCalendarHeaderProps) {
  const { title, layout, channels, filter, onLayout, onShift, onFilter } = props
  const unit = layout === 'week' ? 'week' : 'month'
  const filters = [
    { id: ALL_CHANNELS, label: 'All channels' },
    ...channels.map((channel) => ({ id: channel.toolkit, label: channel.label })),
    { id: VIDEO_ONLY, label: 'Video' },
  ]
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex flex-wrap items-center gap-3.5">
        <h1 className="font-serif text-[32px] font-normal leading-[1.1] tracking-[-0.01em] text-foreground max-md:text-[26px]">
          {title}
        </h1>
        {layout !== 'list' && (
          <div className="flex items-center gap-0.5">
            <Button type="button" size="sm" variant="ghost" aria-label={`Previous ${unit}`} onClick={() => onShift(-1)}>
              <ChevronLeft className="h-[18px] w-[18px]" aria-hidden />
            </Button>
            <Button type="button" size="sm" variant="ghost" aria-label={`Next ${unit}`} onClick={() => onShift(1)}>
              <ChevronRight className="h-[18px] w-[18px]" aria-hidden />
            </Button>
            <Button type="button" size="sm" variant="secondary" onClick={() => onShift(0)}>
              Today
            </Button>
          </div>
        )}
        <div role="group" aria-label="View" className="flex rounded-xl border border-border bg-background/60 p-[3px]">
          {LAYOUTS.map((item) => (
            <button
              key={item.value}
              type="button"
              aria-pressed={layout === item.value}
              className={cn(SEGMENT, layout === item.value && SEGMENT_ON)}
              onClick={() => onLayout(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
      {layout !== 'list' && (
      <div role="group" aria-label="Show channels" className="flex flex-wrap gap-1.5">
        {filters.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={filter === item.id}
            className={cn(FILTER_CHIP, filter === item.id && FILTER_ON)}
            onClick={() => onFilter(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      )}
    </div>
  )
}
