'use client'

/**
 * PRD-251B US-B108 — one post on the Socials calendar (Main.dc.html `.pc`): line 1 the time
 * in the post's timezone and the format ("09:00 · Video 0:30"), line 2 the title, line 3
 * the channel badges and the status WORD, so the state never rests on colour alone. Tones
 * per docs/PRDS/prd251b-reference/TOKENS.md. A click opens the post; a movable post drags.
 */
import type { CSSProperties, HTMLAttributes } from 'react'

import { cn } from '@/lib/utils'
import type { SocialPost } from '@/lib/api-client'
import { formatLabel, postBadges, postTime, statusWord, type StatusTone } from './socials-calendar-model'

export const CHIP_KEY = 'font-mono text-[10.5px] font-medium tracking-[.02em] text-muted-foreground'
export const CHANNEL_BADGE =
  'inline-flex h-5 min-w-[24px] items-center justify-center rounded-md bg-muted px-[5px] font-mono text-[10.5px] font-semibold text-foreground'

export const TONES: Record<StatusTone, { chip: string; key: string }> = {
  planned: { chip: 'border-dashed border-border bg-transparent', key: 'text-muted-foreground' },
  making: { chip: 'border-[hsl(var(--info))] bg-[hsl(var(--info)/0.10)]', key: 'text-[hsl(var(--info))]' },
  review: { chip: 'border-[hsl(var(--warning))] bg-[hsl(var(--warning)/0.10)]', key: 'text-[hsl(var(--warning))]' },
  scheduled: { chip: 'border-[hsl(var(--success)/0.7)] bg-secondary', key: 'text-[hsl(var(--success))]' },
  posted: { chip: 'border-border bg-card', key: 'text-muted-foreground' },
  missed: { chip: 'border-border bg-secondary', key: 'text-muted-foreground' },
  failed: { chip: 'border-destructive bg-secondary', key: 'text-destructive' },
}

interface SocialsCalendarChipProps {
  post: SocialPost
  /** The slot the chip shows (slotOf). */
  slot: string
  onOpen: () => void
  /** The drag props of a movable post; none keeps it in place. */
  dragProps?: HTMLAttributes<HTMLElement>
  style?: CSSProperties
  className?: string
}

export function SocialsCalendarChip({ post, slot, onOpen, dragProps, style, className }: SocialsCalendarChipProps) {
  const status = statusWord(post.status)
  if (!status) return null
  const tone = TONES[status.tone]
  const badges = postBadges(post)
  const time = postTime(post, slot)
  const format = formatLabel(post)
  // Read aloud as one sentence: the title first, then when, what, where and its state.
  const label = [post.title, `${time} ${format}`, ...badges, status.word].join(', ')
  return (
    <button
      type="button"
      {...dragProps}
      onClick={onOpen}
      aria-label={label}
      title={label}
      style={style}
      data-status={status.tone}
      className={cn(
        'flex w-full min-w-0 flex-col gap-[3px] rounded-lg border px-2 py-1.5 text-left text-xs leading-[1.3] text-foreground',
        'transition-colors hover:border-muted-foreground',
        tone.chip,
        className,
      )}
    >
      <span className={cn(CHIP_KEY, 'truncate')}>
        {time} · {format}
      </span>
      <span className="truncate">{post.title}</span>
      <span className="flex items-center justify-between gap-1">
        <span className="flex min-w-0 gap-[3px]">
          {badges.map((badge) => (
            <span key={badge} className={CHANNEL_BADGE}>
              {badge}
            </span>
          ))}
        </span>
        <span className={cn(CHIP_KEY, tone.key)}>{status.word}</span>
      </span>
    </button>
  )
}
