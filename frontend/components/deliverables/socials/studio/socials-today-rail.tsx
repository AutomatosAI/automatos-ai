'use client'

/**
 * PRD-251B US-B108 — the calendar's right rail (Main.dc.html): Today, with the day's posts
 * and "Review N posts" into the Queue (N = today's posts waiting for approval), and the
 * Status key: the six words and what each means. The plan card arrives with Wave 2.
 */
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { SocialPost } from '@/lib/api-client'
import { CHIP_KEY, SocialsCalendarChip, TONES } from './socials-calendar-chip'
import { STATUS_KEY, isOnDay, opensInQueue, slotOf } from './socials-calendar-model'
import { PRIMARY_ACTION } from './studio-nav'

const CARD = 'flex flex-col gap-3 rounded-xl border border-border bg-card p-[18px]'

/** "Today · Wed 14 Oct". */
export function todayHeading(now: Date): string {
  const day = now.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }).replace(',', '')
  return `Today · ${day}`
}

export function reviewLabel(count: number): string {
  return count === 1 ? 'Review 1 post' : `Review ${count} posts`
}

interface SocialsTodayRailProps {
  posts: ReadonlyArray<SocialPost>
  onOpen: (post: SocialPost) => void
  onReview: () => void
}

export function SocialsTodayRail({ posts, onOpen, onReview }: SocialsTodayRailProps) {
  const now = new Date()
  const today = useMemo(
    () =>
      posts
        .map((post) => ({ post, slot: slotOf(post) }))
        .filter((entry): entry is { post: SocialPost; slot: string } => !!entry.slot && isOnDay(entry.slot, new Date()))
        .sort((a, b) => a.slot.localeCompare(b.slot)),
    [posts],
  )
  const waiting = today.filter(({ post }) => opensInQueue(post)).length

  return (
    <aside className="flex flex-col gap-4">
      <section aria-label="Today" className={CARD}>
        <h2 className="text-[15px] font-semibold text-foreground">{todayHeading(now)}</h2>
        {today.length === 0 ? (
          <p className="text-[12.5px] leading-[1.45] text-muted-foreground">Nothing is planned for today.</p>
        ) : (
          today.map(({ post, slot }) => (
            <SocialsCalendarChip key={post.id} post={post} slot={slot} onOpen={() => onOpen(post)} className="px-3 py-2.5" />
          ))
        )}
        {waiting > 0 && (
          <Button type="button" className={PRIMARY_ACTION} onClick={onReview}>
            {reviewLabel(waiting)}
          </Button>
        )}
      </section>
      <section aria-label="Status key" className={CARD}>
        <h2 className="text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground">Status</h2>
        <ul className="grid grid-cols-2 gap-2">
          {STATUS_KEY.map((entry) => (
            <li key={entry.word} className={cn('flex flex-col gap-[3px] rounded-lg border px-2 py-1.5', TONES[entry.tone].chip)}>
              <span className={cn(CHIP_KEY, TONES[entry.tone].key)}>{entry.word}</span>
              <span className="text-xs leading-[1.3] text-foreground">{entry.meaning}</span>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  )
}
