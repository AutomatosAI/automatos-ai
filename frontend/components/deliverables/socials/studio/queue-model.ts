/**
 * PRD-251B US-B111 — the Queue's reading of the posts (pure): what waits for approval, by
 * the day of its slot (today first, then later days, then no slot), the heading, when each
 * publishes and how long is left, and the pane's meta line.
 */
import type { SocialPost } from '@/lib/api-client'
import { hasChannels } from '../socials-review'
import { channelLabel } from '../socials-status'
import { formatLabel, postTime } from './socials-calendar-model'

export const QUEUE_STATUS = 'needs_approval'
export const NO_SLOT = 'No slot'
const MINUTE_MS = 60_000
const HOUR_MIN = 60
const DAY_MIN = 24 * HOUR_MIN

export function slotOfQueued(post: Pick<SocialPost, 'planned_for' | 'scheduled_for'>): string | null {
  return post.planned_for ?? post.scheduled_for ?? null
}

/** The posts waiting for approval, the earliest slot first, then the oldest. */
export function queuedPosts(posts: ReadonlyArray<SocialPost>): SocialPost[] {
  const slot = (post: SocialPost) => slotOfQueued(post) ?? '9999'
  return posts
    .filter((post) => post.status === QUEUE_STATUS)
    .sort((a, b) => slot(a).localeCompare(slot(b)) || a.created_at.localeCompare(b.created_at))
}

/** "Wed 14 Oct". */
export function dayLabel(day: Date): string {
  return day.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }).replace(',', '')
}

export interface QueueGroup {
  label: string
  today: boolean
  posts: SocialPost[]
}

/** The queued posts by their slot's day: today first, then the other days in order, then no slot. */
export function queueGroups(posts: ReadonlyArray<SocialPost>, now: Date): QueueGroup[] {
  const byDay = new Map<string, QueueGroup>()
  const unslotted: SocialPost[] = []
  for (const post of queuedPosts(posts)) {
    const slot = slotOfQueued(post)
    if (!slot) {
      unslotted.push(post)
      continue
    }
    const day = new Date(slot)
    const key = day.toDateString()
    const group = byDay.get(key) ?? { label: dayLabel(day), today: key === now.toDateString(), posts: [] }
    byDay.set(key, { ...group, posts: [...group.posts, post] })
  }
  const days = Array.from(byDay.values())
  const ordered = [...days.filter((g) => g.today), ...days.filter((g) => !g.today)]
  return unslotted.length ? [...ordered, { label: NO_SLOT, today: false, posts: unslotted }] : ordered
}

export function queueHeading(todayCount: number): string {
  if (todayCount === 0) return 'All caught up for today'
  return todayCount === 1 ? '1 post needs you today' : `${todayCount} posts need you today`
}

/** "in 4h 40m", "in 25m", "in 2d"; "now" at the slot; "passed" after it. */
export function timeLeft(slot: string, now: Date): string {
  const minutes = Math.round((new Date(slot).getTime() - now.getTime()) / MINUTE_MS)
  if (minutes < 0) return 'passed'
  if (minutes === 0) return 'now'
  if (minutes < HOUR_MIN) return `in ${minutes}m`
  if (minutes < DAY_MIN) return `in ${Math.floor(minutes / HOUR_MIN)}h ${minutes % HOUR_MIN}m`
  return `in ${Math.round(minutes / DAY_MIN)}d`
}

/** "Approve · publishes 12:00", or "Approve" for a post with no slot, or with no channel, which
 * publishes nothing (F256). */
export function approveLabel(post: SocialPost): string {
  const slot = slotOfQueued(post)
  return slot && hasChannels(post) ? `Approve · publishes ${postTime(post, slot)}` : 'Approve'
}

/** "12:00 · X, LinkedIn · Image · WebSummit countdown". */
export function metaLine(post: SocialPost, campaignName: string | null): string {
  const slot = slotOfQueued(post)
  const channels = Array.from(new Set((post.targets ?? []).map((t) => channelLabel(t.toolkit)))).join(', ')
  return [slot ? postTime(post, slot) : null, channels || null, formatLabel(post), campaignName].filter(Boolean).join(' · ')
}
