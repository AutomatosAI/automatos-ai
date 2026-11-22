/**
 * PRD-251B US-B111 — the Queue's reading of the posts (pure): what waits for approval, by
 * the day of its slot (today first, then later days, then no slot), the heading, when each
 * publishes and how long is left, and the pane's meta line.
 *
 * PRD-251C US-C204: a weekly or monthly plan's batch waiting for approval is one section,
 * "Week of 19 Oct · Countdown · 7 posts", its posts in slot order, approved in one sitting.
 *
 * P256-FIX-SOC-2 (F433): a plan's draft the render refused waits for the owner's input
 * (the server's `service.waits_for_input`): listed apart as "Waits for you", never approved here.
 *
 * An approved post with no time posts nothing. A post's own Approve publishes one with no slot at
 * once (`approveLabel`, `post_now`), but a post unscheduled, approved in bulk (its plan's week, its
 * series, Approve all shown) or from the phone app, can still stand approved with neither slot "for
 * a person to schedule", and the calendar's grids,
 * which need a slot, leave it out. Listed apart as "Approved, no time yet" and counted in the nav's
 * Queue badge.
 *
 * One Approve (17 Nov 2026): Approve's label says what the server will do with the post (`approveLabel`).
 */
import type { SocialPost } from '@/lib/api-client'
import { hasChannels } from '../socials-review'
import { channelLabel, channelNames } from '../socials-status'
import { formatLabel, postTime } from './socials-calendar-model'

export const QUEUE_STATUS = 'needs_approval'
/** The status of a post approved and not scheduled (modules/socials/service.py APPROVED). */
export const APPROVED_STATUS = 'approved'
/** The review_log action of a post that waits for the owner's input (modules/socials/service.py ACTION_NEEDS_INPUT). */
export const NEEDS_INPUT = 'needs_input'
export const NO_SLOT = 'No slot'
/** Approve's three labels (approveLabel): it approves only, it posts at the slot, it posts now. */
export const APPROVE = 'Approve'
export const APPROVE_POSTS_AT = 'Approve · posts'
export const APPROVE_AND_POST_NOW = 'Approve & post now to'
const MINUTE_MS = 60_000
const DAY_MS = 86_400_000
const WEEK_KEY = /^(\d{4})-W(\d{2})$/
const MONTH_KEY = /^(\d{4})-(\d{2})$/
const WEEKDAY_INDEX: Record<string, number> = { mon: 0, tue: 1, wed: 2, thu: 3, fri: 4, sat: 5, sun: 6 }
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

/** Whether the post is a draft waiting for the owner's input: its last history entry asks for it. */
export function waitsForInput(post: Pick<SocialPost, 'status' | 'review_log'>): boolean {
  const log = post.review_log ?? []
  return post.status === 'draft' && log.length > 0 && log[log.length - 1].action === NEEDS_INPUT
}

/** The drafts waiting for the owner's input, the earliest slot first, then the oldest. */
export function waitingForInput(posts: ReadonlyArray<SocialPost>): SocialPost[] {
  const slot = (post: SocialPost) => slotOfQueued(post) ?? '9999'
  return posts.filter(waitsForInput).sort((a, b) => slot(a).localeCompare(slot(b)) || a.created_at.localeCompare(b.created_at))
}

/** Whether the post is approved with no time (neither `planned_for` nor `scheduled_for`): nothing posts it
 * until a person schedules it or publishes it now. A missing field counts as no time. */
export function approvedWithNoTime(post: Pick<SocialPost, 'status' | 'planned_for' | 'scheduled_for'>): boolean {
  return post.status === APPROVED_STATUS && slotOfQueued(post) === null
}

/** The approved posts with no time, the oldest first. */
export function waitingForATime(posts: ReadonlyArray<SocialPost>): SocialPost[] {
  return posts.filter(approvedWithNoTime).sort((a, b) => a.created_at.localeCompare(b.created_at))
}

/** The nav's Queue badge: the posts waiting for approval and the approved posts with no time. */
export function queueCount(posts: ReadonlyArray<SocialPost>): number {
  return queuedPosts(posts).length + waitingForATime(posts).length
}

/** What a waiting draft asks for: the fields its last history entry names, else why. */
export function inputAsked(post: Pick<SocialPost, 'review_log'>): string {
  const log = post.review_log ?? []
  const last = log[log.length - 1]
  const fields = (last?.fields ?? []).filter(Boolean)
  return fields.length ? `Fill in: ${fields.join(', ')}` : last?.comment ?? ''
}

/** "Wed 14 Oct". */
export function dayLabel(day: Date): string {
  return day.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }).replace(',', '')
}

export interface QueueGroup {
  label: string
  today: boolean
  posts: SocialPost[]
  /** A stable key when the label may repeat (a batch). */
  id?: string
}

/** PRD-251C (C2): a plan's batch waiting for approval, reviewed as one section. */
export interface QueueBatch extends QueueGroup {
  planId: string
  batchKey: string
}

export function inBatch(post: Pick<SocialPost, 'campaign_id' | 'batch_key'>): boolean {
  return Boolean(post.campaign_id && post.batch_key)
}

export function isMonthBatch(batchKey: string): boolean {
  return MONTH_KEY.test(batchKey)
}

/** A batch's first day (UTC midnight): a week's ISO Monday moved to the day after the plan's
 * batch day; a month's first. Null for a key it cannot read. */
export function batchStart(batchKey: string, batchDay = 'sun'): Date | null {
  const week = WEEK_KEY.exec(batchKey)
  if (week) {
    const jan4 = new Date(Date.UTC(Number(week[1]), 0, 4))
    const monday = jan4.getTime() - ((jan4.getUTCDay() + 6) % 7) * DAY_MS + (Number(week[2]) - 1) * 7 * DAY_MS
    return new Date(monday + (((WEEKDAY_INDEX[batchDay] ?? 6) + 1) % 7) * DAY_MS)
  }
  const month = MONTH_KEY.exec(batchKey)
  return month ? new Date(Date.UTC(Number(month[1]), Number(month[2]) - 1, 1)) : null
}

/** "Week of 19 Oct · Countdown · 7 posts", "November 2026 · Countdown · 31 posts". */
export function batchLabel(batchKey: string, batchDay: string | undefined, planName: string | null, count: number): string {
  const start = batchStart(batchKey, batchDay)
  const short = (options: Intl.DateTimeFormatOptions) => start?.toLocaleDateString('en-GB', { ...options, timeZone: 'UTC' })
  const when = !start ? batchKey : isMonthBatch(batchKey) ? short({ month: 'long', year: 'numeric' }) : `Week of ${short({ day: 'numeric', month: 'short' })}`
  return [when, planName, `${count} ${count === 1 ? 'post' : 'posts'}`].filter(Boolean).join(' · ')
}

/** The plans' batches waiting for approval: one group each, its posts in slot order, the earliest first. */
export function queueBatches(
  posts: ReadonlyArray<SocialPost>, plans: ReadonlyArray<{ id: string; name: string; make?: { batch_day?: string } }>,
): QueueBatch[] {
  const byBatch = new Map<string, SocialPost[]>()
  for (const post of queuedPosts(posts).filter(inBatch)) {
    const id = `${post.campaign_id}|${post.batch_key}`
    byBatch.set(id, [...(byBatch.get(id) ?? []), post])
  }
  return Array.from(byBatch, ([id, batch]) => {
    const [planId, batchKey] = id.split('|')
    const plan = plans.find((candidate) => candidate.id === planId)
    return { id, planId, batchKey, today: false, posts: batch, label: batchLabel(batchKey, plan?.make?.batch_day, plan?.name ?? null, batch.length) }
  })
}

/** How many queued posts have their slot today, in a batch or not. */
export function waitingToday(posts: ReadonlyArray<SocialPost>, now: Date): number {
  return queuedPosts(posts).filter((post) => {
    const slot = slotOfQueued(post)
    return slot !== null && new Date(slot).toDateString() === now.toDateString()
  }).length
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

/** The time approving the post schedules it into, as `service.approve` picks it: its planned slot
 * while still ahead, else a schedule kept from before an edit while still ahead; null when neither is. */
export function approvalSlot(post: Pick<SocialPost, 'planned_for' | 'scheduled_for'>, now: Date): string | null {
  const ahead = (slot: string | null | undefined): slot is string => typeof slot === 'string' && new Date(slot).getTime() > now.getTime()
  return [post.planned_for, post.scheduled_for].find(ahead) ?? null
}

/** Whether Approve also posts it now, as its label says: it has channels and neither a planned slot
 * nor a schedule. The click sends it (`post_now`), and the server publishes only if it agrees then,
 * so a label gone stale (a slot cleared in another tab) never publishes. */
export function approvePostsNow(post: SocialPost): boolean {
  return hasChannels(post) && !post.planned_for && !post.scheduled_for
}

/** What Approve does, said on its button (one Approve, 17 Nov 2026), as the server decides it
 * (`service.approve`, then `api/socials_publish.publish_on_approval`):
 * - no planned slot and no schedule (`approvePostsNow`): "Approve & post now to X, LinkedIn", which
 *   sends `post_now` and publishes it at once;
 * - a slot still ahead (`approvalSlot`: the planned one, else a schedule kept from before an edit):
 *   "Approve · posts 12:00", which schedules it there;
 * - a slot or a kept schedule that passed, or no channel (F256): plain "Approve", which approves it
 *   and posts nothing until a person gives it a time. */
export function approveLabel(post: SocialPost, now: Date = new Date()): string {
  if (approvePostsNow(post)) return `${APPROVE_AND_POST_NOW} ${channelNames(post)}`
  const slot = hasChannels(post) ? approvalSlot(post, now) : null
  return slot ? `${APPROVE_POSTS_AT} ${postTime(post, slot)}` : APPROVE
}

/** "12:00 · X, LinkedIn · Image · WebSummit countdown". */
export function metaLine(post: SocialPost, campaignName: string | null): string {
  const slot = slotOfQueued(post)
  const channels = Array.from(new Set((post.targets ?? []).map((t) => channelLabel(t.toolkit)))).join(', ')
  return [slot ? postTime(post, slot) : null, channels || null, formatLabel(post), campaignName].filter(Boolean).join(' · ')
}
