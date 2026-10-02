/**
 * PRD-251B US-B108 — the Socials calendar's reading of a post (pure): where it sits (its
 * slot), what its chip says (the time in the post's own timezone, the format and a
 * video's length, the channel badges, the status WORD: never colour alone), and how a
 * post becomes an event on the Command Center's grid (calendar-model.ts).
 */
import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { SocialPost, SocialPostStatus } from '@/lib/api-client'
import { occurrence, type CalEvent, type WindowSpan } from '@/components/command-center/calendar-model'
import { wallInZone } from '@/lib/social-time'

/** A week-grid box tall enough for the chip's three lines (the grid's own floor is 38 px). */
export const SOCIAL_EVENT_MIN = 75
export const DEFAULT_TZ = 'UTC'
const SECONDS_PER_MINUTE = 60

export type StatusTone = 'planned' | 'making' | 'review' | 'scheduled' | 'posted' | 'missed' | 'failed'

export interface StatusWord {
  word: string
  tone: StatusTone
}

const WORDS: Partial<Record<SocialPostStatus, StatusWord>> = {
  draft: { word: 'Planned', tone: 'planned' },
  rendering: { word: 'Making', tone: 'making' },
  needs_approval: { word: 'Needs you', tone: 'review' },
  changes_requested: { word: 'Needs you', tone: 'review' },
  approved: { word: 'Approved', tone: 'scheduled' },
  scheduled: { word: 'Scheduled', tone: 'scheduled' },
  publishing: { word: 'Posting', tone: 'making' },
  published: { word: 'Posted', tone: 'posted' },
  partially_published: { word: 'Posted', tone: 'posted' },
  missed: { word: 'Skipped', tone: 'missed' },
  failed: { word: 'Failed', tone: 'failed' },
}

/** The chip's status word and tone; null for a post the calendar leaves out (archived). */
export function statusWord(status: SocialPostStatus): StatusWord | null {
  return WORDS[status] ?? null
}

/** The rail's Status key: the six words and what each means (Main.dc.html). */
export const STATUS_KEY: ReadonlyArray<StatusWord & { meaning: string }> = [
  { word: 'Planned', tone: 'planned', meaning: 'Slot and topic, not made' },
  { word: 'Making', tone: 'making', meaning: 'Auto is making it' },
  { word: 'Needs you', tone: 'review', meaning: 'Waiting for approval' },
  { word: 'Scheduled', tone: 'scheduled', meaning: 'Approved, will publish' },
  { word: 'Posted', tone: 'posted', meaning: 'Published, with receipts' },
  { word: 'Skipped', tone: 'missed', meaning: 'Not approved in time' },
]

const FORMAT_LABELS: Record<string, string> = {
  image: 'Image',
  carousel: 'Carousel',
  video: 'Video',
  fact_card: 'Fact card',
  infographic: 'Infographic',
  text: 'Text',
}

/** 30 → "0:30", 95 → "1:35". */
export function lengthLabel(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds))
  return `${Math.floor(whole / SECONDS_PER_MINUTE)}:${String(whole % SECONDS_PER_MINUTE).padStart(2, '0')}`
}

/** "Image", "Video 0:30" (a video with its chosen length), "Post" before a format is set. */
export function formatLabel(post: Pick<SocialPost, 'format' | 'length_seconds'>): string {
  const base = post.format ? FORMAT_LABELS[post.format] ?? post.format : 'Post'
  return post.format === 'video' && post.length_seconds ? `${base} ${lengthLabel(post.length_seconds)}` : base
}

const BADGES: Record<string, string> = { twitter: 'X', linkedin: 'in', instagram: 'IG', tiktok: 'TT', youtube: 'YT' }

/** The mockup's short channel badge: "X", "in", "IG"; any other toolkit's first two letters. */
export function channelBadge(toolkit: string): string {
  return BADGES[toolkit] ?? toolkit.slice(0, 2).toUpperCase()
}

export function postBadges(post: Pick<SocialPost, 'targets'>): string[] {
  return Array.from(new Set((post.targets ?? []).map((target) => channelBadge(target.toolkit))))
}

function firstPublished(post: Pick<SocialPost, 'targets'>): string | null {
  const times = (post.targets ?? []).map((target) => target.published_at).filter((at): at is string => !!at)
  return times.length ? times.sort()[0] : null
}

const ON_ITS_SCHEDULE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>(['scheduled', 'publishing', 'failed'])

/**
 * The slot a post sits at: a published post at its first publish time, a scheduled one at
 * its schedule, any other at its planned slot (B11), each falling back to the next; null
 * keeps it off the grid.
 */
export function slotOf(post: SocialPost): string | null {
  if (post.status === 'published' || post.status === 'partially_published') {
    return firstPublished(post) ?? post.scheduled_for ?? post.planned_for
  }
  if (ON_ITS_SCHEDULE.has(post.status)) return post.scheduled_for ?? post.planned_for
  return post.planned_for ?? post.scheduled_for
}

/** "09:00": the slot's time in the post's own timezone. */
export function postTime(post: Pick<SocialPost, 'timezone'>, slot: string): string {
  return wallInZone(slot, post.timezone || DEFAULT_TZ).slice(11, 16)
}

/** The Command Center feed's item for a post at `slot`: the grid, the drag and the dialog read it. */
export function postItem(post: SocialPost, slot: string): ScheduleItem {
  return {
    id: `social-${post.id}`,
    name: post.title,
    type: 'social',
    next_run_at: slot,
    frequency: '',
    agent_name: null,
    agent_id: null,
    post_id: post.id,
    timezone: post.timezone || DEFAULT_TZ,
    status: post.status,
  }
}

/** The grid's events: each post with a slot inside the window, at its slot. */
export function postEvents(posts: ReadonlyArray<SocialPost>, span: WindowSpan): CalEvent[] {
  return posts.flatMap((post) => {
    const slot = slotOf(post)
    if (!slot || !statusWord(post.status)) return []
    const at = new Date(slot)
    return at >= span.start && at <= span.end ? [occurrence(postItem(post, slot), at, SOCIAL_EVENT_MIN, false)] : []
  })
}

/** "all", "video", or a channel's toolkit. */
export type ChannelFilter = string
export const ALL_CHANNELS: ChannelFilter = 'all'
export const VIDEO_ONLY: ChannelFilter = 'video'

export function matchesFilter(post: SocialPost, filter: ChannelFilter): boolean {
  if (filter === ALL_CHANNELS) return true
  if (filter === VIDEO_ONLY) return post.format === 'video'
  return (post.targets ?? []).some((target) => target.toolkit === filter)
}

/** The statuses PUT /slot moves (US-B105); a post being made, posted or archived stays put. */
const MOVABLE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>([
  'draft', 'needs_approval', 'changes_requested', 'approved', 'scheduled', 'missed',
])

export function isMovable(post: Pick<SocialPost, 'status'>): boolean {
  return MOVABLE.has(post.status)
}

/** A post waiting for approval opens in the Queue; any other in its own view. */
export function opensInQueue(post: Pick<SocialPost, 'status'>): boolean {
  return post.status === 'needs_approval'
}

export function isOnDay(slot: string, day: Date): boolean {
  return new Date(slot).toDateString() === day.toDateString()
}
