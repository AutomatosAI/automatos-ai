/**
 * PRD-251 US-307 (D10) — a coming social post in the calendar: pure helpers.
 *
 * The grid is laid out in the viewer's own time; a post's slot is SHOWN in the post's
 * own timezone (`social_posts.timezone`, set from the scheduler's browser: there is no
 * workspace timezone). A drop on a day column is a new slot at the quarter hour under
 * the pointer; the reschedule dialog reads a wall time in the post's timezone. Both go
 * to PUT /api/socials/posts/{id}/slot with the post's timezone: the slot moves, an
 * approval stands. The feed carries every post still to go out, not only a scheduled
 * one, so the chip says where a post stands until it is scheduled.
 */
import { SOCIAL_STATUS_LABELS } from '@/components/deliverables/socials/socials-status'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { SocialPostStatus } from '@/lib/api-client'
import { slotTimeLabel, wallInZone, zonedWallToIso } from '@/lib/social-time'

export { wallInZone, zonedWallToIso } from '@/lib/social-time'

/** The drag payload's type: only a social item is draggable. */
export const SOCIAL_DRAG_TYPE = 'application/x-automatos-social-post'
export const SNAP_MINUTES = 15

/** "09:00 GMT+1" — the slot's time in the post's own timezone. */
export function socialTimeLabel(item: Pick<ScheduleItem, 'next_run_at' | 'timezone'>): string {
  return slotTimeLabel(item.next_run_at, item.timezone)
}

const SCHEDULED: SocialPostStatus = 'scheduled'
/** A post waiting for approval opens in the Socials Queue, where it is approved. */
export const WAITS_FOR_APPROVAL: SocialPostStatus = 'needs_approval'

/** "09:00 GMT+1 · Needs approval": the slot's time and, until the post is scheduled, where it stands. */
export function socialChipLabel(item: Pick<ScheduleItem, 'next_run_at' | 'timezone' | 'status'>): string {
  const standing = item.status && item.status !== SCHEDULED ? SOCIAL_STATUS_LABELS[item.status as SocialPostStatus] : undefined
  return standing ? `${socialTimeLabel(item)} · ${standing}` : socialTimeLabel(item)
}

/** The slot a drop at `offsetY` px down a day column means, snapped to the quarter hour. */
export function slotFromDrop(day: Date, offsetY: number, startHour: number, hourPx: number): Date {
  const minutes = Math.max(0, Math.round(((offsetY / hourPx) * 60) / SNAP_MINUTES) * SNAP_MINUTES)
  const slot = new Date(day)
  slot.setHours(startHour, 0, 0, 0)
  slot.setMinutes(slot.getMinutes() + minutes)
  return slot
}

const NO_SLOT_TIME = '12:00'

/**
 * A Month-view drop: the post's slot moved to `day` (the calendar date the viewer
 * dropped on), at the time of day it had IN THE POST'S TIMEZONE (midday when it had
 * none), so the post keeps its own time, wherever the viewer is.
 */
export function slotOnDay(day: Date, slot: string | null, tz: string): Date {
  const time = slot ? wallInZone(slot, tz).slice(11, 16) : NO_SLOT_TIME
  const pad = (n: number) => String(n).padStart(2, '0')
  const date = `${day.getFullYear()}-${pad(day.getMonth() + 1)}-${pad(day.getDate())}`
  return new Date(zonedWallToIso(`${date}T${time}`, tz))
}

export function isSocialItem(item: Pick<ScheduleItem, 'type' | 'post_id'>): boolean {
  return item.type === 'social' && Boolean(item.post_id)
}
