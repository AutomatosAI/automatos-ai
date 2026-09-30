/**
 * PRD-251 US-307 (D10) — a scheduled social post in the calendar: pure helpers.
 *
 * The grid is laid out in the viewer's own time; a post's slot is SHOWN in the post's
 * own timezone (`social_posts.timezone`, set from the scheduler's browser: there is no
 * workspace timezone). A drop on a day column is a new slot at the quarter hour under
 * the pointer; the reschedule dialog reads a wall time in the post's timezone. Both go
 * to POST /api/socials/posts/{id}/schedule with the post's timezone: the slot moves,
 * the approval stands.
 */
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { slotTimeLabel } from '@/lib/social-time'

export { wallInZone, zonedWallToIso } from '@/lib/social-time'

/** The drag payload's type: only a social item is draggable. */
export const SOCIAL_DRAG_TYPE = 'application/x-automatos-social-post'
export const SNAP_MINUTES = 15

/** "09:00 GMT+1" — the slot's time in the post's own timezone. */
export function socialTimeLabel(item: Pick<ScheduleItem, 'next_run_at' | 'timezone'>): string {
  return slotTimeLabel(item.next_run_at, item.timezone)
}

/** The slot a drop at `offsetY` px down a day column means, snapped to the quarter hour. */
export function slotFromDrop(day: Date, offsetY: number, startHour: number, hourPx: number): Date {
  const minutes = Math.max(0, Math.round(((offsetY / hourPx) * 60) / SNAP_MINUTES) * SNAP_MINUTES)
  const slot = new Date(day)
  slot.setHours(startHour, 0, 0, 0)
  slot.setMinutes(slot.getMinutes() + minutes)
  return slot
}

export function isSocialItem(item: Pick<ScheduleItem, 'type' | 'post_id'>): boolean {
  return item.type === 'social' && Boolean(item.post_id)
}
