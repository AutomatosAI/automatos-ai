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

/** The drag payload's type: only a social item is draggable. */
export const SOCIAL_DRAG_TYPE = 'application/x-automatos-social-post'
export const SNAP_MINUTES = 15
const DEFAULT_TZ = 'UTC'

function zoneOf(item: Pick<ScheduleItem, 'timezone'>): string {
  return item.timezone || DEFAULT_TZ
}

/** "09:00 GMT+1" — the slot's time in the post's own timezone. */
export function socialTimeLabel(item: Pick<ScheduleItem, 'next_run_at' | 'timezone'>): string {
  if (!item.next_run_at) return ''
  const parts = new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: zoneOf(item),
    timeZoneName: 'short',
  }).formatToParts(new Date(item.next_run_at))
  const pick = (type: string) => parts.find((p) => p.type === type)?.value ?? ''
  return `${pick('hour')}:${pick('minute')} ${pick('timeZoneName')}`.trim()
}

/** The slot a drop at `offsetY` px down a day column means, snapped to the quarter hour. */
export function slotFromDrop(day: Date, offsetY: number, startHour: number, hourPx: number): Date {
  const minutes = Math.max(0, Math.round(((offsetY / hourPx) * 60) / SNAP_MINUTES) * SNAP_MINUTES)
  const slot = new Date(day)
  slot.setHours(startHour, 0, 0, 0)
  slot.setMinutes(slot.getMinutes() + minutes)
  return slot
}

function offsetMinutes(instant: Date, tz: string): number {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: tz,
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).formatToParts(instant)
  const n = (type: string) => Number(parts.find((p) => p.type === type)?.value)
  const asUtc = Date.UTC(n('year'), n('month') - 1, n('day'), n('hour'), n('minute'), n('second'))
  return Math.round((asUtc - instant.getTime()) / 60_000)
}

/** A `YYYY-MM-DDTHH:mm` wall time in `tz` as the instant it names (ISO, UTC). */
export function zonedWallToIso(wall: string, tz: string): string {
  const [date, time] = wall.split('T')
  const [y, m, d] = date.split('-').map(Number)
  const [hh, mm] = (time ?? '00:00').split(':').map(Number)
  const guess = Date.UTC(y, m - 1, d, hh, mm)
  const first = guess - offsetMinutes(new Date(guess), tz) * 60_000
  // A second pass settles a wall time near a daylight-saving change.
  const settled = guess - offsetMinutes(new Date(first), tz) * 60_000
  return new Date(settled).toISOString()
}

/** The instant `iso` as a `YYYY-MM-DDTHH:mm` wall time in `tz` (a datetime-local value). */
export function wallInZone(iso: string, tz: string): string {
  const instant = new Date(iso)
  const shifted = new Date(instant.getTime() + offsetMinutes(instant, tz) * 60_000)
  return shifted.toISOString().slice(0, 16)
}

export function isSocialItem(item: Pick<ScheduleItem, 'type' | 'post_id'>): boolean {
  return item.type === 'social' && Boolean(item.post_id)
}
