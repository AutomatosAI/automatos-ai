/**
 * PRD-251 Wave 3 — a social post's slot in its own timezone, without a date library.
 *
 * A slot is stored in UTC and shown in the post's own timezone (`social_posts.timezone`,
 * set from the scheduler's browser: there is no workspace timezone). Used by the
 * Command Center calendar (US-307) and the post view's publish controls (US-308).
 */

const DEFAULT_TZ = 'UTC'

function parts(iso: string, tz: string, options: Intl.DateTimeFormatOptions): Intl.DateTimeFormatPart[] {
  return new Intl.DateTimeFormat('en-GB', { ...options, timeZone: tz || DEFAULT_TZ, timeZoneName: 'short' }).formatToParts(
    new Date(iso),
  )
}

function pick(list: Intl.DateTimeFormatPart[], type: string): string {
  return list.find((p) => p.type === type)?.value ?? ''
}

/** "18:00 GMT+9": the slot's time in `tz`. */
export function slotTimeLabel(iso: string | null | undefined, tz: string | null | undefined): string {
  if (!iso) return ''
  const p = parts(iso, tz || DEFAULT_TZ, { hour: '2-digit', minute: '2-digit', hour12: false })
  return `${pick(p, 'hour')}:${pick(p, 'minute')} ${pick(p, 'timeZoneName')}`.trim()
}

/** "Mon 9 Nov 2026, 18:00 GMT+9": the slot's date and time in `tz`. */
export function slotDateLabel(iso: string | null | undefined, tz: string | null | undefined): string {
  if (!iso) return ''
  const p = parts(iso, tz || DEFAULT_TZ, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
  return `${pick(p, 'weekday')} ${pick(p, 'day')} ${pick(p, 'month')} ${pick(p, 'year')}, ${slotTimeLabel(iso, tz)}`
}

function offsetMinutes(instant: Date, tz: string): number {
  const p = new Intl.DateTimeFormat('en-US', {
    timeZone: tz,
    hourCycle: 'h23',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).formatToParts(instant)
  const n = (type: string) => Number(pick(p, type))
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
  return new Date(guess - offsetMinutes(new Date(first), tz) * 60_000).toISOString()
}

/** The instant `iso` as a `YYYY-MM-DDTHH:mm` wall time in `tz` (a datetime-local value). */
export function wallInZone(iso: string, tz: string): string {
  const instant = new Date(iso)
  return new Date(instant.getTime() + offsetMinutes(instant, tz) * 60_000).toISOString().slice(0, 16)
}

/** The browser's own timezone: the one a person schedules in (US-308). */
export function browserTimezone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || DEFAULT_TZ
}
