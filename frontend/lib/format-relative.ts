/**
 * Elapsed time for "when did this last happen" columns.
 *
 * A console is scanned to spot who is live; "3d ago" answers that at a glance
 * where a timestamp has to be subtracted first. Beyond a month the elapsed form
 * stops being informative, so it falls back to the absolute date. Input must be
 * an ISO string WITH an offset (the backend's `utc_iso`) — an offset-less
 * string is parsed as local time by every browser and the arithmetic goes wrong
 * by the viewer's UTC offset.
 *
 * Extracted 2026-09-11 from app/admin/workspaces/page.tsx so the team page and
 * the admin console share one definition.
 */
export function formatAbsoluteDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

export function formatRelative(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return 'never'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return 'never'
  const seconds = Math.floor((now - then) / 1000)
  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 31) return `${days}d ago`
  return formatAbsoluteDate(iso)
}

/** True when `iso` falls inside the last `days` days (default: a week). */
export function isWithinDays(iso: string | null | undefined, days = 7, now: number = Date.now()): boolean {
  if (!iso) return false
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return false
  return now - then <= days * 86_400_000
}
