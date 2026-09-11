import { describe, expect, it } from 'vitest'
import { formatAbsoluteDate, formatRelative, isWithinDays } from '../format-relative'

// A fixed "now" so the buckets are deterministic, and every input carries an
// explicit offset — the contract the backend's utc_iso guarantees.
const NOW = Date.parse('2026-09-11T12:00:00+00:00')
const ago = (seconds: number) => new Date(NOW - seconds * 1000).toISOString()

describe('formatRelative', () => {
  it('reads "never" for nothing and for garbage', () => {
    expect(formatRelative(null, NOW)).toBe('never')
    expect(formatRelative(undefined, NOW)).toBe('never')
    expect(formatRelative('not a date', NOW)).toBe('never')
  })

  it('buckets elapsed time the way a console is scanned', () => {
    expect(formatRelative(ago(5), NOW)).toBe('just now')
    expect(formatRelative(ago(59), NOW)).toBe('just now')
    expect(formatRelative(ago(60), NOW)).toBe('1m ago')
    expect(formatRelative(ago(59 * 60), NOW)).toBe('59m ago')
    expect(formatRelative(ago(3600), NOW)).toBe('1h ago')
    expect(formatRelative(ago(23 * 3600), NOW)).toBe('23h ago')
    expect(formatRelative(ago(86_400), NOW)).toBe('1d ago')
    expect(formatRelative(ago(30 * 86_400), NOW)).toBe('30d ago')
  })

  it('falls back to the absolute date once elapsed time stops informing', () => {
    const old = ago(31 * 86_400)
    expect(formatRelative(old, NOW)).toBe(formatAbsoluteDate(old))
    expect(formatRelative(old, NOW)).not.toMatch(/ago$/)
  })

  it('is independent of the viewer timezone when the offset is explicit', () => {
    // The same instant written with two different offsets must read identically.
    const utc = '2026-09-11T11:00:00+00:00'
    const bst = '2026-09-11T12:00:00+01:00'
    expect(formatRelative(utc, NOW)).toBe('1h ago')
    expect(formatRelative(bst, NOW)).toBe('1h ago')
  })
})

describe('isWithinDays', () => {
  it('counts a recent stamp and rejects an old, missing or invalid one', () => {
    expect(isWithinDays(ago(6 * 86_400), 7, NOW)).toBe(true)
    expect(isWithinDays(ago(7 * 86_400), 7, NOW)).toBe(true)
    expect(isWithinDays(ago(7 * 86_400 + 1), 7, NOW)).toBe(false)
    expect(isWithinDays(null, 7, NOW)).toBe(false)
    expect(isWithinDays('nope', 7, NOW)).toBe(false)
  })
})

describe('formatAbsoluteDate', () => {
  it('renders a dash for nothing and a short date otherwise', () => {
    expect(formatAbsoluteDate(null)).toBe('—')
    expect(formatAbsoluteDate('garbage')).toBe('—')
    expect(formatAbsoluteDate('2026-04-28T00:00:00+00:00')).toMatch(/2026/)
  })
})
