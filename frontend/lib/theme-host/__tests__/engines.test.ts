/**
 * PRD-261 A-01 — a theme's engines.sdk range: an exact version or a caret range on the host's
 * major, no newer than the host; every other form is refused.
 */
import { describe, expect, it } from 'vitest'

import { satisfiesSdkRange } from '../engines'

const HOST = '1.0.0'

describe('PRD-261 A-01 — satisfiesSdkRange', () => {
  it.each(['^1.0.0', '1.0.0'])('accepts %s on SDK 1.0.0', (range) => {
    expect(satisfiesSdkRange(range, HOST)).toBe(true)
  })

  it.each(['^1.1.0', '^1.0.1', '^2.0.0', '^0.9.0', '1.0.1', '*', 'x', '>=1.0.0', '~1.0.0', '', '1', ' ^1.0.0'])(
    'refuses %j on SDK 1.0.0',
    (range) => {
      expect(satisfiesSdkRange(range, HOST)).toBe(false)
    },
  )

  it('refuses a missing range or one that is not a string', () => {
    expect(satisfiesSdkRange(undefined, HOST)).toBe(false)
    expect(satisfiesSdkRange(1, HOST)).toBe(false)
    expect(satisfiesSdkRange(null, HOST)).toBe(false)
  })

  it('accepts an older minor or patch on the same major', () => {
    expect(satisfiesSdkRange('^1.2.3', '1.4.0')).toBe(true)
    expect(satisfiesSdkRange('^1.4.0', '1.4.2')).toBe(true)
    expect(satisfiesSdkRange('^1.4.3', '1.4.2')).toBe(false)
  })
})
