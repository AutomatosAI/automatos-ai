/**
 * PRD-261 M-03 — the scan report's preset line for a theme painted from its looks. The contrast check
 * (services/themes/mobile_contrast.py) measures the prepaint table's defaults first, as id
 * `boot/prepaint.json#defaults` and kind `defaults`; they are no preset, so the line leaves them out
 * and keeps a preset that is itself named `defaults` (a valid look id).
 */
import { describe, expect, it } from 'vitest'

import { scanChecksOf } from '../scan-report'

const subject = (id: string, kind: string) => ({ id, kind, status: 'pass', missing: [], pairs: [] })

const REPORT = {
  schema: 'automatos-theme-scan/v2',
  verdict: 'pass',
  checks: {
    looks: { level: 'pass', status: 'pass', presets: ['defaults', 'night'] },
    contrast: {
      source: 'fromLooks',
      status: 'pass',
      level: 'pass',
      problems: [],
      presets: [
        subject('boot/prepaint.json#defaults', 'defaults'),
        subject('defaults', 'preset'),
        subject('night', 'preset'),
      ],
    },
  },
}

function detailOf(key: string): string | null | undefined {
  return scanChecksOf(REPORT).find((check) => check.key === key)?.detail
}

describe('PRD-261 M-03 — the scan report lists presets, never the table defaults', () => {
  it('leaves the table defaults out of the contrast line and keeps a preset named defaults', () => {
    expect(detailOf('contrast')).toBe('Presets: defaults, night')
  })

  it('lists the looks check presets as they are', () => {
    expect(detailOf('looks')).toBe('Presets: defaults, night')
  })
})
