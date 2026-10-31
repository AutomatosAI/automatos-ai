/**
 * PRD-268 §6.11.1, §6.12.2 (US-C21b) — every state's glyph, word and colour, with §6.11.1's
 * corrections to the handoff: done is ✓ (never —, which means stopped) and readiness ready is ✓
 * (never ●, which means in motion). Unknown draws nothing.
 */
import { describe, expect, it } from 'vitest'

import {
  DISPLAY_STATE_LABELS,
  NEXT_STEP_GLYPH,
  READINESS_STATE_LABELS,
  STATUS_ROLE_TOKENS,
  displayStateLabel,
  isDisplayState,
  readinessStateLabel,
} from '@/lib/proactive/display-state'
import { DISPLAY_STATES } from '@/lib/proactive/types'

describe('DISPLAY_STATE_LABELS', () => {
  it.each([
    ['needs_you', '◆', 'Needs you', 'NEEDS YOU', 'needs_you'],
    ['working', '●', 'Working', 'WORKING', 'working'],
    ['ready', '✓', 'Ready to look at', 'READY', 'ready'],
    ['done', '✓', 'Done', 'DONE', 'done'],
    ['failed', '✕', "Couldn't finish", 'FAILED', 'failed'],
  ] as const)('%s is %s %s (HUD %s) in the %s role', (state, glyph, word, hudWord, role) => {
    const label = DISPLAY_STATE_LABELS[state]
    expect(label).toMatchObject({ glyph, word, hudWord, role })
    expect(label.token).toBe(STATUS_ROLE_TOKENS[role])
    expect(displayStateLabel(state)).toBe(label)
    expect(isDisplayState(state)).toBe(true)
  })

  it('labels every state the server sends, and nothing else', () => {
    expect(Object.keys(DISPLAY_STATE_LABELS).sort()).toEqual([...DISPLAY_STATES].sort())
  })

  it('draws done as ✓, never —, which means stopped', () => {
    expect(DISPLAY_STATE_LABELS.done.glyph).toBe('✓')
    const glyphs = Object.values(DISPLAY_STATE_LABELS).map((label) => label.glyph)
    expect(glyphs).not.toContain('—')
  })

  it('gives every label a word beside its glyph, so no state is shown by colour alone', () => {
    for (const label of [...Object.values(DISPLAY_STATE_LABELS), ...Object.values(READINESS_STATE_LABELS)]) {
      expect(label.glyph.trim()).not.toBe('')
      expect(label.word.trim()).not.toBe('')
      expect(label.hudWord).toBe(label.hudWord.toUpperCase())
      expect(label.hudWord.trim()).not.toBe('')
    }
  })

  it('names the ▸ glyph of the next-step line', () => {
    expect(NEXT_STEP_GLYPH).toBe('▸')
  })
})

describe('READINESS_STATE_LABELS', () => {
  it.each([
    ['ready', '✓', 'Ready', 'ready'],
    ['not_checked', '○', 'Not checked', 'not_checked'],
    ['needs_setup', '◆', 'Needs setup', 'needs_setup'],
  ] as const)('%s is %s %s in the %s role', (state, glyph, word, role) => {
    const label = READINESS_STATE_LABELS[state]
    expect(label).toMatchObject({ glyph, word, role })
    expect(readinessStateLabel(state)).toBe(label)
  })

  it('draws readiness ready as ✓, never ●, which means in motion', () => {
    expect(READINESS_STATE_LABELS.ready.glyph).toBe('✓')
    expect(READINESS_STATE_LABELS.ready.glyph).not.toBe(DISPLAY_STATE_LABELS.working.glyph)
  })
})

describe('colour roles (§6.12.2)', () => {
  it('paints needs you in warn-ink on light grounds and warning on dark ones', () => {
    expect(DISPLAY_STATE_LABELS.needs_you.token).toEqual({ light: '--warn-ink', dark: '--warning' })
  })

  it.each([
    ['working', '--running'],
    ['ready', '--success'],
    ['done', '--success'],
    ['failed', '--destructive'],
  ] as const)('paints %s in %s on every ground', (state, token) => {
    expect(DISPLAY_STATE_LABELS[state].token).toEqual({ light: token, dark: token })
  })

  it('paints needs setup in info and not checked in muted-foreground', () => {
    expect(READINESS_STATE_LABELS.needs_setup.token).toEqual({ light: '--info', dark: '--info' })
    expect(READINESS_STATE_LABELS.not_checked.token).toEqual({ light: '--muted-foreground', dark: '--muted-foreground' })
  })

  it('never sends a status through --accent, the error colour under every Play look', () => {
    const tokens = Object.values(STATUS_ROLE_TOKENS).flatMap((token) => [token.light, token.dark])
    expect(tokens).not.toContain('--accent')
  })
})

describe('unknown draws nothing', () => {
  it.each([undefined, null, '', 'paused', 'cancelled', 'unknown', 'NEEDS_YOU', 'constructor', '__proto__', 'toString', 3])(
    '%s has no label',
    (state) => {
      expect(displayStateLabel(state)).toBeNull()
      expect(readinessStateLabel(state)).toBeNull()
      expect(isDisplayState(state)).toBe(false)
    },
  )

  it('cannot be changed by a caller', () => {
    expect(Object.isFrozen(DISPLAY_STATE_LABELS)).toBe(true)
    expect(Object.isFrozen(DISPLAY_STATE_LABELS.done)).toBe(true)
    expect(Object.isFrozen(READINESS_STATE_LABELS)).toBe(true)
    expect(Object.isFrozen(STATUS_ROLE_TOKENS.needs_you)).toBe(true)
  })
})
