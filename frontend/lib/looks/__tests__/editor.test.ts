// @vitest-environment node
/** PRD-260 — the Looks page's rules: edited or not, Save as new look, Duplicate, rename, delete. */
import { describe, expect, it } from 'vitest'

import { safeGlow } from '@/lib/looks/css'
import {
  SAVED_SUB,
  asDuplicate,
  asNewLook,
  baseLook,
  clampGlow,
  copyName,
  freshLookId,
  glowLabel,
  isEdited,
  isSavedLook,
  newLookName,
  statusLine,
  withValues,
} from '@/lib/looks/editor'
import { AURORA, EMBER, WORKSHOP } from '@/lib/looks/presets'
import { LOOK_ID_PATTERN, LOOK_TEXT_MAX } from '@/lib/looks/types'
import { validateLook } from '@/lib/looks/validate'

import { myLook } from './helpers'

function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const inner of Object.values(value)) deepFreeze(inner)
    Object.freeze(value)
  }
  return value
}

describe('baseLook and isEdited', () => {
  it('finds the preset or saved look the active one was picked from', () => {
    expect(baseLook(withValues(AURORA, { accent: 'rose' }), [])).toBe(AURORA)
    const mine = myLook()
    expect(baseLook({ ...mine, glow: 0 }, [mine])).toBe(mine)
    expect(baseLook(myLook({ id: 'gone' }), [mine])).toBeUndefined()
  })

  it('is edited only when a value differs from that look', () => {
    expect(isEdited(EMBER, [])).toBe(false)
    expect(isEdited(withValues(EMBER, { glow: 0 }), [])).toBe(true)
    expect(isEdited(withValues(EMBER, { accent: 'molten' }), [])).toBe(false)
    expect(isEdited(myLook(), [myLook()])).toBe(false)
    expect(isEdited(myLook({ shell: 'cobalt' }), [myLook()])).toBe(true)
    // A name change alone (a rename) is not an edit of the values.
    expect(isEdited({ ...EMBER, name: 'Mine' }, [])).toBe(false)
  })

  it("can't call a look edited when there is nothing to compare it with", () => {
    expect(isEdited(myLook({ id: 'gone', accent: 'gold' }), [])).toBe(false)
  })
})

describe('statusLine', () => {
  it("names the look, and adds '· edited' once it is changed", () => {
    expect(statusLine(EMBER, [])).toBe('Ember')
    expect(statusLine(withValues(EMBER, { font: 'round' }), [])).toBe('Ember · edited')
    expect(statusLine(myLook({ light: 'ice' }), [myLook()])).toBe('My look 1 · edited')
  })
})

describe('withValues', () => {
  it('returns a new look and leaves the old one alone', () => {
    const frozen = deepFreeze({ ...WORKSHOP })
    const next = withValues(frozen, { ground: 'paper', glow: 3 })
    expect(next).toEqual({ ...WORKSHOP, ground: 'paper', glow: 3 })
    expect(frozen).toEqual(WORKSHOP)
  })
})

describe('the glow', () => {
  it('stays a whole number from 0 to 40', () => {
    expect(clampGlow(24)).toBe(24)
    expect(clampGlow(12.6)).toBe(13)
    expect(clampGlow(-3)).toBe(0)
    expect(clampGlow(99)).toBe(40)
    expect(clampGlow(Number.NaN)).toBe(0)
  })

  it("stops at the cap the Glow slider gives it, the look's safe glow, and never past 40 (M50)", () => {
    const cap = safeGlow('paper', 'molten')
    expect(cap).toBeLessThan(24)
    expect(clampGlow(24, cap)).toBe(cap)
    expect(clampGlow(cap - 1, cap)).toBe(cap - 1)
    expect(clampGlow(-3, cap)).toBe(0)
    expect(clampGlow(99, 60)).toBe(40)
  })

  it('reads Off at 0 and as a percentage otherwise', () => {
    expect(glowLabel(0)).toBe('Off')
    expect(glowLabel(24)).toBe('24%')
    expect(glowLabel(40)).toBe('40%')
  })
})

describe('Save as new look', () => {
  it("names the next one 'My look N', skipping names already taken", () => {
    expect(newLookName([])).toBe('My look 1')
    expect(newLookName([myLook()])).toBe('My look 2')
    expect(newLookName([myLook({ id: 'a', name: 'My look 2' })])).toBe('My look 3')
  })

  it('gives it an id no preset or saved look has', () => {
    expect(freshLookId([])).toBe('look-1')
    expect(freshLookId([myLook({ id: 'look-1' }), myLook({ id: 'look-3' })])).toBe('look-2')
    expect(freshLookId([])).toMatch(LOOK_ID_PATTERN)
  })

  it('keeps the active values under the new id, name and line, and validates', () => {
    const edited = withValues(EMBER, { accent: 'rose', glow: 30 })
    const fresh = asNewLook(edited, [myLook()])
    expect(fresh).toEqual({ ...edited, id: 'look-1', name: 'My look 2', sub: SAVED_SUB })
    expect(validateLook(fresh)).toEqual({ ok: true, look: fresh })
    expect(edited.id).toBe('ember')
  })
})

describe('Duplicate, rename and delete', () => {
  it("counts only the user's own saved looks as renameable and deletable, never a preset", () => {
    expect(isSavedLook(myLook(), [myLook()])).toBe(true)
    expect(isSavedLook(myLook(), [])).toBe(false)
    expect(isSavedLook(EMBER, [])).toBe(false)
    expect(isSavedLook(EMBER, [{ ...EMBER }])).toBe(false)
  })

  it("names a copy '<name> copy', then 'copy 2', skipping names already taken", () => {
    expect(copyName('Ember', [])).toBe('Ember copy')
    expect(copyName('Ember', [myLook({ id: 'a', name: 'Ember copy' })])).toBe('Ember copy 2')
    expect(
      copyName('Ember', [myLook({ id: 'a', name: 'Ember copy' }), myLook({ id: 'b', name: 'Ember copy 2' })]),
    ).toBe('Ember copy 3')
  })

  it("keeps a copy's name within the 40-character limit", () => {
    const long = 'A'.repeat(LOOK_TEXT_MAX)
    const name = copyName(long, [])
    expect(Array.from(name).length).toBeLessThanOrEqual(LOOK_TEXT_MAX)
    expect(name.endsWith(' copy')).toBe(true)
    const taken = Array.from({ length: 12 }, (_, i) => myLook({ id: `c${i}`, name: i === 0 ? name : `${name} ${i + 1}` }))
    expect(Array.from(copyName(long, taken)).length).toBeLessThanOrEqual(LOOK_TEXT_MAX)
  })

  it('duplicates a preset or a saved look under a fresh id, keeping its values, and validates', () => {
    const copy = asDuplicate(AURORA, [myLook()])
    expect(copy).toEqual({ ...AURORA, id: 'look-1', name: 'Aurora copy' })
    expect(validateLook(copy)).toEqual({ ok: true, look: copy })
    expect(asDuplicate(myLook(), [myLook()])).toEqual({ ...myLook(), id: 'look-1', name: 'My look 1 copy' })
  })
})
