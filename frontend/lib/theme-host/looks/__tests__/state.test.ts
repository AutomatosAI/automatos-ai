// @vitest-environment node
/**
 * PRD-261 A-19d1 — ported from lib/looks/__tests__/store.test.ts to the generic look runtime, on
 * hello-a09b's data: the runtime's state as a pure reducer. The looks belong to one scope (PRD-260
 * F5) and one theme, another tab's change is taken without being written back (M25), a refused write
 * is remembered (M98), and the theme's presets are never saved over; deleting the active look goes
 * back to the theme's default look.
 */
import { describe, expect, it } from 'vitest'

import type { ThemeLook } from '@/lib/theme-host/looks/spec'
import { initialLookState, type LookAction, type LookState, lookReducer } from '@/lib/theme-host/looks/state'

import { DUSK, HELLO_RULES, HELLO_THEME_ID, INK, myLook, PAPER } from './hello-looks'

const SCOPE = 'ws-1:user-a'
const OTHER_SCOPE = 'ws-1:user-b'
const INITIAL = initialLookState(HELLO_RULES)

function run(actions: readonly LookAction[], from: LookState = INITIAL): LookState {
  return actions.reduce(lookReducer, from)
}

const loaded = (active: ThemeLook | null = DUSK, saved: readonly ThemeLook[] = [myLook()], scope = SCOPE): LookAction => ({
  type: 'loaded',
  themeId: HELLO_THEME_ID,
  rules: HELLO_RULES,
  scope,
  active,
  saved,
  persisted: true,
})

const synced = (active: ThemeLook | null, saved: readonly ThemeLook[], scope = SCOPE): LookAction => ({
  type: 'synced',
  themeId: HELLO_THEME_ID,
  rules: HELLO_RULES,
  scope,
  active,
  saved,
  persisted: true,
})

describe('lookReducer', () => {
  it("starts on the theme's default look with nothing saved and nothing to write", () => {
    expect(INITIAL).toMatchObject({
      look: PAPER,
      saved: [],
      loaded: false,
      scope: null,
      themeId: null,
      persisted: true,
      lookChanged: false,
      savedChanged: false,
    })
    expect(initialLookState().look).toBeNull()
  })

  it('takes what storage holds without marking it for writing', () => {
    const state = run([loaded()])
    expect(state).toEqual({
      look: DUSK,
      saved: [myLook()],
      loaded: true,
      scope: SCOPE,
      themeId: HELLO_THEME_ID,
      rules: HELLO_RULES,
      persisted: true,
      lookChanged: false,
      savedChanged: false,
    })
    expect(run([loaded(null, [])]).look).toBe(PAPER)
    expect(run([loaded(null, [])], initialLookState()).look).toBe(PAPER)
  })

  it('keeps a change made before storage was read', () => {
    const state = run([{ type: 'set', look: INK }, loaded()])
    expect(state.look).toBe(INK)
    expect(state.saved).toEqual([myLook()])
  })

  it('switches the active look', () => {
    const state = run([loaded(), { type: 'set', look: INK }])
    expect(state.look).toBe(INK)
    expect(state.lookChanged).toBe(true)
    expect(state.savedChanged).toBe(false)
  })

  it('adds a new look and replaces one with the same id, but never saves over a preset', () => {
    const second = myLook({ id: 'my-look-2', name: 'My look 2' })
    const edited = myLook({ glow: 3 })
    const state = run([loaded(), { type: 'save', look: second }, { type: 'save', look: edited }])
    expect(state.saved).toEqual([edited, second])
    expect(state.savedChanged).toBe(true)
    const presetSave = run([loaded(), { type: 'save', look: { ...PAPER, glow: 0 } }])
    expect(presetSave.saved).toEqual([myLook()])
    expect(presetSave.savedChanged).toBe(false)
  })

  it("renames a saved look, and the active look too when it's that one; never a preset", () => {
    const state = run([loaded(myLook()), { type: 'rename', id: 'my-look-1', name: 'Sunset' }])
    expect(state.saved[0]?.name).toBe('Sunset')
    expect(state.look?.name).toBe('Sunset')
    expect(state.look?.accent).toBe('green')
    expect(run([loaded(), { type: 'rename', id: 'nope', name: 'Sunset' }])).toEqual(run([loaded()]))
    expect(run([loaded(PAPER), { type: 'rename', id: 'paper', name: 'Sunset' }])).toEqual(run([loaded(PAPER)]))
  })

  it("deletes a saved look, and goes back to the theme's default when it was the active one; never a preset", () => {
    const other = run([loaded(), { type: 'delete', id: 'my-look-1' }])
    expect(other.saved).toEqual([])
    expect(other.look).toBe(DUSK)
    const active = run([loaded(myLook()), { type: 'delete', id: 'my-look-1' }])
    expect(active.look).toBe(PAPER)
    expect(active.lookChanged).toBe(true)
    const before = run([loaded(DUSK)])
    expect(lookReducer(before, { type: 'delete', id: 'dusk' })).toBe(before)
  })

  it('never changes the state it was given', () => {
    const before = run([loaded()])
    const copy = structuredClone(before)
    run([{ type: 'save', look: myLook({ id: 'x' }) }, { type: 'rename', id: 'my-look-1', name: 'B' }, { type: 'delete', id: 'x' }], before)
    expect(before).toEqual(copy)
  })

  it("takes another tab's looks whole and marks nothing for writing (M25)", () => {
    const theirs = myLook({ id: 'look-1', name: 'Theirs' })
    const state = run([loaded(), { type: 'set', look: INK }, synced(DUSK, [myLook(), theirs])])
    expect(state).toMatchObject({ look: DUSK, saved: [myLook(), theirs], lookChanged: false, savedChanged: false })
    expect(run([loaded(), synced(null, [])]).look).toBe(PAPER)
  })

  it('ignores a sync for another scope, and one before storage was read', () => {
    const before = run([loaded()])
    expect(run([synced(INK, [], OTHER_SCOPE)], before)).toBe(before)
    expect(run([synced(INK, [], SCOPE)])).toBe(INITIAL)
  })

  it("reads another account's looks whole: nothing of the first is kept or written to it (F5)", () => {
    const state = run([loaded(), { type: 'set', look: INK }, { type: 'save', look: myLook({ glow: 3 }) }, loaded(null, [], OTHER_SCOPE)])
    expect(state).toMatchObject({ look: PAPER, saved: [], loaded: true, scope: OTHER_SCOPE, lookChanged: false, savedChanged: false })
  })

  it('keeps what it holds, unsaved edits included, when the same scope is read again', () => {
    const edited = run([loaded(), { type: 'set', look: INK }])
    expect(run([loaded(PAPER, [], SCOPE)], edited)).toBe(edited)
  })

  it('remembers a refused read or write until storage is read again (M98)', () => {
    const refused = run([loaded(), { type: 'written', ok: false }])
    expect(refused.persisted).toBe(false)
    expect(run([{ type: 'written', ok: true }], refused).persisted).toBe(false)
    expect(run([{ type: 'written', ok: true }], run([loaded()]))).toEqual(run([loaded()]))
    const blocked: LookAction = { type: 'loaded', themeId: HELLO_THEME_ID, rules: HELLO_RULES, scope: SCOPE, active: null, saved: [], persisted: false }
    expect(run([blocked]).persisted).toBe(false)
    expect(run([blocked, synced(DUSK, [])]).persisted).toBe(true)
  })

  it('forgets what it read when no theme with looks runs any more, so the next read is a first read', () => {
    const edited = run([loaded(), { type: 'set', look: INK }])
    const reset = lookReducer(edited, { type: 'reset' })
    expect(reset).toEqual(initialLookState(HELLO_RULES))
    expect(run([loaded(PAPER, [])], reset)).toMatchObject({ look: PAPER, saved: [], loaded: true, lookChanged: false })
    expect(lookReducer(INITIAL, { type: 'reset' })).toBe(INITIAL)
  })
})
