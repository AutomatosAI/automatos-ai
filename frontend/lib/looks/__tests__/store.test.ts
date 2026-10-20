// @vitest-environment node
/**
 * PRD-260 — the LookProvider's state as a pure reducer: looks belong to one
 * scope (F5), another tab's change is taken without being written back (M25),
 * and a refused write is remembered (M98).
 */
import { describe, expect, it } from 'vitest'

import { AURORA, DAYLIGHT, EMBER, OBSIDIAN } from '@/lib/looks/presets'
import { INITIAL_LOOK_STATE, type LookAction, type LookState, lookReducer } from '@/lib/theme-host/builtin/play-look-state'
import type { Look } from '@/lib/looks/types'

import { myLook } from './helpers'

function run(actions: readonly LookAction[], from: LookState = INITIAL_LOOK_STATE): LookState {
  return actions.reduce(lookReducer, from)
}

const SCOPE = 'ws-1:user-a'
const OTHER_SCOPE = 'ws-1:user-b'

const loaded = (active: Look | null = DAYLIGHT, saved: readonly Look[] = [myLook()], scope = SCOPE): LookAction => ({
  type: 'loaded',
  scope,
  active,
  saved,
  persisted: true,
})

const synced = (active: Look | null, saved: readonly Look[], scope = SCOPE): LookAction => ({
  type: 'synced',
  scope,
  active,
  saved,
  persisted: true,
})

describe('lookReducer', () => {
  it('starts in Ember with nothing saved and nothing to write', () => {
    expect(INITIAL_LOOK_STATE).toMatchObject({
      look: EMBER,
      saved: [],
      loaded: false,
      scope: null,
      persisted: true,
      lookChanged: false,
      savedChanged: false,
    })
  })

  it('takes what storage holds without marking it for writing', () => {
    const state = run([loaded()])
    expect(state).toEqual({
      look: DAYLIGHT,
      saved: [myLook()],
      loaded: true,
      scope: SCOPE,
      persisted: true,
      lookChanged: false,
      savedChanged: false,
    })
    expect(run([loaded(null, [])]).look).toBe(EMBER)
  })

  it('keeps a change made before storage was read', () => {
    const state = run([{ type: 'set', look: AURORA }, loaded()])
    expect(state.look).toBe(AURORA)
    expect(state.saved).toEqual([myLook()])
  })

  it('switches the active look', () => {
    const state = run([loaded(), { type: 'set', look: AURORA }])
    expect(state.look).toBe(AURORA)
    expect(state.lookChanged).toBe(true)
    expect(state.savedChanged).toBe(false)
  })

  it('adds a new look and replaces one with the same id, but never saves over a preset', () => {
    const second = myLook({ id: 'my-look-2', name: 'My look 2' })
    const edited = myLook({ glow: 3 })
    const state = run([loaded(), { type: 'save', look: second }, { type: 'save', look: edited }])
    expect(state.saved).toEqual([edited, second])
    expect(state.savedChanged).toBe(true)
    const presetSave = run([loaded(), { type: 'save', look: { ...EMBER, glow: 0 } }])
    expect(presetSave.saved).toEqual([myLook()])
    expect(presetSave.savedChanged).toBe(false)
  })

  it("renames a saved look, and the active look too when it's that one", () => {
    const state = run([loaded(myLook()), { type: 'rename', id: 'my-look-1', name: 'Sunset' }])
    expect(state.saved[0]?.name).toBe('Sunset')
    expect(state.look.name).toBe('Sunset')
    expect(state.look.accent).toBe('rose')
    expect(run([loaded(), { type: 'rename', id: 'nope', name: 'Sunset' }])).toEqual(run([loaded()]))
  })

  it('deletes a saved look, and goes back to Ember when it was the active one', () => {
    const other = run([loaded(), { type: 'delete', id: 'my-look-1' }])
    expect(other.saved).toEqual([])
    expect(other.look).toBe(DAYLIGHT)
    const active = run([loaded(myLook()), { type: 'delete', id: 'my-look-1' }])
    expect(active.look).toBe(EMBER)
    expect(active.lookChanged).toBe(true)
  })

  it('never changes the state it was given', () => {
    const before = run([loaded()])
    const copy = structuredClone(before)
    run([{ type: 'save', look: myLook({ id: 'x' }) }, { type: 'rename', id: 'my-look-1', name: 'B' }, { type: 'delete', id: 'x' }], before)
    expect(before).toEqual(copy)
  })

  it("takes another tab's looks whole and marks nothing for writing (M25)", () => {
    const theirs = myLook({ id: 'look-1', name: 'Theirs' })
    const state = run([loaded(), { type: 'set', look: AURORA }, synced(DAYLIGHT, [myLook(), theirs])])
    expect(state).toMatchObject({ look: DAYLIGHT, saved: [myLook(), theirs], lookChanged: false, savedChanged: false })
    expect(run([loaded(), synced(null, [])]).look).toBe(EMBER)
  })

  it("ignores a sync for another scope, and one before storage was read", () => {
    const before = run([loaded()])
    expect(run([synced(AURORA, [], OTHER_SCOPE)], before)).toBe(before)
    expect(run([synced(AURORA, [], SCOPE)])).toBe(INITIAL_LOOK_STATE)
  })

  it("reads another account's looks whole: nothing of the first is kept or written to it (F5)", () => {
    const state = run([loaded(), { type: 'set', look: AURORA }, { type: 'save', look: myLook({ glow: 3 }) }, loaded(null, [], OTHER_SCOPE)])
    expect(state).toEqual({
      look: EMBER,
      saved: [],
      loaded: true,
      scope: OTHER_SCOPE,
      persisted: true,
      lookChanged: false,
      savedChanged: false,
    })
  })

  it('keeps what it holds, unsaved edits included, when the same scope is read again', () => {
    const edited = run([loaded(), { type: 'set', look: AURORA }])
    expect(run([loaded(OBSIDIAN, [], SCOPE)], edited)).toBe(edited)
  })

  it('remembers a refused read or write until storage is read again (M98)', () => {
    const refused = run([loaded(), { type: 'written', ok: false }])
    expect(refused.persisted).toBe(false)
    expect(run([{ type: 'written', ok: true }], refused).persisted).toBe(false)
    expect(run([{ type: 'written', ok: true }], run([loaded()]))).toEqual(run([loaded()]))
    const blocked: LookAction = { type: 'loaded', scope: SCOPE, active: null, saved: [], persisted: false }
    expect(run([blocked]).persisted).toBe(false)
    expect(run([blocked, synced(DAYLIGHT, [])]).persisted).toBe(true)
  })
})
