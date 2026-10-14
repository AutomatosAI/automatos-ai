// @vitest-environment node
/** PRD-260 — the LookProvider's state as a pure reducer. */
import { describe, expect, it } from 'vitest'

import { AURORA, DAYLIGHT, EMBER } from '@/lib/looks/presets'
import { INITIAL_LOOK_STATE, type LookAction, type LookState, lookReducer } from '@/lib/looks/store'

import { myLook } from './helpers'

function run(actions: readonly LookAction[], from: LookState = INITIAL_LOOK_STATE): LookState {
  return actions.reduce(lookReducer, from)
}

const loaded = (active = DAYLIGHT, saved = [myLook()]): LookAction => ({ type: 'loaded', active, saved })

describe('lookReducer', () => {
  it('starts in Ember with nothing saved and nothing to write', () => {
    expect(INITIAL_LOOK_STATE).toMatchObject({ look: EMBER, saved: [], loaded: false, lookChanged: false, savedChanged: false })
  })

  it('takes what storage holds without marking it for writing', () => {
    const state = run([loaded()])
    expect(state).toEqual({ look: DAYLIGHT, saved: [myLook()], loaded: true, lookChanged: false, savedChanged: false })
    expect(run([{ type: 'loaded', active: null, saved: [] }]).look).toBe(EMBER)
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
})
