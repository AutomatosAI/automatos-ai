// @vitest-environment node
/**
 * PRD-261 A-09b — the server's copy in the reducer: it is the truth for every part the user
 * hasn't changed since the read, never over a change made here; it belongs to one scope and one
 * theme; and the first meeting with it loses none of this browser's own looks (mergeLooks).
 */
import { describe, expect, it, vi } from 'vitest'

import { AURORA, DAYLIGHT, EMBER, OBSIDIAN } from '@/lib/looks/presets'
import {
  createLookOutbox,
  INITIAL_LOOK_STATE,
  type LookAction,
  type LookState,
  lookReducer,
  mergeLooks,
  sameLookSet,
} from '@/lib/looks/store'
import type { Look } from '@/lib/looks/types'

import { myLook } from './helpers'

const SCOPE = 'ws-1:user-a'
const PLAY = 'automatos/play'
const HELLO = 'acme/hello'

function run(actions: readonly LookAction[], from: LookState = INITIAL_LOOK_STATE): LookState {
  return actions.reduce(lookReducer, from)
}

const loaded = (themeId = PLAY, active: Look | null = DAYLIGHT, saved: readonly Look[] = []): LookAction => ({
  type: 'loaded',
  themeId,
  scope: SCOPE,
  active,
  saved,
  persisted: true,
})

const served = (active: Look | null, saved: readonly Look[], themeId = PLAY, scope = SCOPE): LookAction => ({
  type: 'served',
  themeId,
  scope,
  active,
  saved,
})

describe("the server's copy", () => {
  it('replaces what this browser held, without marking it for writing', () => {
    const state = run([loaded(), served(OBSIDIAN, [myLook()])])
    expect(state).toMatchObject({ look: OBSIDIAN, saved: [myLook()], themeId: PLAY, lookChanged: false, savedChanged: false })
  })

  it('never replaces a change the user made since the read', () => {
    const lookEdited = run([loaded(), { type: 'set', look: AURORA }, served(OBSIDIAN, [myLook()])])
    expect(lookEdited).toMatchObject({ look: AURORA, saved: [myLook()], lookChanged: true })
    const savedEdited = run([loaded(), { type: 'save', look: myLook({ name: 'Mine here' }) }, served(OBSIDIAN, [])])
    expect(savedEdited).toMatchObject({ look: OBSIDIAN, saved: [myLook({ name: 'Mine here' })], savedChanged: true })
  })

  it('keeps the look when the server has no active one, and the same objects when nothing differs', () => {
    expect(run([loaded(), served(null, [])]).look).toEqual(DAYLIGHT)
    const before = run([loaded(PLAY, DAYLIGHT, [myLook()])])
    const after = lookReducer(before, served({ ...DAYLIGHT }, [{ ...myLook() }]))
    expect(after).toBe(before)
  })

  it("is ignored for another scope, another theme, or before this browser's copy is read", () => {
    const state = run([loaded()])
    expect(lookReducer(state, served(OBSIDIAN, [], PLAY, 'ws-1:user-b'))).toBe(state)
    expect(lookReducer(state, served(OBSIDIAN, [], HELLO))).toBe(state)
    expect(lookReducer(INITIAL_LOOK_STATE, served(OBSIDIAN, []))).toBe(INITIAL_LOOK_STATE)
  })
})

describe('one theme per set', () => {
  it("another theme's read in the same scope replaces the looks; another tab's for another theme is ignored", () => {
    const play = run([loaded(PLAY, DAYLIGHT, [myLook()]), { type: 'set', look: AURORA }])
    const hello = lookReducer(play, loaded(HELLO, null, []))
    expect(hello).toMatchObject({ themeId: HELLO, look: EMBER, saved: [], lookChanged: false })
    const synced: LookAction = { type: 'synced', themeId: PLAY, scope: SCOPE, active: OBSIDIAN, saved: [], persisted: true }
    expect(lookReducer(hello, synced)).toBe(hello)
  })
})

describe('mergeLooks (the first meeting)', () => {
  it("keeps the server's looks first, then this browser's that it lacks", () => {
    const mine = myLook({ id: 'look-7', name: 'Only here' })
    const merged = mergeLooks({ active: OBSIDIAN, saved: [myLook()] }, { active: DAYLIGHT, saved: [myLook(), mine] })
    expect(merged).toEqual({ active: OBSIDIAN, saved: [myLook(), mine] })
  })

  it("gives a local look a new id when the server's look with that id is another one", () => {
    const local = myLook({ name: 'Numbered the same here' })
    const merged = mergeLooks({ active: null, saved: [myLook()] }, { active: DAYLIGHT, saved: [local] })
    expect(merged.active).toEqual(DAYLIGHT)
    expect(merged.saved).toEqual([myLook(), { ...local, id: 'look-1' }])
  })

  it('takes everything this browser kept when the server holds nothing', () => {
    const local = { active: DAYLIGHT, saved: [myLook()] }
    expect(mergeLooks({ active: null, saved: [] }, local)).toEqual(local)
    expect(sameLookSet(mergeLooks({ active: null, saved: [] }, local), local)).toBe(true)
    expect(sameLookSet(local, { active: DAYLIGHT, saved: [] })).toBe(false)
  })
})

describe('createLookOutbox (one save at a time)', () => {
  /** A send the test answers by hand, noting when it started. */
  function heldSend(log: string[], name: string): { readonly send: () => Promise<unknown>; readonly answer: () => void } {
    let resolveIt: () => void = () => undefined
    const send = () => {
      log.push(name)
      return new Promise<void>((resolve) => {
        resolveIt = resolve
      })
    }
    return { send, answer: () => resolveIt() }
  }

  const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

  it('starts a save only once the one before it has its answer', async () => {
    const outbox = createLookOutbox()
    const log: string[] = []
    const first = heldSend(log, 'first')
    const second = heldSend(log, 'second')
    outbox.queue('a', first.send)
    outbox.queue('b', second.send)
    await settle()
    expect(log).toEqual(['first'])
    first.answer()
    await settle()
    expect(log).toEqual(['first', 'second'])
  })

  it('drops a waiting save when a newer one comes for the same key, and keeps going after a failure', async () => {
    const outbox = createLookOutbox()
    const log: string[] = []
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const first = heldSend(log, 'first')
    outbox.queue('k', first.send)
    outbox.queue('k', () => Promise.resolve(log.push('stale')))
    outbox.queue('k', () => Promise.resolve(log.push('newest')))
    await settle()
    first.answer()
    await settle()
    expect(log).toEqual(['first', 'newest'])
    outbox.queue('k', () => Promise.reject(new Error('handler broke')))
    outbox.queue('other', () => Promise.resolve(log.push('after')))
    await settle()
    expect(log).toEqual(['first', 'newest', 'after'])
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})
