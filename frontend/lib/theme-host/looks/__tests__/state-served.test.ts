// @vitest-environment node
/**
 * PRD-261 A-19d1 — ported from lib/looks/__tests__/store-served.test.ts to the generic look runtime,
 * on hello-a09b's data: the server's copy in the reducer is the truth for every part the user hasn't
 * changed since the read, never over a change made here; it belongs to one scope and one theme; the
 * first meeting with it loses none of this browser's own looks (mergeLooks); and the store's saves
 * go out one at a time (createLookOutbox).
 */
import { describe, expect, it, vi } from 'vitest'

import type { ThemeLook } from '@/lib/theme-host/looks/spec'
import {
  createLookOutbox,
  initialLookState,
  type LookAction,
  type LookState,
  lookReducer,
  mergeLooks,
  sameLookSet,
} from '@/lib/theme-host/looks/state'

import { DUSK, HELLO_RULES, HELLO_THEME_ID, INK, myLook, PAPER } from './hello-looks'

const SCOPE = 'ws-1:user-a'
const OTHER_THEME = 'acme/hello'
const INITIAL = initialLookState(HELLO_RULES)

function run(actions: readonly LookAction[], from: LookState = INITIAL): LookState {
  return actions.reduce(lookReducer, from)
}

const loaded = (themeId = HELLO_THEME_ID, active: ThemeLook | null = DUSK, saved: readonly ThemeLook[] = []): LookAction => ({
  type: 'loaded',
  themeId,
  rules: HELLO_RULES,
  scope: SCOPE,
  active,
  saved,
  persisted: true,
})

const served = (active: ThemeLook | null, saved: readonly ThemeLook[], themeId = HELLO_THEME_ID, scope = SCOPE): LookAction => ({
  type: 'served',
  themeId,
  scope,
  active,
  saved,
})

describe("the server's copy", () => {
  it('replaces what this browser held, without marking it for writing', () => {
    const state = run([loaded(), served(INK, [myLook()])])
    expect(state).toMatchObject({ look: INK, saved: [myLook()], themeId: HELLO_THEME_ID, lookChanged: false, savedChanged: false })
  })

  it('never replaces a change the user made since the read', () => {
    const lookEdited = run([loaded(), { type: 'set', look: PAPER }, served(INK, [myLook()])])
    expect(lookEdited).toMatchObject({ look: PAPER, saved: [myLook()], lookChanged: true })
    const savedEdited = run([loaded(), { type: 'save', look: myLook({ name: 'Mine here' }) }, served(INK, [])])
    expect(savedEdited).toMatchObject({ look: INK, saved: [myLook({ name: 'Mine here' })], savedChanged: true })
  })

  it('keeps the look when the server has no active one, and the same objects when nothing differs', () => {
    expect(run([loaded(), served(null, [])]).look).toEqual(DUSK)
    const before = run([loaded(HELLO_THEME_ID, DUSK, [myLook()])])
    const after = lookReducer(before, served({ ...DUSK }, [{ ...myLook() }]))
    expect(after).toBe(before)
  })

  it("is ignored for another scope, another theme, or before this browser's copy is read", () => {
    const state = run([loaded()])
    expect(lookReducer(state, served(INK, [], HELLO_THEME_ID, 'ws-1:user-b'))).toBe(state)
    expect(lookReducer(state, served(INK, [], OTHER_THEME))).toBe(state)
    expect(lookReducer(INITIAL, served(INK, []))).toBe(INITIAL)
  })
})

describe('one theme per set', () => {
  it("another theme's read in the same scope replaces the looks; another tab's for another theme is ignored", () => {
    const hello = run([loaded(HELLO_THEME_ID, DUSK, [myLook()]), { type: 'set', look: INK }])
    const other = lookReducer(hello, loaded(OTHER_THEME, null, []))
    expect(other).toMatchObject({ themeId: OTHER_THEME, look: PAPER, saved: [], lookChanged: false })
    const synced: LookAction = {
      type: 'synced', themeId: HELLO_THEME_ID, rules: HELLO_RULES, scope: SCOPE, active: INK, saved: [], persisted: true,
    }
    expect(lookReducer(other, synced)).toBe(other)
  })
})

describe('mergeLooks (the first meeting)', () => {
  it("keeps the server's looks first, then this browser's that it lacks", () => {
    const mine = myLook({ id: 'look-7', name: 'Only here' })
    const merged = mergeLooks({ active: INK, saved: [myLook()] }, { active: DUSK, saved: [myLook(), mine] }, HELLO_RULES)
    expect(merged).toEqual({ active: INK, saved: [myLook(), mine] })
  })

  it("gives a local look a new id when the server's look with that id is another one", () => {
    const local = myLook({ name: 'Numbered the same here' })
    const merged = mergeLooks({ active: null, saved: [myLook()] }, { active: DUSK, saved: [local] }, HELLO_RULES)
    expect(merged.active).toEqual(DUSK)
    expect(merged.saved).toEqual([myLook(), { ...local, id: 'look-1' }])
  })

  it("never gives it a preset's id", () => {
    const rules = { presets: [...HELLO_RULES.presets, { ...PAPER, id: 'look-1' }] }
    const merged = mergeLooks({ active: null, saved: [myLook()] }, { active: null, saved: [myLook({ name: 'Here' })] }, rules)
    expect(merged.saved.map((look) => look.id)).toEqual(['my-look-1', 'look-2'])
  })

  it('takes everything this browser kept when the server holds nothing', () => {
    const local = { active: DUSK, saved: [myLook()] }
    expect(mergeLooks({ active: null, saved: [] }, local, HELLO_RULES)).toEqual(local)
    expect(sameLookSet(mergeLooks({ active: null, saved: [] }, local, HELLO_RULES), local)).toBe(true)
    expect(sameLookSet(local, { active: DUSK, saved: [] })).toBe(false)
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
