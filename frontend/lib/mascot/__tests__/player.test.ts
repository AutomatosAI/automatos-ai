// @vitest-environment node
/**
 * PRD-260 D6 — the player's queue, ported from Automatos Play's
 * tests/mascot/player.test.ts: the clip boundary, urgent swaps, the hold time.
 */
import { describe, expect, it } from 'vitest'

import { atClipBoundary, clipBoundary, jumpTo, type PlayerState, retarget, startPlayer } from '@/lib/mascot/player'
import type { MascotName } from '@/lib/mascot/types'

import { deepFreeze } from './helpers'

const playing = (showing: MascotName, queued: MascotName | null = null): PlayerState =>
  deepFreeze({ showing, queued, finished: false })

describe('atClipBoundary', () => {
  it("is true within 80 ms of the clip's end, or past it", () => {
    expect(atClipBoundary(4.93, 5)).toBe(true)
    expect(atClipBoundary(4.921, 5)).toBe(true)
    expect(atClipBoundary(5, 5)).toBe(true)
    expect(atClipBoundary(5.2, 5)).toBe(true)
  })

  it('is false earlier in the clip', () => {
    expect(atClipBoundary(4.9, 5)).toBe(false)
    expect(atClipBoundary(0, 5)).toBe(false)
    expect(atClipBoundary(2.5, 5)).toBe(false)
  })

  it("is false before the clip's duration is known", () => {
    expect(atClipBoundary(0, Number.NaN)).toBe(false)
    expect(atClipBoundary(0, 0)).toBe(false)
    expect(atClipBoundary(0, Number.POSITIVE_INFINITY)).toBe(false)
    expect(atClipBoundary(Number.NaN, 5)).toBe(false)
  })
})

describe('retarget', () => {
  it('queues a new state for the clip boundary', () => {
    expect(retarget(playing('idle'), 'thinking')).toEqual(playing('idle', 'thinking'))
    expect(retarget(playing('idle', 'thinking'), 'searching')).toEqual(playing('idle', 'searching'))
  })

  it('swaps oops and hello at once', () => {
    expect(retarget(playing('working', 'idle'), 'oops')).toEqual(playing('oops'))
    expect(retarget(playing('sleeping'), 'hello')).toEqual(playing('hello'))
  })

  it('cancels the queue when the target goes back to the clip on screen', () => {
    expect(retarget(playing('thinking', 'searching'), 'thinking')).toEqual(playing('thinking'))
  })

  it('returns the same object when nothing changes', () => {
    const player = playing('idle', 'working')
    expect(retarget(player, 'working')).toBe(player)
    const still = playing('idle')
    expect(retarget(still, 'idle')).toBe(still)
  })

  it('swaps at once after a one-shot has finished', () => {
    const finished = deepFreeze<PlayerState>({ showing: 'celebrate', queued: null, finished: true })
    expect(retarget(finished, 'idle')).toEqual(playing('idle'))
    expect(retarget(finished, 'celebrate')).toBe(finished)
  })
})

describe('clipBoundary', () => {
  it('swaps in the queued state', () => {
    expect(clipBoundary(playing('thinking', 'searching'))).toEqual({ player: playing('searching'), action: 'swap' })
  })

  it('restarts a loop when nothing is queued', () => {
    const player = playing('working')
    expect(clipBoundary(player)).toEqual({ player, action: 'restart' })
    expect(clipBoundary(player).player).toBe(player)
  })

  it('finishes a one-shot once, then does nothing', () => {
    const first = clipBoundary(playing('celebrate'))
    expect(first).toEqual({ player: { showing: 'celebrate', queued: null, finished: true }, action: 'finish' })
    expect(clipBoundary(first.player)).toEqual({ player: first.player, action: 'none' })
  })

  it("lets a queued state cut in after a one-shot's clip", () => {
    expect(clipBoundary(playing('hello', 'listening'))).toEqual({ player: playing('listening'), action: 'swap' })
  })
})

describe('hold time', () => {
  it('never shows a tool call that ends before the clip does', () => {
    // thinking is on screen; a 200 ms search starts and ends; the answer starts.
    const targets: readonly MascotName[] = ['searching', 'thinking', 'idle']
    const afterTool = targets.reduce<PlayerState>((player, target) => retarget(player, target), playing('thinking'))
    expect(afterTool).toEqual(playing('thinking', 'idle'))
    expect(clipBoundary(afterTool).player).toEqual(playing('idle'))
  })

  it('holds a loop for its whole clip: only the boundary swaps it', () => {
    const swapped = clipBoundary(playing('idle', 'working')).player
    const queued = retarget(swapped, 'reading')
    expect(queued.showing).toBe('working')
    expect(clipBoundary(queued).player.showing).toBe('reading')
  })
})

describe('startPlayer and jumpTo', () => {
  it('shows a state with nothing queued', () => {
    expect(startPlayer('hello')).toEqual({ showing: 'hello', queued: null, finished: false })
  })

  it('shows the target at once when no video plays', () => {
    expect(jumpTo(playing('idle', 'working'), 'oops')).toEqual(playing('oops'))
    expect(jumpTo(playing('idle', 'working'), 'idle')).toEqual(playing('idle'))
    const settled = playing('reading')
    expect(jumpTo(settled, 'reading')).toBe(settled)
  })
})
