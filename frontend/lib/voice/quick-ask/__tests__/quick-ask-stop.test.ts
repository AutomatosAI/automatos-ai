/**
 * Theme voice, web quick-ask — stopping (over quick-ask-harness.ts's fakes): Stop cancels the recognition, the speech
 * and the reply still streaming (on the server too, through the chat's own Stop); a press while Auto speaks stops it
 * at once (barge-in), and "Auto, stop" said as the whole hold then sends nothing; a hidden tab or an unloading page
 * stops the microphone and the speech, the turn finishing into the chat unheard; dispose ends the engine.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: { getBaseUrl: () => '', getAuthHeaders: async () => ({}), request: async () => ({}) },
}))

import { QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'

import { ask, flush, GREETING, quickAskHarness, type Harness } from './quick-ask-harness'

afterEach(() => {
  vi.restoreAllMocks()
})

const ASKED = "What's on today?"

/** Auto speaks the first sentence of a reply that still streams; the press's own cancel is forgotten. */
async function speaking(h: Harness) {
  const turn = await ask(h, ASKED)
  turn.handlers.onText('Two meetings. One at ten.')
  h.synthesis.start(0)
  h.synthesis.port.cancel.mockClear()
  return turn
}

describe('Stop', () => {
  it('while listening: the recognition aborted, the microphone closed, and a late end sends nothing', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('What is')

    h.engine.stop()
    expect(hold.abort).toHaveBeenCalledTimes(1)
    expect(h.engine.snapshot()).toMatchObject({ phase: 'idle', micOpen: false })
    hold.end()
    await flush()

    expect(h.chat.sendTurn).not.toHaveBeenCalled()
  })

  it('while Auto speaks: the speech cancelled, the reply aborted and cancelled on the server, nothing more said', async () => {
    const h = quickAskHarness()
    const turn = await speaking(h)
    expect(h.engine.snapshot().phase).toBe('speaking')

    h.engine.stop()
    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    expect(turn.signal.aborted).toBe(true)
    await flush()
    expect(h.cancelTurn).toHaveBeenCalledWith('c-1')
    turn.handlers.onText('Two meetings. One at ten. Then lunch. ')
    h.synthesis.end(0)
    turn.settle({ kind: 'aborted' })
    await flush()

    expect(h.synthesis.spoken()).toEqual([GREETING, 'Two meetings.'])
    expect(h.engine.snapshot().phase).toBe('idle')
    expect(h.engine.levels().agent).toBe(0)
  })

  it('is always allowed: it does nothing when idle, and clears an error', async () => {
    const h = quickAskHarness({ recognition: () => null })

    h.engine.stop()
    expect(h.engine.snapshot().phase).toBe('idle')
    await h.engine.start({ chatId: 'c-1' })
    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.unsupported })
    h.engine.stop()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'idle', error: null })
  })
})

describe('barge-in and "Auto, stop"', () => {
  it('a press while Auto speaks stops it at once, and "Auto, stop" said as the whole hold sends nothing', async () => {
    const h = quickAskHarness()
    const turn = await speaking(h)

    await h.engine.start({ chatId: 'c-1' })
    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    expect(turn.signal.aborted).toBe(true)
    const hold = h.recognition.last()
    hold.capture()
    hold.say('Auto, stop.')
    h.engine.release()
    hold.end()
    await flush()

    expect(h.chat.sendTurn).toHaveBeenCalledTimes(1)
    expect(h.cancelTurn).toHaveBeenCalledWith('c-1')
    expect(h.engine.snapshot()).toMatchObject({ phase: 'idle', micOpen: false, error: null })
    turn.handlers.onText('Two meetings. One at ten. Then lunch. ')
    expect(h.synthesis.spoken()).toEqual([GREETING, 'Two meetings.'])
  })

  it('a request that mentions stopping is sent as said', async () => {
    const h = quickAskHarness()

    const turn = await ask(h, 'Stop the deploy')

    expect(turn.request).toMatchObject({ text: 'Stop the deploy', spokenMode: true })
  })

  it('a press once the reply finished streaming stops the speech and cancels nothing on the server', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, ASKED)
    turn.handlers.onText('Done.')
    turn.settle({ kind: 'finished' })
    await flush()
    h.synthesis.start(0)
    h.synthesis.port.cancel.mockClear()

    await h.engine.start({ chatId: 'c-1' })
    await flush()

    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    expect(h.cancelTurn).not.toHaveBeenCalled()
    expect(h.engine.snapshot().phase).toBe('connecting')
  })
})

describe('a hidden tab or an unloading page', () => {
  it('while listening: the microphone closes and nothing is sent', async () => {
    const h = quickAskHarness()
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('Book a')

    h.page.hide()
    expect(hold.abort).toHaveBeenCalled()
    expect(h.engine.snapshot()).toMatchObject({ phase: 'idle', micOpen: false })
    hold.end()
    await flush()

    expect(h.chat.sendTurn).not.toHaveBeenCalled()
  })

  it('while Auto speaks: the speech stops and the turn finishes into the chat unheard', async () => {
    const h = quickAskHarness()
    const turn = await speaking(h)

    h.page.hide()
    await flush()
    turn.handlers.onText('Two meetings. One at ten. Then lunch. ')

    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    expect(turn.signal.aborted).toBe(true)
    expect(h.cancelTurn).not.toHaveBeenCalled()
    expect(h.synthesis.spoken()).toEqual([GREETING, 'Two meetings.'])
    expect(h.engine.snapshot().phase).toBe('idle')
  })

  it('leaves an error on screen for the member to read', async () => {
    const h = quickAskHarness({ recognition: () => null })
    await h.engine.start({ chatId: 'c-1' })

    h.page.hide()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.unsupported })
  })
})

describe('dispose', () => {
  it('stops everything, stops watching the page, and the engine goes quiet', async () => {
    const h = quickAskHarness()
    const turn = await speaking(h)
    const listener = vi.fn()
    h.engine.subscribe(listener)

    h.engine.dispose()
    expect(h.page.unwatch).toHaveBeenCalledTimes(1)
    expect(turn.signal.aborted).toBe(true)
    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    listener.mockClear()
    await h.engine.start({ chatId: 'c-1' })
    h.engine.subscribe(listener)
    h.engine.stop()

    expect(h.recognition.factory).toHaveBeenCalledTimes(1)
    expect(listener).not.toHaveBeenCalled()
  })
})
