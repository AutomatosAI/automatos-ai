/**
 * Theme voice, web quick-ask — what goes wrong becomes a plain sentence (over quick-ask-harness.ts's fakes): the
 * browser's recognition failures, a browser with no speech, a recognition that will not start, a turn that fails or
 * cannot reach Auto, anything unexpected, and a listener that throws. snapshot.error carries the sentence, the phase
 * is error, the microphone is closed, and the engine never throws to its caller.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: { getBaseUrl: () => '', getAuthHeaders: async () => ({}), request: async () => ({}) },
}))

import { FALLBACK_TURN_ERROR } from '@/lib/chat/errors'
import { QUICK_ASK_ERRORS } from '@/lib/voice/quick-ask/copy'
import type { RecognitionFailure, RecognitionPort } from '@/lib/voice/quick-ask/recognition'

import { ask, flush, quickAskHarness } from './quick-ask-harness'

afterEach(() => {
  vi.restoreAllMocks()
})

const SENTENCE = /^[A-Z].*[.!?]$/

const FAILURES: readonly RecognitionFailure[] = [
  'permission', 'noSpeech', 'noMicrophone', 'unsupported', 'serviceOff', 'recognitionNetwork', 'recognizer',
]

describe('the browser’s recognition fails', () => {
  it('says each failure as its sentence, closes the microphone and ends the hold', async () => {
    for (const failure of FAILURES) {
      const h = quickAskHarness()
      await h.engine.start({ chatId: 'c-1' })
      const hold = h.recognition.last()
      hold.capture()

      hold.fail(failure)
      hold.end()
      await flush()

      expect(h.engine.snapshot(), failure).toMatchObject({
        phase: 'error', error: QUICK_ASK_ERRORS[failure], micOpen: false,
      })
      expect(h.engine.snapshot().error, failure).toMatch(SENTENCE)
      expect(hold.abort, failure).toHaveBeenCalled()
      expect(h.chat.sendTurn, failure).not.toHaveBeenCalled()
    }
  })

  it('a browser without speech recognition, or without speech synthesis, says voice is not available here', async () => {
    const noRecognition = quickAskHarness({ recognition: () => null })
    const noSynthesis = quickAskHarness({ synthesis: null })

    await noRecognition.engine.start({ chatId: 'c-1' })
    await noSynthesis.engine.start({ chatId: 'c-1' })

    expect(noRecognition.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.unsupported })
    expect(noSynthesis.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.unsupported })
    expect(noSynthesis.recognition.factory).not.toHaveBeenCalled()
  })

  it('a recognition that will not start is the recognizer stopping, never a throw', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const refusing: RecognitionPort = {
      start: () => {
        throw new Error('InvalidStateError')
      },
      stop: vi.fn(),
      abort: vi.fn(),
    }
    const h = quickAskHarness({ recognition: () => refusing })

    await expect(h.engine.start({ chatId: 'c-1' })).resolves.toBeUndefined()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.recognizer })
    expect(warn).toHaveBeenCalled()
  })

  it('anything unexpected is logged and said, never thrown', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const h = quickAskHarness({
      recognition: () => {
        throw new Error('constructor failed')
      },
    })

    await expect(h.engine.start({ chatId: 'c-1' })).resolves.toBeUndefined()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.recognizer })
    expect(logged).toHaveBeenCalled()
  })
})

describe('the turn fails', () => {
  it('a turn that cannot reach Auto says so, and the speech stops', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, 'Hello?')
    turn.handlers.onText('Partly. ')
    h.synthesis.port.cancel.mockClear()

    turn.settle({ kind: 'failed', reason: 'network', message: null })
    await flush()

    expect(h.synthesis.port.cancel).toHaveBeenCalled()
    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.chatNetwork })
  })

  it('a failed turn says its own sentence, or the chat’s fallback when it has none', async () => {
    const h = quickAskHarness()
    const first = await ask(h, 'One?')
    first.settle({ kind: 'failed', reason: 'turn', message: 'The model stopped answering.' })
    await flush()
    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: 'The model stopped answering.' })

    const second = await ask(h, 'Two?')
    second.settle({ kind: 'failed', reason: 'turn', message: null })
    await flush()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: FALLBACK_TURN_ERROR })
    expect(h.announceTurn).toHaveBeenCalledWith('c-1')
  })

  it('a send that throws is a turn that cannot reach Auto', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const h = quickAskHarness({
      sendTurn: async () => {
        throw new Error('boom')
      },
    })
    await h.engine.start({ chatId: 'c-1' })
    const hold = h.recognition.last()
    hold.capture()
    hold.say('Hello?')
    h.engine.release()
    hold.end()
    await flush()

    expect(h.engine.snapshot()).toMatchObject({ phase: 'error', error: QUICK_ASK_ERRORS.chatNetwork })
  })

  it('the next press clears the error', async () => {
    const h = quickAskHarness()
    const turn = await ask(h, 'One?')
    turn.settle({ kind: 'failed', reason: 'network', message: null })
    await flush()

    await h.engine.start({ chatId: 'c-1' })

    expect(h.engine.snapshot()).toMatchObject({ phase: 'connecting', error: null, heard: '', reply: '' })
  })
})

describe('a listener that throws', () => {
  it('is logged, and the engine and the other listeners go on', async () => {
    const logged = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const h = quickAskHarness()
    h.engine.subscribe(() => {
      throw new Error('a theme bug')
    })

    await h.engine.start({ chatId: 'c-1' })
    h.recognition.last().capture()

    expect(logged).toHaveBeenCalled()
    expect(h.phases).toEqual(['connecting', 'listening'])
  })
})
