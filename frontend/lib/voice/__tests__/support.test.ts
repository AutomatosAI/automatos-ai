/**
 * Theme voice — quickAskSupported(): quick-ask needs the browser's speech recognition (either name) for the
 * member AND its speech synthesis for Auto; on the server it is never supported.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

import { IDLE_VOICE } from '@/lib/voice/engine'
import { quickAskSupported } from '@/lib/voice/support'

class FakeRecognition {}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('quickAskSupported', () => {
  it('is true with SpeechRecognition and speechSynthesis', () => {
    vi.stubGlobal('SpeechRecognition', FakeRecognition)
    vi.stubGlobal('speechSynthesis', {})

    expect(quickAskSupported()).toBe(true)
  })

  it('is true with the webkit-prefixed recognition (Safari, older Chrome)', () => {
    vi.stubGlobal('webkitSpeechRecognition', FakeRecognition)
    vi.stubGlobal('speechSynthesis', {})

    expect(quickAskSupported()).toBe(true)
  })

  it('is false without speech recognition (Firefox)', () => {
    vi.stubGlobal('speechSynthesis', {})

    expect(quickAskSupported()).toBe(false)
  })

  it('is false without speech synthesis', () => {
    vi.stubGlobal('webkitSpeechRecognition', FakeRecognition)

    expect(quickAskSupported()).toBe(false)
  })

  it('is false on the server, where there is no window', () => {
    vi.stubGlobal('window', undefined)

    expect(quickAskSupported()).toBe(false)
  })
})

describe('IDLE_VOICE', () => {
  it('is a frozen idle snapshot with no mode and no conversation', () => {
    expect(Object.isFrozen(IDLE_VOICE)).toBe(true)
    expect(IDLE_VOICE).toEqual({
      mode: null, phase: 'idle', micOpen: false, muted: false, heard: '', reply: '', durationSec: 0, error: null,
      chatId: null,
    })
  })
})
