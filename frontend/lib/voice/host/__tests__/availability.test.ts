// Theme voice — what the host can offer (lib/voice/host/availability.ts, lib/voice/support.ts): Live from
// live-status and the browser's capture, quick-ask from the browser's speech, the reason in the member's words, and
// the engine an `auto` start picks.
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request } }))

import {
  LIVE_STATUS_CHECKING,
  LIVE_STATUS_PATH,
  readLiveStatus,
  resolveVoiceMode,
  voiceAvailability,
} from '@/lib/voice/host/availability'
import {
  LIVE_BROWSER_UNSUPPORTED,
  LIVE_REASON_SENTENCES,
  LIVE_STATUS_UNREAD,
} from '@/lib/voice/live/refusal'
import { quickAskSupported } from '@/lib/voice/support'

const speech = window as unknown as Record<string, unknown>

afterEach(() => {
  delete speech.webkitSpeechRecognition
  delete speech.SpeechRecognition
  delete speech.speechSynthesis
  vi.restoreAllMocks()
})

describe('readLiveStatus', () => {
  it('reads available and the reason from live-status', async () => {
    api.request.mockResolvedValue({ available: false, reason: 'voice_live_plan_required', armed: true })
    await expect(readLiveStatus()).resolves.toEqual({ state: 'read', available: false, reason: 'voice_live_plan_required' })
    expect(api.request).toHaveBeenCalledWith(LIVE_STATUS_PATH)
  })

  it('says the read failed instead of throwing', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    api.request.mockRejectedValue(new Error('offline'))
    await expect(readLiveStatus()).resolves.toEqual({ state: 'failed' })
  })
})

describe('voiceAvailability', () => {
  const read = (available: boolean, reason: string | null = null) => ({ state: 'read' as const, available, reason })

  it('offers Live when live-status says so and the browser can capture', () => {
    expect(voiceAvailability(read(true), true, true)).toEqual({ quick: true, live: true, reason: null })
  })

  it("gives the plan reason in the member's words", () => {
    expect(voiceAvailability(read(false, 'voice_live_plan_required'), true, true)).toEqual({
      quick: true,
      live: false,
      reason: LIVE_REASON_SENTENCES.voice_live_plan_required,
    })
  })

  it('names the browser when the workspace may use Live but the browser cannot capture', () => {
    expect(voiceAvailability(read(true), false, false)).toEqual({ quick: false, live: false, reason: LIVE_BROWSER_UNSUPPORTED })
  })

  it('has no reason while checking, and says so when the status could not be read', () => {
    expect(voiceAvailability(LIVE_STATUS_CHECKING, false, true)).toEqual({ quick: false, live: false, reason: null })
    expect(voiceAvailability({ state: 'failed' }, false, true).reason).toBe(LIVE_STATUS_UNREAD)
  })
})

describe('resolveVoiceMode', () => {
  const offer = (live: boolean, quick: boolean) => ({ live, quick, reason: null })

  it('auto: Live when it can start, else quick-ask, else Live (its mint tells the member why)', () => {
    expect(resolveVoiceMode('auto', offer(true, true))).toBe('live')
    expect(resolveVoiceMode('auto', offer(false, true))).toBe('quick')
    expect(resolveVoiceMode('auto', offer(false, false))).toBe('live')
  })

  it('honours the engine asked for', () => {
    expect(resolveVoiceMode('live', offer(false, true))).toBe('live')
    expect(resolveVoiceMode('quick', offer(true, false))).toBe('quick')
  })
})

describe('quickAskSupported', () => {
  it('needs speech recognition (prefixed or not) and speech synthesis', () => {
    expect(quickAskSupported()).toBe(false)
    speech.webkitSpeechRecognition = function Recognition() {}
    expect(quickAskSupported()).toBe(false)
    speech.speechSynthesis = { speak: () => undefined }
    expect(quickAskSupported()).toBe(true)
    delete speech.webkitSpeechRecognition
    speech.SpeechRecognition = function Recognition() {}
    expect(quickAskSupported()).toBe(true)
  })
})
