// Theme voice — the words a live call shows: the exchange in progress off Retell's transcript, and why Live can't
// start in the member's words (the raw-JSON refusal since #1264 included).
import { describe, expect, it } from 'vitest'

import {
  LIVE_REASON_SENTENCES,
  LIVE_UNAVAILABLE,
  liveRefusalSentence,
  liveUnavailableSentence,
} from '@/lib/voice/live/refusal'
import { currentExchange } from '@/lib/voice/live/transcript'

const said = (role: 'user' | 'agent', content: string) => ({ role, content })

describe('currentExchange', () => {
  it("is the member's last words and Auto's answer after them", () => {
    expect(currentExchange([said('user', 'first'), said('agent', 'one'), said('user', 'second'), said('agent', 'two')])).toEqual({
      userText: 'second',
      agentText: 'two',
    })
  })

  it("drops Auto's previous answer while the member asks something new", () => {
    expect(currentExchange([said('user', 'first'), said('agent', 'one'), said('user', 'and then?')])).toEqual({
      userText: 'and then?',
      agentText: '',
    })
  })

  it("is Auto's greeting alone before the member speaks", () => {
    expect(currentExchange([said('agent', 'Hi, this is Auto, your AI assistant.')])).toEqual({
      userText: '',
      agentText: 'Hi, this is Auto, your AI assistant.',
    })
  })

  it('reads nothing from a transcript that is missing or malformed', () => {
    expect(currentExchange(undefined)).toEqual({ userText: '', agentText: '' })
    expect(currentExchange([{ role: 'user', content: 42 }])).toEqual({ userText: '', agentText: '' })
  })
})

describe('liveRefusalSentence', () => {
  const refusal = (body: unknown, message = 'HTTP 403') => Object.assign(new Error(message), { status: 403, body })

  it('turns the coded plan refusal into a sentence, never the JSON apiClient put in the message', () => {
    const detail = { code: 'voice_live_plan_required', message: 'Auto Live is on the Business plan' }
    const sentence = liveRefusalSentence(refusal({ detail }, JSON.stringify(detail)))
    expect(sentence).toBe(LIVE_REASON_SENTENCES.voice_live_plan_required)
    expect(sentence).not.toMatch(/[{}"]/)
  })

  it("uses a coded detail's message when this build doesn't know the code", () => {
    expect(liveRefusalSentence(refusal({ detail: { code: 'voice_new_gate', message: 'Auto Live is resting.' } }))).toBe(
      'Auto Live is resting.',
    )
  })

  it('keeps a plain-sentence detail as the server wrote it', () => {
    expect(liveRefusalSentence(refusal({ detail: 'Auto Live is not enabled for this workspace' }))).toBe(
      'Auto Live is not enabled for this workspace',
    )
  })

  it('says it plainly when there is no body to read (offline, a sign-in error)', () => {
    expect(liveRefusalSentence(new Error('Failed to fetch'))).toBe(LIVE_UNAVAILABLE)
    expect(liveRefusalSentence(refusal({ detail: { code: 42 } }))).toBe(LIVE_UNAVAILABLE)
    expect(liveRefusalSentence(null)).toBe(LIVE_UNAVAILABLE)
  })
})

describe('liveUnavailableSentence', () => {
  it('has words for every reason live-status gives', () => {
    for (const code of [
      'voice_live_plan_required',
      'voice_live_platform_off',
      'voice_live_not_armed',
      'voice_live_workspace_off',
      'voice_live_cap_reached',
    ]) {
      expect(liveUnavailableSentence(code)).toBe(LIVE_REASON_SENTENCES[code])
      expect(liveUnavailableSentence(code)).not.toContain('_')
    }
    expect(liveUnavailableSentence(null)).toBe(LIVE_UNAVAILABLE)
  })
})
