/**
 * PRD-251B F252 — the line under a Brand kit AI tools dropdown that offers one choice
 * (brand-ai-tools-model.ts): what to connect for more, from the server's toolkit rows, only
 * the toolkits that make what the media type needs; why a connected toolkit adds nothing;
 * and nothing under a dropdown with a real choice.
 */
import { describe, it, expect } from 'vitest'

import { oneChoiceHint, orList } from '../brand-ai-tools-model'

// c1 on 3 Oct 2026: no paid image or footage tool connected, two voice toolkits connected.
const C1 = {
  toolkits: [
    { toolkit: 'fal_ai', label: 'fal.ai', kind: 'Images and footage', status: 'connect', makes: ['video', 'image'] },
    { toolkit: 'kieai', label: 'Kie.ai', kind: 'Images and footage', status: 'connect', makes: ['video', 'image'] },
    { toolkit: 'higgsfield_mcp', label: 'Higgsfield', kind: 'Images and footage', status: 'connect', makes: ['video', 'image'] },
    { toolkit: 'elevenlabs', label: 'ElevenLabs', kind: 'Voice', status: 'available' },
    { toolkit: 'templates', label: 'Templates', kind: 'Images', status: 'builtin' },
    { toolkit: 'kokoro', label: 'Kokoro', kind: 'Voice', status: 'builtin' },
  ],
  offered: {
    images: [{ value: 'templates', label: 'Templates (free)' }],
    ai_images: [{ value: 'ask', label: 'Ask each time' }],
    footage: [{ value: 'off', label: 'Off' }],
    voice: [{ value: 'kokoro', label: 'Kokoro (free)' }, { value: 'elevenlabs', label: 'ElevenLabs' }],
  },
} as any

describe('the one-choice line (F252)', () => {
  it('names what to connect under each one-choice dropdown, and nothing under a real choice', () => {
    expect(oneChoiceHint('images', C1)).toBe('Only Templates (free) for now: connect fal.ai, Kie.ai or Higgsfield above to make images with AI.')
    expect(oneChoiceHint('ai_images', C1)).toBe('Only Ask each time for now: connect fal.ai, Kie.ai or Higgsfield above to choose an AI image tool.')
    expect(oneChoiceHint('footage', C1)).toBe('Only Off for now: connect fal.ai, Kie.ai or Higgsfield above to make AI footage.')
    expect(oneChoiceHint('voice', C1)).toBeNull()
  })

  it('names only the toolkits that make what the media type needs', () => {
    const tools = {
      ...C1,
      toolkits: [
        { toolkit: 'vid', label: 'Clips', kind: 'Images and footage', status: 'connect', makes: ['video'] },
        { toolkit: 'pic', label: 'Stills', kind: 'Images and footage', status: 'connect', makes: ['image'] },
        { toolkit: 'fish_audio', label: 'Fish Audio', kind: 'Voice', status: 'connect' },
      ],
    }
    expect(oneChoiceHint('images', tools)).toBe('Only Templates (free) for now: connect Stills above to make images with AI.')
    expect(oneChoiceHint('footage', tools)).toBe('Only Off for now: connect Clips above to make AI footage.')
    const voiceOnly = { ...tools, offered: { ...C1.offered, voice: [{ value: 'kokoro', label: 'Kokoro (free)' }] } }
    expect(oneChoiceHint('voice', voiceOnly)).toBe('Only Kokoro (free) for now: connect Fish Audio above to choose another voice.')
  })

  it('says why a connected toolkit adds nothing, and when nothing is set up at all', () => {
    const unavailable = {
      ...C1,
      toolkits: [{ toolkit: 'fal_ai', label: 'fal.ai', kind: 'Images and footage', status: 'unavailable', reason: 'FAL_AI_SUBMIT_ASYNC_JOB is not cached' }],
    }
    expect(oneChoiceHint('images', unavailable)).toBe('Only Templates (free) for now: fal.ai is connected but unavailable (FAL_AI_SUBMIT_ASYNC_JOB is not cached).')
    const none = { ...C1, toolkits: [], offered: { ...C1.offered, voice: [{ value: 'kokoro', label: 'Kokoro (free)' }] } }
    expect(oneChoiceHint('voice', none)).toBe('Only Kokoro (free) for now: no AI tool for this is set up on this platform.')
  })

  it('lists names the way a sentence does', () => {
    expect([orList([]), orList(['a']), orList(['a', 'b']), orList(['a', 'b', 'c'])]).toEqual(['', 'a', 'a or b', 'a, b or c'])
  })
})
