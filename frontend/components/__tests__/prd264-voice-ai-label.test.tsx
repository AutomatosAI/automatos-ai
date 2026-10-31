// PRD-264 EUAI-04 — the voice call shows "AI voice assistant" for the whole call
// (EU AI Act Art. 50(1) and 50(5)). The SDK never runs here: Studio's call strip
// renders over a stand-in for the host voice provider, in each state a call
// passes through.
//
// The product owner's policy (23 Oct 2026, a deliberate change from the first
// PRD-264 build): the label is a tiny muted caption, never a pill or a badge,
// and the voice control is never fenced on it. VoiceCallPanel, the dead docked
// panel the first build also labelled, is deleted (theme voice).

import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import React from 'react'

const voice = vi.hoisted(() => ({ current: null as unknown }))
vi.mock('@/components/voice/voice-provider', () => ({ useHostVoice: () => voice.current }))

import { AI_VOICE_LABEL } from '@/components/voice/AiVoiceLabel'
import { LiveVoiceMode } from '@/components/voice/LiveVoiceMode'
import { hostVoiceStub } from '@/components/voice/__tests__/support/host-voice-stub'
import type { VoicePhase } from '@/lib/voice/engine'
import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'

const CALL_STATES: ReadonlyArray<{ phase: VoicePhase }> = [
  { phase: 'connecting' },
  { phase: 'listening' },
  { phase: 'speaking' },
  { phase: 'thinking' },
  { phase: 'ended' },
]

function inCallState(phase: VoicePhase, error: string | null = null) {
  voice.current = hostVoiceStub({ snapshot: { phase, error } })
  render(<LiveVoiceMode />)
}

function expectTheLabel() {
  const label = screen.getByRole('note', { name: AI_VOICE_LABEL })
  expect(label).toHaveTextContent('AI voice assistant')
  // theme tokens, never the inert `dark:` variant (PRD-264 §8)
  expect(label.className).not.toMatch(/dark:/)
  expect(label.className).toMatch(/text-muted-foreground/)
  return label
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("PRD-264 EUAI-04 — LiveVoiceMode (the chat screen's call strip)", () => {
  it.each(CALL_STATES)('shows "AI voice assistant" while $phase', ({ phase }) => {
    inCallState(phase)
    expectTheLabel()
  })

  it('is a tiny caption: no pill, no badge, no box of its own', () => {
    inCallState('listening')
    const label = expectTheLabel()
    expect(label.className).not.toMatch(/rounded|border|\bbg-|\bpx-|\bpy-/)
    expect(label.className).toMatch(/text-\[10px\]/)
  })

  it('the call controls are unchanged beside it', () => {
    inCallState('listening')
    expectTheLabel()
    expect(screen.getByRole('button', { name: 'Mute microphone' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'End call' })).toBeTruthy()
  })

  // A Retell/network fault can land mid-call, while Auto is still talking:
  // the failure box replaces the controls, never the label (PRD-264 §8).
  it.each([
    { kind: 'refusal', error: LIVE_REASON_SENTENCES.voice_live_workspace_off },
    { kind: 'mid-call error', error: 'Could not connect the call — check your connection and try again.' },
  ])('keeps "AI voice assistant" on a $kind', ({ error }) => {
    inCallState('error', error)
    expectTheLabel()
    expect(screen.getByText(error)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'End call' })).toBeNull()
  })
})
