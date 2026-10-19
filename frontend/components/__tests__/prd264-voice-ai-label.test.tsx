// PRD-264 EUAI-04 — the voice call shows "AI voice assistant" for the whole call
// (EU AI Act Art. 50(1) and 50(5)). The SDK never runs here: both call surfaces
// render with their transport hook mocked, in each state a call passes through.

import { describe, it, expect, vi, afterEach, beforeAll } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import React from 'react'

const hookState = {
  orbState: 'listening',
  stateLabel: 'Auto is listening',
  levelsRef: { current: { agent: 0, user: 0 } },
  captions: [] as Array<{ role: string; text: string }>,
  durationSec: 42,
  muted: false,
  refusal: null as string | null,
  error: null as string | null,
  isLive: true,
  micSilent: false,
  inputDevices: [] as Array<{ deviceId: string; label: string }>,
  start: vi.fn(async () => {}),
  stop: vi.fn(),
  toggleMute: vi.fn(),
}

vi.mock('@/hooks/use-retell-call', () => ({
  useRetellCall: () => hookState,
}))
vi.mock('@/components/voice/PresenceOrb', () => ({
  PresenceOrb: ({ state }: { state: string }) => <div data-testid="orb" data-orb-state={state} />,
}))

import { AI_VOICE_LABEL } from '@/components/voice/AiVoiceLabel'
import { LiveVoiceMode } from '@/components/voice/LiveVoiceMode'
import { VoiceCallPanel } from '@/components/voice/VoiceCallPanel'

const CALL_STATES = [
  { orbState: 'connecting', stateLabel: 'Connecting…', isLive: false },
  { orbState: 'listening', stateLabel: 'Auto is listening', isLive: true },
  { orbState: 'speaking', stateLabel: 'Auto is speaking', isLive: true },
  { orbState: 'thinking', stateLabel: 'Auto is thinking', isLive: true },
]

function inCallState(state: (typeof CALL_STATES)[number]) {
  Object.assign(hookState, { ...state, refusal: null, error: null, muted: false, captions: [] })
}

function expectTheLabel() {
  const label = screen.getByRole('note', { name: AI_VOICE_LABEL })
  expect(label).toHaveTextContent('AI voice assistant')
  // theme tokens, never the inert `dark:` variant (PRD-264 §8)
  expect(label.className).not.toMatch(/dark:/)
  expect(label.className).toMatch(/text-foreground/)
  return label
}

beforeAll(() => {
  // jsdom has no layout; the panel scrolls its captions into view
  Element.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('PRD-264 EUAI-04 — VoiceCallPanel', () => {
  it.each(CALL_STATES)('shows "AI voice assistant" while $orbState', (state) => {
    inCallState(state)
    render(<VoiceCallPanel workspaceId="ws-1" />)
    expectTheLabel()
  })

  it('keeps the label when the host renames the agent', () => {
    inCallState(CALL_STATES[1])
    render(<VoiceCallPanel workspaceId="ws-1" agentName="Harbour Coffee" />)
    expect(screen.getByText('Harbour Coffee')).toBeTruthy()
    expectTheLabel()
  })
})

describe('PRD-264 EUAI-04 — LiveVoiceMode (the chat screen\'s call strip)', () => {
  it.each(CALL_STATES)('shows "AI voice assistant" while $orbState', (state) => {
    inCallState(state)
    render(<LiveVoiceMode onExit={() => {}} />)
    expectTheLabel()
  })

  it('the call controls are unchanged beside it', () => {
    inCallState(CALL_STATES[1])
    render(<LiveVoiceMode onExit={() => {}} />)
    expectTheLabel()
    expect(screen.getByRole('button', { name: 'Mute microphone' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'End call' })).toBeTruthy()
  })
})
