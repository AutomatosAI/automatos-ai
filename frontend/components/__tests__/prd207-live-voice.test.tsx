// PRD-207 S5 — the presence orb's pure state machine + the live-mode UX
// states. The SDK never runs here: the reducer is pure, and LiveVoiceMode is
// rendered with its transport hook mocked.

import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import React from 'react'

import {
  THINKING_AFTER_MS,
  initialOrb,
  orbReducer,
  rmsLevel,
  type OrbSnapshot,
} from '@/lib/voice/orb-state'

afterEach(() => {
  cleanup()
  vi.resetAllMocks()
})

describe('PRD-207 — rmsLevel', () => {
  it('is 0 for silence and clamps to 1', () => {
    expect(rmsLevel(new Float32Array(256))).toBe(0)
    expect(rmsLevel(new Float32Array(64).fill(4))).toBe(1)
  })

  it('grows with amplitude', () => {
    const quiet = rmsLevel(new Float32Array(64).fill(0.05))
    const loud = rmsLevel(new Float32Array(64).fill(0.5))
    expect(loud).toBeGreaterThan(quiet)
  })
})

describe('PRD-207 — orbReducer', () => {
  const live = (over: Partial<OrbSnapshot> = {}): OrbSnapshot => ({
    ...initialOrb,
    state: 'listening',
    ...over,
  })

  it('walks connect → started → listening', () => {
    let s = orbReducer(initialOrb, { type: 'call_connecting' })
    expect(s.state).toBe('connecting')
    s = orbReducer(s, { type: 'call_started' })
    expect(s.state).toBe('listening')
  })

  it('agent talking wins: speaking on start, derived on stop', () => {
    let s = orbReducer(live(), { type: 'agent_start_talking' })
    expect(s.state).toBe('speaking')
    s = orbReducer(s, { type: 'agent_stop_talking', now: 10_000 })
    expect(s.state).toBe('listening')
  })

  it('dead air after the user speaks becomes THINKING, never silent limbo', () => {
    const t0 = 50_000
    let s = orbReducer(live(), { type: 'user_voice', now: t0 })
    expect(s.state).toBe('listening') // still their floor
    s = orbReducer(s, { type: 'tick', now: t0 + THINKING_AFTER_MS + 50 })
    expect(s.state).toBe('thinking') // Auto is visibly working
    // Auto's audio arriving flips to speaking
    s = orbReducer(s, { type: 'agent_start_talking' })
    expect(s.state).toBe('speaking')
  })

  it('after Auto answers, quiet is listening again (not thinking)', () => {
    const t0 = 80_000
    let s = orbReducer(live(), { type: 'user_voice', now: t0 })
    s = orbReducer(s, { type: 'agent_start_talking' })
    s = orbReducer(s, { type: 'agent_stop_talking', now: t0 + 4_000 })
    s = orbReducer(s, { type: 'tick', now: t0 + 6_000 })
    expect(s.state).toBe('listening')
  })

  it('terminal states ignore late audio events', () => {
    let s = orbReducer(live(), { type: 'call_ended' })
    s = orbReducer(s, { type: 'agent_start_talking' })
    expect(s.state).toBe('ended')
  })
})

// ---------------------------------------------------------------------------
// LiveVoiceMode — every failure state has designed copy; controls are named
// ---------------------------------------------------------------------------

const hookState = {
  orbState: 'listening',
  stateLabel: 'Auto is listening',
  levelsRef: { current: { agent: 0, user: 0 } },
  captions: [] as Array<{ role: string; text: string }>,
  durationSec: 5,
  muted: false,
  refusal: null as string | null,
  error: null as string | null,
  isLive: true,
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

import { LiveVoiceMode } from '@/components/voice/LiveVoiceMode'

describe('PRD-207 — LiveVoiceMode', () => {
  it('live: orb + mic-live indicator + named mute and end controls', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, orbState: 'listening',
      stateLabel: 'Auto is listening',
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    expect(screen.getByTestId('orb')).toBeTruthy()
    expect(screen.getByText(/Mic live/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Mute microphone' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'End call' })).toBeTruthy()
  })

  it('a refused mint shows the server\'s honest reason and a way out', () => {
    Object.assign(hookState, {
      refusal: 'Voice budget used: 100/100 min this month',
      isLive: false,
      orbState: 'error',
      stateLabel: 'Voice connection error',
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    expect(screen.getByText(/100\/100 min/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy()
  })

  it('renders NO caption lines — the thread is the one source of words', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, orbState: 'speaking',
      stateLabel: 'Auto is speaking',
      captions: [
        { role: 'user', text: 'where did we leave off' },
        { role: 'agent', text: 'We were mid-way through the social pack.' },
      ],
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    // the words belong to the chat's live-typing bubbles, not a second
    // caption strip under the wave (Gerard: "why am I seeing two sources")
    expect(screen.queryByText(/where did we leave off/)).toBeNull()
    expect(screen.queryByText(/mid-way through the social pack/)).toBeNull()
  })
})
