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
import {
  MIC_SILENT_AFTER_MS,
  MIC_SILENT_PEAK,
  feedMicLevel,
  initialMicHealth,
} from '@/lib/voice/mic-health'

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
// mic-health — the silent-capture detector (pure, the orb-state discipline)
// ---------------------------------------------------------------------------

describe('PRD-207 — feedMicLevel', () => {
  it('a full window of digital silence raises silent', () => {
    let h = initialMicHealth(1_000)
    h = feedMicLevel(h, 0, 1_060)
    expect(h.silent).toBe(false) // not yet — the window must fill
    h = feedMicLevel(h, 0.0001, 1_000 + MIC_SILENT_AFTER_MS)
    expect(h.silent).toBe(true)
  })

  it('any real signal clears silent instantly and restarts the window', () => {
    let h = initialMicHealth(0)
    h = feedMicLevel(h, 0, MIC_SILENT_AFTER_MS) // silent
    expect(h.silent).toBe(true)
    h = feedMicLevel(h, MIC_SILENT_PEAK * 3, MIC_SILENT_AFTER_MS + 60)
    expect(h.silent).toBe(false)
    expect(h.windowStartAt).toBe(MIC_SILENT_AFTER_MS + 60)
  })

  it('a quiet-but-live mic (noise floor above the threshold) never trips it', () => {
    let h = initialMicHealth(0)
    for (let t = 60; t <= MIC_SILENT_AFTER_MS * 2; t += 60) {
      h = feedMicLevel(h, MIC_SILENT_PEAK * 1.5, t)
    }
    expect(h.silent).toBe(false)
  })

  it('is immutable — feeding never mutates the previous snapshot', () => {
    const first = initialMicHealth(0)
    feedMicLevel(first, 0, 2_000)
    expect(first).toEqual(initialMicHealth(0))
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

import { LiveVoiceMode } from '@/components/voice/LiveVoiceMode'

describe('PRD-207 — LiveVoiceMode', () => {
  it('live: mic-live indicator + named mute and end controls (the wave lives in the chat background, not here)', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, orbState: 'listening',
      stateLabel: 'Auto is listening',
    })
    render(<LiveVoiceMode onExit={() => {}} />)
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

  it('a silent capture raises the honest banner with a way to pick another mic', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, muted: false,
      orbState: 'listening', stateLabel: 'Auto is listening', captions: [],
      micSilent: true,
      inputDevices: [
        { deviceId: 'default', label: 'MacBook Pro Microphone' },
        { deviceId: 'iphone-1', label: "Gerard's iPhone" },
      ],
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    expect(screen.getByTestId('mic-silent-banner')).toBeTruthy()
    expect(screen.getByText(/microphone is silent/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Pick another mic' })).toBeTruthy()
  })

  it('a healthy capture shows no banner; the strip offers the device picker', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, muted: false,
      orbState: 'listening', stateLabel: 'Auto is listening', captions: [],
      micSilent: false,
      inputDevices: [
        { deviceId: 'default', label: 'MacBook Pro Microphone' },
        { deviceId: 'usb-1', label: 'USB Interface' },
      ],
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    expect(screen.queryByTestId('mic-silent-banner')).toBeNull()
    expect(screen.getByRole('button', { name: 'Choose microphone' })).toBeTruthy()
  })

  it('one input device = nothing to choose, no picker chrome', () => {
    Object.assign(hookState, {
      refusal: null, error: null, isLive: true, muted: false,
      orbState: 'listening', stateLabel: 'Auto is listening', captions: [],
      micSilent: false,
      inputDevices: [{ deviceId: 'default', label: 'MacBook Pro Microphone' }],
    })
    render(<LiveVoiceMode onExit={() => {}} />)
    expect(screen.queryByRole('button', { name: 'Choose microphone' })).toBeNull()
  })
})
