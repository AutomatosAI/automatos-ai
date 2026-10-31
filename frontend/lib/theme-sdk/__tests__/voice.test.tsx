/**
 * Theme voice (TV-C5) — SDK area `voice`. useVoice reads the host's voice provider (a real VoiceProvider over a host
 * with stand-in engines) and hands back a frozen, theme-neutral copy of each phase, with stop; outside the provider it
 * answers the idle snapshot with nothing available and the reason, never a throw. readVoiceLevels is a plain getter:
 * zeros while the tab has no voice, the host's levels kept within 0 to 1. VoiceButton is the host's button with the
 * theme's props checked: a prop of the wrong kind throws, naming it.
 */
import { act, cleanup, render, renderHook, screen, waitFor } from '@testing-library/react'
import React, { type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request } }))
const levels = vi.hoisted(() => ({ read: null as null | (() => { user: number; agent: number }) }))
vi.mock('@/lib/voice/host/shared-host', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/voice/host/shared-host')>()
  return { ...actual, readHostVoiceLevels: () => (levels.read ?? actual.readHostVoiceLevels)() }
})

import { VoiceProvider } from '@/components/voice/voice-provider'
import { VOICE_BUTTON_WORDS } from '@/components/voice/voice-button-view'
import { VoiceHost, type VoiceHostDeps } from '@/lib/voice/host/voice-host'
import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'
import type { VoiceSnapshot } from '@/lib/voice/engine'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { FakeEngine, FakeLive } from '@/lib/voice/host/__tests__/support/fake-engines'

import { VoiceButton, readVoiceLevels, useVoice, type VoiceButtonProps } from '../voice'

const PHASES: ReadonlyArray<Partial<VoiceSnapshot>> = [
  { phase: 'connecting' },
  { phase: 'listening', micOpen: true, heard: 'what is on' },
  { phase: 'thinking', micOpen: false, heard: 'what is on today' },
  { phase: 'speaking', reply: 'Three things', durationSec: 4 },
  { phase: 'error', error: 'Nothing was heard. Hold to talk and try again.' },
]

function setup() {
  const live = new FakeLive()
  const quick = new FakeEngine()
  const deps: VoiceHostDeps = {
    liveEngine: () => live,
    quickEngine: () => quick,
    onScreenChatId: () => useChatSessionStore.getState().session.activeChatId,
    adoptChat: (chatId) => useChatSessionStore.getState().chatIdAssigned(chatId),
    publishTurn: vi.fn(),
    clearTurn: vi.fn(),
  }
  const host = new VoiceHost(deps)
  const wrapper = ({ children }: { children: ReactNode }) => <VoiceProvider host={host}>{children}</VoiceProvider>
  return { host, live, quick, wrapper }
}

beforeEach(() => {
  api.request.mockReset()
  api.request.mockResolvedValue({ available: false, reason: 'voice_live_workspace_off' })
  act(() => useChatSessionStore.getState().openChat('chat-1'))
})

afterEach(() => {
  cleanup()
  levels.read = null
  vi.restoreAllMocks()
})

describe('voice.useVoice — outside the provider', () => {
  it('answers the idle snapshot with nothing available and the reason, and never throws', () => {
    const { result } = renderHook(() => useVoice())
    expect(result.current).toMatchObject({
      available: { quick: false, live: false, reason: VOICE_BUTTON_WORDS.unavailable },
      mode: null, phase: 'idle', micOpen: false, muted: false, heard: '', reply: '', durationSec: 0, error: null, chatId: null,
    })
    expect(() => result.current.stop()).not.toThrow()
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(Object.isFrozen(result.current.available)).toBe(true)
  })
})

describe('voice.useVoice — a session', () => {
  it('copies each phase of the session, frozen, with the availability in the member’s words', async () => {
    const { host, quick, wrapper } = setup()
    const { result } = renderHook(() => useVoice(), { wrapper })
    await waitFor(() => expect(result.current.available.reason).toBe(LIVE_REASON_SENTENCES.voice_live_workspace_off))
    await act(() => host.start('quick'))
    expect(result.current).toMatchObject({ phase: 'connecting', chatId: 'chat-1' })
    for (const patch of PHASES) {
      act(() => quick.emit({ mode: 'quick', ...patch }))
      expect(result.current).toMatchObject(patch)
      expect(Object.isFrozen(result.current)).toBe(true)
    }
    expect(result.current.available.live).toBe(false)
  })

  it('hands over only the snapshot’s fields, the availability and stop, never the host', async () => {
    const { host, wrapper } = setup()
    const { result } = renderHook(() => useVoice(), { wrapper })
    await act(() => host.start('live'))
    expect(Object.keys(result.current).sort()).toEqual([
      'available', 'chatId', 'durationSec', 'error', 'heard', 'micOpen', 'mode', 'muted', 'phase', 'reply', 'stop',
    ])
    expect(Object.keys(result.current.available).sort()).toEqual(['live', 'quick', 'reason'])
  })

  it('stop ends the session', async () => {
    const { host, live, wrapper } = setup()
    const { result } = renderHook(() => useVoice(), { wrapper })
    await act(() => host.start('live'))
    act(() => live.emit({ phase: 'listening', micOpen: true }))
    act(() => result.current.stop())
    expect(live.stop).toHaveBeenCalledTimes(1)
    expect(result.current.phase).toBe('idle')
  })
})

describe('voice.readVoiceLevels', () => {
  it('answers zeros while the tab has no voice, without making any', () => {
    expect(readVoiceLevels()).toEqual({ user: 0, agent: 0 })
  })

  it('reads the host’s levels, kept within 0 to 1, frozen', () => {
    levels.read = () => ({ user: 0.4, agent: 1.7 })
    const read = readVoiceLevels()
    expect(read).toEqual({ user: 0.4, agent: 1 })
    expect(Object.isFrozen(read)).toBe(true)
    levels.read = () => ({ user: -0.2, agent: Number.NaN })
    expect(readVoiceLevels()).toEqual({ user: 0, agent: 0 })
  })
})

describe('voice.VoiceButton — the theme’s props, checked', () => {
  it('draws the host’s button with the theme’s props', () => {
    const { wrapper } = setup()
    render(<VoiceButton press="toggle" size="sm" label="Ask Auto" conversation="chat-1" />, { wrapper })
    const press = screen.getByRole('button', { name: 'Ask Auto' })
    const root = press.closest<HTMLElement>('[data-voice-button="root"]')
    expect({ ...root?.dataset }).toMatchObject({ voicePress: 'toggle', voiceSize: 'sm', voiceState: 'idle' })
  })

  it.each<[string, Record<string, unknown>]>([
    ['mode', { mode: 'shout' }],
    ['press', { press: 'tap' }],
    ['size', { size: 'xl' }],
    ['conversation', { conversation: 42 }],
    ['hotkey', { hotkey: '' }],
    ['label', { label: null }],
  ])('throws naming %s when it is of the wrong kind', (prop, props) => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    expect(() => render(<VoiceButton {...(props as VoiceButtonProps)} />)).toThrow(
      new RegExp(`^Automatos theme SDK: VoiceButton needs ${prop} `),
    )
  })
})
