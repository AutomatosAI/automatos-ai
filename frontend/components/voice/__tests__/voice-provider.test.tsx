// Theme voice — the host voice provider (components/voice/voice-provider.tsx) over a host with stand-in engines:
// what it offers (live-status and the browser), the engine an `auto` start runs, the session following the
// conversation on screen, the frozen view a surface reads, and nothing at all outside it.
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import React, { type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request } }))

import { VoiceProvider, useHostVoice, useOptionalHostVoice } from '@/components/voice/voice-provider'
import { VoiceHost, type VoiceHostDeps } from '@/lib/voice/host/voice-host'
import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { FakeEngine, FakeLive } from '@/lib/voice/host/__tests__/support/fake-engines'

const speech = window as unknown as Record<string, unknown>

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

function onScreen(chatId: string | null) {
  act(() => {
    if (chatId) useChatSessionStore.getState().openChat(chatId)
    else useChatSessionStore.getState().newDraft()
  })
}

beforeEach(() => {
  api.request.mockReset()
  api.request.mockResolvedValue({ available: true, reason: null })
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn() } })
  speech.webkitSpeechRecognition = function Recognition() {}
  speech.speechSynthesis = {}
  onScreen('chat-1')
})

afterEach(() => {
  cleanup()
  delete speech.webkitSpeechRecognition
  delete speech.speechSynthesis
})

describe('VoiceProvider — what it offers', () => {
  it('offers Live when live-status says so, and quick-ask when the browser can', async () => {
    const { wrapper } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await waitFor(() => expect(result.current.available).toEqual({ quick: true, live: true, reason: null }))
    expect(api.request).toHaveBeenCalledWith('/api/voice/live-status')
  })

  it("says why Live can't start, in the member's words (the plan's code, never JSON)", async () => {
    api.request.mockResolvedValue({ available: false, reason: 'voice_live_plan_required' })
    const { wrapper } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await waitFor(() => expect(result.current.available.reason).toBe(LIVE_REASON_SENTENCES.voice_live_plan_required))
    expect(result.current.available.live).toBe(false)
    expect(result.current.available.reason).not.toMatch(/[{}"_]/)
  })
})

describe('VoiceProvider — a session', () => {
  it('an auto start runs Live when it can start, on the conversation on screen', async () => {
    const { wrapper, live, quick } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await waitFor(() => expect(result.current.available.live).toBe(true))
    await act(() => result.current.start())
    expect(live.start).toHaveBeenCalledWith({ chatId: 'chat-1' })
    expect(quick.start).not.toHaveBeenCalled()
    expect(result.current.snapshot).toMatchObject({ phase: 'connecting', chatId: 'chat-1' })
  })

  it("an auto start runs quick-ask when Live can't start; an asked-for engine runs as asked", async () => {
    api.request.mockResolvedValue({ available: false, reason: 'voice_live_workspace_off' })
    const { wrapper, live, quick } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await waitFor(() => expect(result.current.available.reason).not.toBeNull())
    await act(() => result.current.start())
    expect(quick.start).toHaveBeenCalledWith({ chatId: 'chat-1' })
    await act(() => result.current.start({ mode: 'live' }))
    expect(live.start).toHaveBeenCalled()
  })

  it('moving to another conversation ends the session', async () => {
    const { wrapper, live } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await act(() => result.current.start({ mode: 'live' }))
    act(() => live.emit({ phase: 'listening' }))
    onScreen('chat-2')
    expect(live.stop).toHaveBeenCalledTimes(1)
    expect(result.current.snapshot.phase).toBe('idle')
  })

  it('a new chat takes the conversation the session names, and the session carries on', async () => {
    onScreen(null)
    const { wrapper, live } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await act(() => result.current.start({ mode: 'live' }))
    act(() => live.emit({ phase: 'listening', chatId: 'voice-thread-1' }))
    expect(useChatSessionStore.getState().session.activeChatId).toBe('voice-thread-1')
    expect(live.stop).not.toHaveBeenCalled()
  })

  it('hands the levels getter and the controls through; the view is frozen', async () => {
    const { wrapper, live } = setup()
    const { result } = renderHook(() => useHostVoice(), { wrapper })
    await act(() => result.current.start({ mode: 'live' }))
    expect(result.current.readLevels()).toEqual({ user: 0.2, agent: 0.7 })
    act(() => {
      result.current.live.toggleMute()
      result.current.live.pickDevice('usb-2')
      result.current.stop()
    })
    expect(live.toggleMute).toHaveBeenCalled()
    expect(live.pickDevice).toHaveBeenCalledWith('usb-2')
    expect(live.stop).toHaveBeenCalled()
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(Object.isFrozen(result.current.live)).toBe(true)
  })
})

describe('outside a VoiceProvider', () => {
  it('useHostVoice says where the provider is mounted; useOptionalHostVoice is null', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    function Reader() {
      useHostVoice()
      return null
    }
    expect(() => render(<Reader />)).toThrow(/outside VoiceProvider/)
    const { result } = renderHook(() => useOptionalHostVoice())
    expect(result.current).toBeNull()
  })
})
