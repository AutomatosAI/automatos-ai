// Theme voice — the tab's one host wired to the app (lib/voice/host/shared-host.ts): the conversation on screen and
// a new chat's adoption go through the chat session store, the words go to the live-turns store and settle on the
// window's chat_changed, the levels read off the host, and a session outlives its provider only for the handover.
// The quick-ask bridge (quick-engine.ts) runs the stand-in while this lane builds alone: a start fails in words.
import { afterEach, describe, expect, it, vi } from 'vitest'

const session = vi.hoisted(() => ({
  state: { session: { activeChatId: 'chat-1' as string | null }, chatIdAssigned: vi.fn() },
}))
vi.mock('@/stores/chat-session-store', () => ({ useChatSessionStore: { getState: () => session.state } }))
vi.mock('@/lib/api-client', () => ({ apiClient: { request: vi.fn() } }))

import { VoiceHost } from '@/lib/voice/host/voice-host'
import {
  VOICE_HANDOVER_MS,
  appVoiceHostDeps,
  holdVoiceHost,
  readHostVoiceLevels,
  sharedVoiceHost,
} from '@/lib/voice/host/shared-host'
import { quickAskEngine } from '@/lib/voice/host/quick-engine'
import { QUICK_ASK_NOT_BUILT } from '@/lib/voice/host/quick-ask-stand-in'
import { CHAT_CHANGED_EVENT, useLiveTurnStore } from '@/lib/voice/live-turns'

import { FakeEngine, FakeLive } from './support/fake-engines'

afterEach(() => {
  vi.useRealTimers()
  useLiveTurnStore.getState().clear()
})

function hostOver(quick = new FakeEngine()) {
  const live = new FakeLive()
  return { host: new VoiceHost({ ...appVoiceHostDeps(), liveEngine: () => live, quickEngine: () => quick }), live, quick }
}

describe('the app voice host — its ties to the app', () => {
  it('reads the conversation on screen from the session store and adopts a new chat through it', async () => {
    const { host, live } = hostOver()
    await host.start('live')
    expect(live.start).toHaveBeenCalledWith({ chatId: 'chat-1' })
    live.emit({ chatId: 'voice-thread-1', phase: 'listening' })
    expect(session.state.chatIdAssigned).toHaveBeenCalledWith('voice-thread-1')
  })

  it("publishes the session's words, and the tab's host settles them when their saved rows land", async () => {
    sharedVoiceHost()
    const { host, live } = hostOver()
    await host.start('live')
    live.emit({ phase: 'speaking', heard: 'hello', reply: 'Hi there' })
    expect(useLiveTurnStore.getState().turn).toEqual({ chatId: 'chat-1', userText: 'hello', agentText: 'Hi there' })
    window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: 'chat-1' } }))
    expect(useLiveTurnStore.getState().saved).toEqual({ chatId: 'chat-1', userText: 'hello' })
  })

  it('the tab has one host, and its levels read off it', () => {
    expect(sharedVoiceHost()).toBe(sharedVoiceHost())
    expect(readHostVoiceLevels()).toEqual({ user: 0, agent: 0 })
  })
})

describe('the app voice host — the handover between providers', () => {
  it('a session outlives its last provider only for the handover', () => {
    vi.useFakeTimers()
    const { host } = hostOver()
    const stop = vi.spyOn(host, 'stop')
    const release = holdVoiceHost(host)
    release()
    vi.advanceTimersByTime(VOICE_HANDOVER_MS - 1)
    expect(stop).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1)
    expect(stop).toHaveBeenCalledTimes(1)
  })

  it('the next provider mounting within the handover keeps the session', () => {
    vi.useFakeTimers()
    const { host } = hostOver()
    const stop = vi.spyOn(host, 'stop')
    const releaseChat = holdVoiceHost(host)
    releaseChat()
    const releaseTheme = holdVoiceHost(host)
    vi.advanceTimersByTime(VOICE_HANDOVER_MS * 2)
    expect(stop).not.toHaveBeenCalled()
    releaseTheme()
    vi.advanceTimersByTime(VOICE_HANDOVER_MS)
    expect(stop).toHaveBeenCalledTimes(1)
  })
})

describe('the quick-ask bridge, while this lane builds alone', () => {
  it('is one engine per tab, and its start fails in words without listening', async () => {
    const engine = quickAskEngine()
    expect(quickAskEngine()).toBe(engine)
    await engine.start({ chatId: 'chat-1' })
    expect(engine.snapshot()).toMatchObject({ mode: 'quick', phase: 'error', micOpen: false, error: QUICK_ASK_NOT_BUILT })
    engine.stop()
    expect(engine.snapshot().phase).toBe('idle')
  })
})
