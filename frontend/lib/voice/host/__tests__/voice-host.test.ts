// Theme voice — the host's one session (lib/voice/host/voice-host.ts) over stand-in engines: it binds to the
// conversation on screen, a new chat takes the conversation the session names, moving to another conversation ends
// it, the words in progress go where they belong (and only there), and the engines and controls are the ones attached.
import { describe, expect, it, vi } from 'vitest'

import { IDLE_VOICE } from '@/lib/voice/engine'
import { NO_LIVE_DETAILS, VoiceHost, type VoiceHostDeps } from '@/lib/voice/host/voice-host'

import { FakeEngine, FakeLive, LIVE_DETAILS as DETAILS } from './support/fake-engines'

function setup(onScreen: string | null = 'chat-1') {
  const live = new FakeLive()
  const quick = new FakeEngine()
  const screen = { chatId: onScreen }
  const deps = {
    liveEngine: () => live,
    quickEngine: () => quick,
    onScreenChatId: () => screen.chatId,
    adoptChat: vi.fn((chatId: string) => {
      screen.chatId = chatId
    }),
    publishTurn: vi.fn(),
    clearTurn: vi.fn(),
  } satisfies VoiceHostDeps
  return { host: new VoiceHost(deps), live, quick, screen, deps }
}

describe('the voice host — one session, bound to the conversation on screen', () => {
  it('binds a start to the conversation on screen and runs the engine asked for', async () => {
    const { host, live, quick, deps } = setup('chat-1')
    await host.start('live')
    expect(live.start).toHaveBeenCalledWith({ chatId: 'chat-1' })
    expect(quick.start).not.toHaveBeenCalled()
    expect(deps.clearTurn).toHaveBeenCalled()
    expect(host.getState().snapshot).toMatchObject({ phase: 'connecting', chatId: 'chat-1' })
    expect(deps.adoptChat).not.toHaveBeenCalled()
  })

  it('a new chat takes the conversation the session names, once, and keeps its session', async () => {
    const { host, live, deps } = setup(null)
    await host.start('live')
    expect(live.start).toHaveBeenCalledWith({ chatId: null })
    live.emit({ chatId: 'voice-thread-1' })
    live.emit({ phase: 'listening' })
    expect(deps.adoptChat).toHaveBeenCalledTimes(1)
    expect(deps.adoptChat).toHaveBeenCalledWith('voice-thread-1')
    host.screenChanged('voice-thread-1')
    expect(live.stop).not.toHaveBeenCalled()
  })

  it('a session at rest never moves the screen to the conversation it last wrote in', async () => {
    const { host, quick, deps } = setup('chat-2')
    quick.current = Object.freeze({ ...IDLE_VOICE, chatId: 'chat-1' })
    quick.start.mockImplementation(async () => quick.emit({ phase: 'idle', chatId: 'chat-1' }))
    await host.start('quick')
    expect(deps.adoptChat).not.toHaveBeenCalled()
  })

  it('moving to another conversation ends the session; a session at rest is left alone', async () => {
    const { host, live } = setup('chat-1')
    await host.start('live')
    live.emit({ phase: 'listening' })
    host.screenChanged('chat-1')
    expect(live.stop).not.toHaveBeenCalled()
    host.screenChanged('chat-2')
    expect(live.stop).toHaveBeenCalledTimes(1)
    host.screenChanged('chat-3')
    expect(live.stop).toHaveBeenCalledTimes(1)
  })

  it('starting one engine stops the other', async () => {
    const { host, live, quick } = setup()
    await host.start('live')
    await host.start('quick')
    expect(live.stop).toHaveBeenCalled()
    expect(quick.start).toHaveBeenCalledWith({ chatId: 'chat-1' })
  })
})

describe('the voice host — the words and the turns', () => {
  it('sends the words in progress to the conversation they belong to', async () => {
    const { host, live, deps } = setup('chat-1')
    await host.start('live')
    live.emit({ phase: 'speaking', heard: 'where did we leave off', reply: 'We were mid-way' })
    expect(deps.publishTurn).toHaveBeenLastCalledWith({
      chatId: 'chat-1',
      userText: 'where did we leave off',
      agentText: 'We were mid-way',
    })
  })

  it('publishes nothing before the session names a conversation, then the conversation it names', async () => {
    const { host, live, deps } = setup(null)
    await host.start('live')
    live.emit({ phase: 'connecting', heard: '', reply: '' })
    expect(deps.publishTurn).not.toHaveBeenCalled()
    live.emit({ chatId: 'voice-thread-1', phase: 'listening', heard: 'hello' })
    expect(deps.publishTurn).toHaveBeenLastCalledWith({ chatId: 'voice-thread-1', userText: 'hello', agentText: '' })
  })

  it('a new session starts with no words in progress', async () => {
    const { host, deps } = setup('chat-1')
    await host.start('quick')
    await host.start('quick')
    expect(deps.clearTurn).toHaveBeenCalledTimes(2)
  })
})

describe('the voice host — levels, controls and subscribers', () => {
  it('reads the attached engine\'s levels, and silence with none', async () => {
    const { host } = setup()
    expect(host.readLevels()).toEqual({ user: 0, agent: 0 })
    await host.start('live')
    expect(host.readLevels()).toEqual({ user: 0.2, agent: 0.7 })
  })

  it('hands the live call\'s microphone controls to the Live engine, and none to quick-ask', async () => {
    const { host, live } = setup()
    await host.start('live')
    expect(host.getState().live).toBe(DETAILS)
    host.toggleMute()
    host.pickDevice('usb-2')
    expect(live.toggleMute).toHaveBeenCalled()
    expect(live.pickDevice).toHaveBeenCalledWith('usb-2')
    await host.start('quick')
    expect(host.getState().live).toBe(NO_LIVE_DETAILS)
  })

  it('tells subscribers of each change, and release and stop reach the engine', async () => {
    const { host, quick } = setup()
    const listener = vi.fn()
    host.subscribe(listener)
    await host.start('quick')
    expect(listener).toHaveBeenCalled()
    host.release()
    host.stop()
    expect(quick.release).toHaveBeenCalled()
    expect(quick.stop).toHaveBeenCalled()
  })
})
