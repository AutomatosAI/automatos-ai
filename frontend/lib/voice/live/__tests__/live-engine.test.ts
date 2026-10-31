// Theme voice — Auto Live as a VoiceEngine (lib/voice/live/live-engine.ts): the mint, Retell's events walking the
// phases, the exchange in progress, mute, levels and every failure as a plain sentence. Retell's SDK is a stand-in
// that records what the engine asked of it and emits the events a test names; apiClient is mocked.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn() }))
const retell = vi.hoisted(() => {
  type Handler = (...args: unknown[]) => void
  const clients: FakeRetellClient[] = []
  class FakeRetellClient {
    readonly handlers = new Map<string, Handler[]>()
    readonly startCall = vi.fn(async () => undefined)
    readonly mute = vi.fn()
    readonly unmute = vi.fn()
    // The real SDK emits call_ended from stopCall while connected.
    readonly stopCall = vi.fn(() => this.emit('call_ended'))
    constructor() {
      clients.push(this)
    }
    on(event: string, handler: Handler) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler])
      return this
    }
    emit(event: string, ...args: unknown[]) {
      ;(this.handlers.get(event) ?? []).forEach((handler) => handler(...args))
    }
  }
  return { clients, FakeRetellClient }
})

vi.mock('@/lib/api-client', () => ({ apiClient: { request: api.request } }))
vi.mock('retell-client-js-sdk', () => ({ RetellWebClient: retell.FakeRetellClient }))

import { LIVE_MINT_PATH, LiveEngine } from '@/lib/voice/live/live-engine'
import {
  LIVE_BROWSER_UNSUPPORTED,
  LIVE_CONNECT_FAILED,
  LIVE_MIC_DENIED,
  LIVE_REASON_SENTENCES,
} from '@/lib/voice/live/refusal'

const MINTED = { call_id: 'call_1', access_token: 'tok_1', chat_id: 'chat-1' }
const original = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

function namedError(name: string, message = name) {
  return Object.assign(new Error(message), { name })
}

/** A browser that can capture; the analyser tap finds no device unless a test says otherwise. */
function stubMedia(getUserMedia = vi.fn().mockRejectedValue(namedError('NotFoundError'))) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia, enumerateDevices: vi.fn().mockResolvedValue([]), addEventListener: vi.fn(), removeEventListener: vi.fn() },
  })
}

function engineFor(agentId: number | null = null) {
  return new LiveEngine({ agentId: () => agentId })
}

async function connected(engine: LiveEngine, chatId: string | null = 'chat-1') {
  await engine.start({ chatId })
  const client = retell.clients[retell.clients.length - 1]
  client.emit('call_started')
  return client
}

beforeEach(() => {
  retell.clients.length = 0
  api.request.mockReset()
  api.request.mockResolvedValue(MINTED)
  stubMedia()
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  if (original) Object.defineProperty(navigator, 'mediaDevices', original)
  vi.restoreAllMocks()
})

describe('the Live engine — a call from the mint to its end', () => {
  it('mints for the conversation on screen and the picked agent, connects, and walks Retell\'s events', async () => {
    const engine = engineFor(7)
    const pending = engine.start({ chatId: 'chat-1' })
    expect(engine.snapshot()).toMatchObject({ mode: 'live', phase: 'connecting', micOpen: false })
    await pending

    expect(api.request).toHaveBeenCalledWith(LIVE_MINT_PATH, { method: 'POST', body: { chat_id: 'chat-1', agent_id: 7 } })
    const client = retell.clients[0]
    expect(client.startCall).toHaveBeenCalledWith({ accessToken: 'tok_1', emitRawAudioSamples: true })

    client.emit('call_started')
    expect(engine.snapshot()).toMatchObject({ phase: 'listening', micOpen: true, chatId: 'chat-1', error: null })
    client.emit('agent_start_talking')
    expect(engine.snapshot().phase).toBe('speaking')
    client.emit('update', {
      transcript: [
        { role: 'agent', content: 'Hi, this is Auto, your AI assistant.' },
        { role: 'user', content: 'where did we leave off' },
        { role: 'agent', content: 'We were mid-way through' },
      ],
    })
    expect(engine.snapshot()).toMatchObject({ heard: 'where did we leave off', reply: 'We were mid-way through' })
    client.emit('agent_stop_talking')
    expect(engine.snapshot().phase).toBe('listening')

    client.emit('call_ended')
    expect(engine.snapshot()).toMatchObject({ mode: 'live', phase: 'ended', micOpen: false })
    expect(engine.snapshot().heard).toBe('where did we leave off')
  })

  it('a new chat takes the conversation the mint names', async () => {
    api.request.mockResolvedValue({ ...MINTED, chat_id: 'voice-thread-9' })
    const engine = engineFor()
    await engine.start({ chatId: null })
    expect(api.request).toHaveBeenCalledWith(LIVE_MINT_PATH, { method: 'POST', body: {} })
    expect(engine.snapshot().chatId).toBe('voice-thread-9')
  })

  it('tells subscribers of a change, and not of an event that changes nothing', async () => {
    const engine = engineFor()
    const client = await connected(engine)
    const listener = vi.fn()
    engine.subscribe(listener)
    client.emit('agent_stop_talking') // already listening
    expect(listener).not.toHaveBeenCalled()
    client.emit('agent_start_talking')
    expect(listener).toHaveBeenCalledTimes(1)
    expect(listener.mock.calls[0][0]).toBe(engine.snapshot())
  })

  it('stop() ends in idle: the call goes, and the SDK\'s own call_ended no longer counts', async () => {
    const engine = engineFor()
    const client = await connected(engine)
    engine.stop()
    expect(client.stopCall).toHaveBeenCalled()
    expect(engine.snapshot()).toMatchObject({ mode: null, phase: 'idle', micOpen: false, error: null })
  })

  it('a stop during the mint never connects', async () => {
    let answer: (value: unknown) => void = () => undefined
    api.request.mockReturnValue(new Promise((resolve) => (answer = resolve)))
    const engine = engineFor()
    const pending = engine.start({ chatId: 'chat-1' })
    engine.stop()
    answer(MINTED)
    await pending
    expect(retell.clients).toHaveLength(0)
    expect(engine.snapshot().phase).toBe('idle')
  })

  it('mutes and unmutes the call: the mic is open only while unmuted', async () => {
    const engine = engineFor()
    const client = await connected(engine)
    engine.toggleMute()
    expect(client.mute).toHaveBeenCalled()
    expect(engine.snapshot()).toMatchObject({ muted: true, micOpen: false })
    engine.toggleMute()
    expect(client.unmute).toHaveBeenCalled()
    expect(engine.snapshot()).toMatchObject({ muted: false, micOpen: true })
  })

  it("reads Auto's level off the SDK's raw audio, without telling subscribers", async () => {
    const engine = engineFor()
    const client = await connected(engine)
    const listener = vi.fn()
    engine.subscribe(listener)
    client.emit('audio', new Float32Array(64).fill(0.5))
    expect(engine.levels().agent).toBeCloseTo(0.5)
    expect(listener).not.toHaveBeenCalled()
  })
})

describe('the Live engine — every failure is a plain sentence', () => {
  it('the plan refusal (a coded 403) reads as words, never the raw JSON', async () => {
    const detail = { code: 'voice_live_plan_required', message: "Auto Live isn't on this workspace's plan" }
    api.request.mockRejectedValue(Object.assign(new Error(JSON.stringify(detail)), { status: 403, body: { detail } }))
    const engine = engineFor()
    await engine.start({ chatId: 'chat-1' })

    const { phase, error } = engine.snapshot()
    expect(phase).toBe('error')
    expect(error).toBe(LIVE_REASON_SENTENCES.voice_live_plan_required)
    expect(error).not.toMatch(/[{}"]|voice_live_plan_required/)
    expect(retell.clients).toHaveLength(0)
  })

  it("a gate's honest sentence stays as the server wrote it", async () => {
    const detail = 'Voice budget used: 100/100 min this month'
    api.request.mockRejectedValue(Object.assign(new Error(detail), { status: 429, body: { detail } }))
    const engine = engineFor()
    await engine.start({ chatId: 'chat-1' })
    expect(engine.snapshot().error).toBe(detail)
  })

  it("the SDK's start failure is a sentence, and the call is torn down", async () => {
    const engine = engineFor()
    await engine.start({ chatId: 'chat-1' })
    retell.clients[0].emit('error', 'Error starting call')
    expect(engine.snapshot()).toMatchObject({ phase: 'error', error: LIVE_CONNECT_FAILED })
    expect(retell.clients[0].stopCall).toHaveBeenCalled()
  })

  it('a refused microphone fails the call with the reason, not a silent "Mic live"', async () => {
    stubMedia(vi.fn().mockRejectedValue(namedError('NotAllowedError', 'Permission denied')))
    const engine = engineFor()
    await engine.start({ chatId: 'chat-1' })
    expect(engine.snapshot()).toMatchObject({ phase: 'error', error: LIVE_MIC_DENIED })
  })

  it('a browser that cannot capture is told so before any mint', async () => {
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: undefined })
    const engine = engineFor()
    await engine.start({ chatId: 'chat-1' })
    expect(engine.snapshot()).toMatchObject({ phase: 'error', error: LIVE_BROWSER_UNSUPPORTED })
    expect(api.request).not.toHaveBeenCalled()
  })
})
