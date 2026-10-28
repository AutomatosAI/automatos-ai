/**
 * Theme voice, web quick-ask — Auto's server voice over the browser's own objects: the engine with no deps given
 * (deps.ts, the one the host's voice provider runs for Studio and every theme) unlocks an <audio> element and primes
 * the browser's voice inside the press, asks GET /api/voice/voices for the page language's voice, and speaks the
 * reply's sentences as MP3 from POST /api/voice/speech (one ahead), each played on that element; the browser's voice
 * says nothing while the server speaks. deps.ts keeps one voice choice per page session (module scope), so every test
 * in this file shares the first one's: a test that needs another voice list belongs in its own file.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ request: vi.fn(), requestBlob: vi.fn() }))
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'https://api.automatos.test',
    getAuthHeaders: async () => ({ Authorization: 'Bearer t0k' }),
    request: api.request,
    requestBlob: api.requestBlob,
  },
}))

import { createQuickAskEngine } from '@/lib/voice/quick-ask'

import { GREETING } from './quick-ask-harness'
import { mp3, VOICES } from './server-speech-fakes'

class FakeRecognition {
  static last: FakeRecognition | null = null
  lang = ''
  continuous = false
  interimResults = false
  maxAlternatives = 0
  onaudiostart: (() => void) | null = null
  onresult: ((event: { readonly results: { transcript: string }[][] }) => void) | null = null
  onend: (() => void) | null = null
  start = vi.fn()
  stop = vi.fn()
  abort = vi.fn()

  constructor() {
    FakeRecognition.last = this
  }
}

class FakeUtterance {
  lang = ''
  constructor(readonly text: string) {}
}

/** The <audio> element's stand-in: the test fires its events. */
class FakeAudio extends EventTarget {
  static made: FakeAudio[] = []
  src = ''
  error: MediaError | null = null
  readonly play = vi.fn(async () => undefined)
  readonly pause = vi.fn()
  readonly load = vi.fn()
  readonly removeAttribute = vi.fn()

  constructor() {
    super()
    FakeAudio.made.push(this)
  }

  fire(type: string): void {
    this.dispatchEvent(new Event(type))
  }
}

const urls = { createObjectURL: URL.createObjectURL, revokeObjectURL: URL.revokeObjectURL }

beforeEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  FakeAudio.made = []
  api.request.mockImplementation(async (path: string) => (path === '/api/voice/voices' ? VOICES : {}))
  api.requestBlob.mockImplementation(async (_path: string, init: RequestInit) => mp3(String(init.body)))
  vi.stubGlobal('SpeechRecognition', FakeRecognition)
  vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
  vi.stubGlobal('Audio', FakeAudio)
  let serial = 0
  URL.createObjectURL = vi.fn(() => {
    serial += 1
    return `blob:clip-${serial}`
  })
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  URL.createObjectURL = urls.createObjectURL
  URL.revokeObjectURL = urls.revokeObjectURL
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

/** POST /api/chat's stand-in: a 200 whose body the test streams. */
function installChat() {
  let push: (lines: string) => void = () => undefined
  let close: () => void = () => undefined
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      push = (lines) => controller.enqueue(new TextEncoder().encode(lines))
      close = () => controller.close()
    },
  })
  const fetchMock = vi.fn(async () => ({ ok: true, status: 200, body }) as unknown as Response)
  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, push: (lines: string) => push(lines), close: () => close() }
}

/** The pieces asked of POST /api/voice/speech, as sent. */
function asked(): unknown[] {
  return api.requestBlob.mock.calls.map(([path, init]) => [path, JSON.parse(String((init as RequestInit).body))])
}

const said = (text: string): unknown => ['/api/voice/speech', { text, voice: 'af_heart', speed: null }]

describe('quick-ask speaks with the server voice', () => {
  it('unlocks the audio in the press and plays each sentence’s MP3 in the page language’s voice, in order', async () => {
    const synthesis = { speaking: false, pending: false, speak: vi.fn(), cancel: vi.fn() }
    vi.stubGlobal('speechSynthesis', synthesis)
    const chat = installChat()
    const engine = createQuickAskEngine()

    await engine.start({ chatId: null })
    const audio = FakeAudio.made[0]
    expect(audio.load).toHaveBeenCalledTimes(1)
    expect(synthesis.speak).toHaveBeenCalledTimes(1)
    const browser = FakeRecognition.last as FakeRecognition
    browser.onaudiostart?.()
    browser.onresult?.({ results: [[{ transcript: "What's on today" }]] })
    engine.release()
    browser.onend?.()
    await vi.waitFor(() => expect(chat.fetchMock).toHaveBeenCalledTimes(1))
    chat.push('d:{"type":"chat-id","chatId":"c-new"}\n0:"You have two meetings. "\n0:"One is at ten."\n')
    chat.close()

    await vi.waitFor(() => expect(asked()).toEqual([said(GREETING), said('You have two meetings.')]))
    await vi.waitFor(() => expect(audio.src).toBe('blob:clip-1'))
    audio.fire('playing')
    expect(engine.snapshot()).toMatchObject({ phase: 'speaking', reply: GREETING })
    audio.fire('ended')
    await vi.waitFor(() => expect(audio.src).toBe('blob:clip-2'))
    audio.fire('playing')
    audio.fire('ended')
    await vi.waitFor(() => expect(asked()).toHaveLength(3))
    await vi.waitFor(() => expect(audio.src).toBe('blob:clip-3'))
    audio.fire('playing')
    audio.fire('ended')

    expect(asked()[2]).toEqual(said('One is at ten.'))
    expect(engine.snapshot()).toMatchObject({ phase: 'idle', reply: `${GREETING} You have two meetings. One is at ten.` })
    expect(synthesis.speak).toHaveBeenCalledTimes(1)
    expect(api.request).toHaveBeenCalledWith('/api/voice/voices')
    engine.dispose()
  })

  it('speaks with the server voice in a browser with no speech synthesis of its own', async () => {
    const chat = installChat()
    const engine = createQuickAskEngine()

    await engine.start({ chatId: 'c-1' })
    const browser = FakeRecognition.last as FakeRecognition
    browser.onaudiostart?.()
    browser.onresult?.({ results: [[{ transcript: 'Hello' }]] })
    engine.release()
    browser.onend?.()
    await vi.waitFor(() => expect(chat.fetchMock).toHaveBeenCalledTimes(1))
    chat.push('0:"Hi there."\n')
    chat.close()

    await vi.waitFor(() => expect(asked()).toEqual([said(GREETING), said('Hi there.')]))
    engine.dispose()
  })
})
