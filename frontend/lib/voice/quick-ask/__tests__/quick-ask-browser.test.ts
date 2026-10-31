/**
 * Theme voice, web quick-ask — over the browser's own objects: the default deps (deps.ts) watch the tab being hidden
 * and the page unloading and tell the open chat a turn landed; and the engine with no deps given runs end to end over
 * stand-ins for SpeechRecognition, speechSynthesis and fetch: the member's words go to POST /api/chat with spokenMode
 * through apiClient's base URL, the streamed reply is spoken with the AI notice first, and Stop cancels the turn
 * through the chat's own Stop (POST /api/chat/{id}/cancel). apiClient's stand-in answers GET /api/voice/voices with no
 * list, so Auto's server voice has no voice here and the browser's speaks (quick-ask-server-voice.test.ts has the
 * server voice speaking). deps.ts keeps that choice for the page session (module scope): every test here shares it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'https://api.automatos.test',
    getAuthHeaders: async () => ({ Authorization: 'Bearer t0k' }),
    request: vi.fn(async () => ({ cancelled: true })),
  },
}))

import { apiClient } from '@/lib/api-client'
import { cancelChatTurn } from '@/lib/chat/api'
import { streamChatTurn } from '@/lib/chat/turn-stream'
import { createQuickAskEngine } from '@/lib/voice/quick-ask'
import { AI_NOTICE_STORAGE_KEY } from '@/lib/voice/quick-ask/config'
import { browserDeps } from '@/lib/voice/quick-ask/deps'

import { GREETING } from './quick-ask-harness'

const CHAT_CHANGED = 'automatos:chat-changed'

beforeEach(() => {
  window.sessionStorage.clear()
  vi.clearAllMocks()
})

afterEach(() => {
  Reflect.deleteProperty(document, 'visibilityState')
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function setVisibility(state: DocumentVisibilityState): void {
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => state })
}

describe('browserDeps', () => {
  it('calls back when the tab is hidden or the page unloads, until unsubscribed', () => {
    const callback = vi.fn()
    const unwatch = browserDeps().onPageHidden(callback)

    window.dispatchEvent(new Event('pagehide'))
    setVisibility('visible')
    document.dispatchEvent(new Event('visibilitychange'))
    expect(callback).toHaveBeenCalledTimes(1)
    setVisibility('hidden')
    document.dispatchEvent(new Event('visibilitychange'))
    expect(callback).toHaveBeenCalledTimes(2)
    unwatch()
    window.dispatchEvent(new Event('pagehide'))

    expect(callback).toHaveBeenCalledTimes(2)
  })

  it('tells the open conversation the turn landed with the chat-changed window event', () => {
    const changed = vi.fn()
    window.addEventListener(CHAT_CHANGED, changed)

    browserDeps().announceTurn('c-9')
    window.removeEventListener(CHAT_CHANGED, changed)

    expect((changed.mock.calls[0][0] as CustomEvent).detail).toEqual({ chat_id: 'c-9' })
  })

  it('sends through the chat turn stream and stops through the chat’s Stop', () => {
    const deps = browserDeps()

    expect(deps.sendTurn).toBe(streamChatTurn)
    expect(deps.cancelTurn).toBe(cancelChatTurn)
    expect(deps.storage).toBe(window.sessionStorage)
  })

  it('has no recognition in a browser without it, and no browser voice behind the server voice', async () => {
    const deps = browserDeps()
    const events = { onStart: vi.fn(), onBoundary: vi.fn(), onEnd: vi.fn() }

    expect(deps.recognition()).toBeNull()
    deps.synthesis?.speak('Hello.', events)

    // No server voice here (no voice list) and no browser voice: the piece ends unsaid, so a turn still finishes.
    await vi.waitFor(() => expect(events.onEnd).toHaveBeenCalledTimes(1))
    expect(events.onStart).not.toHaveBeenCalled()
  })
})

class FakeRecognition {
  static last: FakeRecognition | null = null
  lang = ''
  continuous = false
  interimResults = false
  maxAlternatives = 0
  onstart: (() => void) | null = null
  onaudiostart: (() => void) | null = null
  onaudioend: (() => void) | null = null
  onresult: ((event: { readonly results: { transcript: string }[][] }) => void) | null = null
  onerror: ((event: { readonly error: string }) => void) | null = null
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
  onstart: (() => void) | null = null
  onboundary: (() => void) | null = null
  onend: (() => void) | null = null
  onerror: ((event: { error: string }) => void) | null = null
  constructor(readonly text: string) {}
}

/** The server voice's <audio> element: unlocked in the press, never played here (the server has no voice). */
class FakeAudio extends EventTarget {
  src = ''
  readonly play = vi.fn(async () => undefined)
  readonly pause = vi.fn()
  readonly load = vi.fn()
  readonly removeAttribute = vi.fn()
}

/** speechSynthesis's stand-in: it keeps every utterance for the test to play. */
function installSpeech() {
  const synthesis = { speaking: false, pending: false, speak: vi.fn(), cancel: vi.fn() }
  vi.stubGlobal('SpeechRecognition', FakeRecognition)
  vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
  vi.stubGlobal('speechSynthesis', synthesis)
  vi.stubGlobal('Audio', FakeAudio)
  const utterance = (index: number): FakeUtterance => synthesis.speak.mock.calls[index][0] as FakeUtterance
  const texts = (): string[] => synthesis.speak.mock.calls.map(([spoken]) => (spoken as FakeUtterance).text)
  return { synthesis, utterance, texts }
}

/** POST /api/chat's stand-in: a 200 whose body the test streams line by line. */
function installChat() {
  let push: (lines: string) => void = () => undefined
  let close: () => void = () => undefined
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      push = (lines) => controller.enqueue(new TextEncoder().encode(lines))
      close = () => controller.close()
    },
  })
  const fetchMock = vi.fn(async (_url: string, _init: RequestInit) => ({ ok: true, status: 200, body }) as unknown as Response)
  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, push: (lines: string) => push(lines), close: () => close() }
}

/** A hold that hears the words and lets go, then waits for the turn to be posted. */
async function askAloud(engine: ReturnType<typeof createQuickAskEngine>, chat: ReturnType<typeof installChat>) {
  await engine.start({ chatId: null })
  const browser = FakeRecognition.last as FakeRecognition
  browser.onaudiostart?.()
  browser.onresult?.({ results: [[{ transcript: "What's on today" }]] })
  expect(engine.snapshot()).toMatchObject({ phase: 'listening', micOpen: true, heard: "What's on today" })
  engine.release()
  expect(browser.stop).toHaveBeenCalledTimes(1)
  browser.onend?.()
  await vi.waitFor(() => expect(chat.fetchMock).toHaveBeenCalledTimes(1))
  return browser
}

describe('quick-ask over the browser’s own objects', () => {
  it('hears the member, posts the turn to /api/chat with spokenMode, and speaks the streamed reply', async () => {
    const speech = installSpeech()
    const chat = installChat()
    const changed = vi.fn()
    window.addEventListener(CHAT_CHANGED, changed)
    const engine = createQuickAskEngine()

    const browser = await askAloud(engine, chat)
    expect(browser).toMatchObject({ continuous: true, interimResults: true, lang: navigator.language })
    const [url, init] = chat.fetchMock.mock.calls[0]
    expect(url).toBe('/api/chat')
    expect(JSON.parse(String(init.body))).toMatchObject({
      id: '', spokenMode: true, message: { role: 'user', parts: [{ type: 'text', text: "What's on today" }] },
    })
    chat.push('d:{"type":"chat-id","chatId":"c-new"}\n0:"You have **two** meetings. "\n0:"One is at ten."\n')
    // The empty utterance first is the press priming the speech (iOS Safari speaks only after a gesture did).
    await vi.waitFor(() => expect(speech.texts()).toEqual(['', GREETING, 'You have two meetings.']))
    expect(engine.snapshot().chatId).toBe('c-new')
    speech.utterance(1).onstart?.()
    expect(engine.snapshot()).toMatchObject({ phase: 'speaking', reply: GREETING })
    chat.close()
    await vi.waitFor(() => expect(speech.texts()).toHaveLength(4))
    speech.utterance(1).onend?.()
    for (const index of [2, 3]) {
      speech.utterance(index).onstart?.()
      speech.utterance(index).onend?.()
    }

    expect(speech.texts()[3]).toBe('One is at ten.')
    expect(engine.snapshot()).toMatchObject({ phase: 'idle', reply: `${GREETING} You have two meetings. One is at ten.` })
    expect((changed.mock.calls[0][0] as CustomEvent).detail).toEqual({ chat_id: 'c-new' })
    expect(window.sessionStorage.getItem(AI_NOTICE_STORAGE_KEY)).toBe('1')
    window.removeEventListener(CHAT_CHANGED, changed)
    engine.dispose()
  })

  it('Stop cancels the speech, aborts the stream and cancels the turn through the chat’s Stop', async () => {
    const speech = installSpeech()
    const chat = installChat()
    const engine = createQuickAskEngine()
    await askAloud(engine, chat)
    chat.push('d:{"type":"chat-id","chatId":"c-new"}\n0:"Two meetings. One"\n')
    await vi.waitFor(() => expect(speech.texts()).toEqual(['', GREETING, 'Two meetings.']))
    speech.synthesis.cancel.mockClear()

    engine.stop()

    expect(speech.synthesis.cancel).toHaveBeenCalledTimes(1)
    expect((chat.fetchMock.mock.calls[0][1].signal as AbortSignal).aborted).toBe(true)
    await vi.waitFor(() => expect(apiClient.request).toHaveBeenCalledWith('/api/chat/c-new/cancel', { method: 'POST' }))
    expect(engine.snapshot().phase).toBe('idle')
    chat.close()
    engine.dispose()
  })

  it('a hidden tab stops the microphone and sends nothing', async () => {
    installSpeech()
    const chat = installChat()
    const engine = createQuickAskEngine()
    await engine.start({ chatId: 'c-1' })
    const browser = FakeRecognition.last as FakeRecognition
    browser.onaudiostart?.()
    browser.onresult?.({ results: [[{ transcript: 'Book a' }]] })

    setVisibility('hidden')
    document.dispatchEvent(new Event('visibilitychange'))
    browser.onend?.()
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(browser.abort).toHaveBeenCalled()
    expect(engine.snapshot()).toMatchObject({ phase: 'idle', micOpen: false })
    expect(chat.fetchMock).not.toHaveBeenCalled()
    engine.dispose()
  })
})
