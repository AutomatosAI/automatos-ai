/**
 * Theme voice — a chat turn sent outside React (lib/chat/turn-stream.ts, turn-frames.ts): the quick-ask voice's turn
 * is a normal POST /api/chat with spokenMode, to the app's own chat route as useChat's, with apiClient's headers; the
 * reply streams to the handlers with narration taken out; a new chat's id is handed over; an `e:` frame, a refused
 * request and a request that never arrived each end as a failure, and Stop as aborted. Only the network and apiClient
 * are stubbed.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/api-client', () => ({
  apiClient: {
    getBaseUrl: () => 'https://api.automatos.test',
    getAuthHeaders: async () => ({ Authorization: 'Bearer t0k', 'X-Workspace-ID': 'w-1' }),
  },
}))

import { parseChatLine } from '@/lib/chat/turn-frames'
import { chatTurnBody, streamChatTurn, type ChatTurnRequest } from '@/lib/chat/turn-stream'

const SPOKEN: ChatTurnRequest = { chatId: 'c-1', text: 'What is on today?', spokenMode: true }

/** A /api/chat response whose body is these lines, cut into the given chunks. */
function chatResponse(chunks: readonly string[], status = 200): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk))
      controller.close()
    },
  })
  return { ok: status < 400, status, body } as unknown as Response
}

function handlers() {
  return { onChatId: vi.fn(), onText: vi.fn(), onAgent: vi.fn() }
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('chatTurnBody', () => {
  it('is useChat’s body for a text turn, with spokenMode when the reply is heard', () => {
    expect(chatTurnBody(SPOKEN)).toEqual({
      id: 'c-1',
      message: { role: 'user', parts: [{ type: 'text', text: 'What is on today?' }] },
      selectedVisibilityType: 'private',
      spokenMode: true,
    })
  })

  it('starts a new chat with an empty id, and leaves spokenMode out of a typed turn', () => {
    const body = chatTurnBody({ chatId: null, text: 'Hi', spokenMode: false })

    expect(body.id).toBe('')
    expect('spokenMode' in body).toBe(false)
  })

  it('carries the locale when the turn names one, so Auto replies in that variant', () => {
    expect(chatTurnBody({ ...SPOKEN, locale: 'pt-BR' })).toMatchObject({ spokenMode: true, locale: 'pt-BR' })
  })

  it('leaves locale out when it is unknown', () => {
    expect('locale' in chatTurnBody(SPOKEN)).toBe(false)
    expect('locale' in chatTurnBody({ ...SPOKEN, locale: null })).toBe(false)
    expect('locale' in chatTurnBody({ ...SPOKEN, locale: '' })).toBe(false)
  })
})

describe('parseChatLine', () => {
  it('reads text, the chat id, narration, the agent and errors', () => {
    expect(parseChatLine('0:"Hello"')).toEqual({ kind: 'text', text: 'Hello' })
    expect(parseChatLine('d:{"type":"chat-id","chatId":"c-9"}')).toEqual({ kind: 'chat-id', chatId: 'c-9' })
    expect(parseChatLine('d:{"type":"narration","data":{"text":"Checking.","retracted":false}}')).toEqual({
      kind: 'narration',
      text: 'Checking.',
    })
    expect(parseChatLine('d:{"type": "agent-info", "agent": {"id": 7, "name": "Auto"}}')).toEqual({
      kind: 'agent',
      name: 'Auto',
    })
    expect(parseChatLine('e:{"message":"The model stopped answering.","code":"turn_failed"}')).toEqual({
      kind: 'error',
      message: 'The model stopped answering.',
      code: 'turn_failed',
    })
    expect(parseChatLine('data: {"type":"text-delta","delta":"Hi"}')).toEqual({ kind: 'text', text: 'Hi' })
  })

  it('skips what a spoken turn does not need, and lines that are not JSON', () => {
    for (const line of ['', 'd:{"type":"tool-start","data":{}}', 'd:{"type":"finish"}', '0:{oops', 'x:1']) {
      expect(parseChatLine(line), line).toBeNull()
    }
  })
})

describe('streamChatTurn', () => {
  it('posts the spoken turn to the app’s own /api/chat, as useChat does, with apiClient’s headers', async () => {
    const fetchMock = vi.fn(async () => chatResponse(['0:"Ok."\n']))
    vi.stubGlobal('fetch', fetchMock)

    await streamChatTurn(SPOKEN, handlers(), new AbortController().signal)

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('/api/chat')
    expect(init.method).toBe('POST')
    expect(init.headers).toEqual({
      Authorization: 'Bearer t0k',
      'X-Workspace-ID': 'w-1',
      'Content-Type': 'application/json',
    })
    expect(JSON.parse(String(init.body))).toMatchObject({ id: 'c-1', spokenMode: true })
  })

  it('streams the answer across chunks, hands over the new chat id and takes narration out', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => chatResponse([
      'd:{"type":"chat-id","chatId":"c-new"}\n0:"Let me check. "\nd:{"type":"narr',
      'ation","data":{"text":"Let me check. "}}\n0:"You have two',
      ' meetings."\nd:{"type":"finish"}',
    ])))
    const told = handlers()

    const outcome = await streamChatTurn({ ...SPOKEN, chatId: null }, told, new AbortController().signal)

    expect(outcome).toEqual({ kind: 'finished' })
    expect(told.onChatId).toHaveBeenCalledWith('c-new')
    // A line cut between chunks is read once it is whole.
    expect(told.onText.mock.calls.map(([text]) => text)).toEqual(['Let me check. ', '', 'You have two meetings.'])
  })

  it('ends as a failed turn with the turn’s own sentence on an e: frame', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => chatResponse(['0:"Partly"\ne:{"message":"The model stopped answering."}\n'])))

    const outcome = await streamChatTurn(SPOKEN, handlers(), new AbortController().signal)

    expect(outcome).toEqual({ kind: 'failed', reason: 'turn', message: 'The model stopped answering.' })
  })

  it('ends as a failed turn, with no internals, when the request is refused', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.stubGlobal('fetch', vi.fn(async () => chatResponse(['{"detail":{"code":"credits_exhausted"}}'], 402)))

    const outcome = await streamChatTurn(SPOKEN, handlers(), new AbortController().signal)

    expect(outcome).toEqual({ kind: 'failed', reason: 'turn', message: null })
  })

  it('ends as a network failure when the request never arrives', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }))

    const outcome = await streamChatTurn(SPOKEN, handlers(), new AbortController().signal)

    expect(outcome).toEqual({ kind: 'failed', reason: 'network', message: null })
  })

  it('says aborted when the caller stopped it', async () => {
    const controller = new AbortController()
    vi.stubGlobal('fetch', vi.fn(async () => {
      controller.abort()
      throw new DOMException('The operation was aborted.', 'AbortError')
    }))

    const outcome = await streamChatTurn(SPOKEN, handlers(), controller.signal)

    expect(outcome).toEqual({ kind: 'aborted' })
  })
})
