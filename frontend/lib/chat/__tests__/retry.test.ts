/**
 * PRD-260 — asking again (lib/chat/retry.ts, useChatWithReceipts' reload): Studio's regenerate
 * and Play's Try again. The hook cases run the REAL useChat over a stubbed /api/chat stream:
 * before this, reload kept a sendMessage from a loading render, so after a failed turn the
 * owner's message left the list and nothing was sent. P260-FX-09 (M30): a conversation loaded
 * from the server has rows with parts and no content, so Try again sent an empty message.
 */
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/types'

vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ isLoaded: true, getToken: async () => null }) }))
vi.mock('@/lib/chat/api', () => ({ getChatMessages: vi.fn(async () => []) }))

import { lastTurn } from '@/lib/chat/retry'
import { useChatWithReceipts } from '@/lib/chat/use-chat-with-receipts'

const ASK = 'Delete the old brief?'
const FAILED = 'e:{"message":"The model stopped answering."}'

const message = (id: string, role: 'user' | 'assistant', content: string) =>
  ({ id, role, content, parts: [{ type: 'text', text: content }] }) as unknown as ChatMessage

/** A row as getChatMessages hands it over (api/chat.py get_chat_messages): parts, no content. */
const stored = (id: string, role: 'user' | 'assistant', text: string, source: unknown = null) =>
  ({ id, role, parts: [{ type: 'text', text }], source }) as unknown as ChatMessage

describe('lastTurn', () => {
  it("is the owner's last message and what stays before it", () => {
    const messages = [message('1', 'user', 'Hello'), message('2', 'assistant', 'Hi.'), message('3', 'user', ASK), message('4', 'assistant', '')]
    expect(lastTurn(messages)).toEqual({ kept: messages.slice(0, 2), content: ASK })
    expect(messages).toHaveLength(4)
  })

  it('is null before the owner has said anything', () => {
    expect(lastTurn([])).toBeNull()
    expect(lastTurn([message('1', 'assistant', 'Hi.')])).toBeNull()
  })

  it('reads a message loaded from the server, which has its words in parts and no content (M30)', () => {
    expect(lastTurn([{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'hello' }] } as unknown as ChatMessage])?.content).toBe('hello')
    const messages = [stored('1', 'user', 'Hello'), stored('2', 'assistant', 'Hi.'), stored('3', 'user', ASK), stored('4', 'assistant', '⚠️ It failed.')]
    expect(lastTurn(messages)).toEqual({ kept: messages.slice(0, 2), content: ASK })
  })
})

// --- the wire: one stream per request, written line by line ----------------------------------------------------

const streams: Array<{ readonly push: (line: string) => void; readonly end: () => void }> = []

const chatEndpoint = vi.fn(async (url: string) => {
  if (url !== '/api/chat') throw new Error(`unexpected request: ${url}`)
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  streams.push({ push: (line) => controller?.enqueue(encoder.encode(`${line}\n`)), end: () => controller?.close() })
  return { ok: true, status: 200, headers: { get: () => null }, body } as unknown as Response
})

async function write(...lines: string[]) {
  await act(async () => {
    lines.forEach((line) => streams[streams.length - 1].push(line))
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

async function end() {
  await act(async () => {
    streams[streams.length - 1].end()
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

const asks = (messages: ChatMessage[]) => messages.filter((m) => m.role === 'user' && m.content === ASK)

/** The words of the message the request at `index` sent (useChat posts { message: { parts } }). */
function sentText(index: number): unknown {
  const init = (chatEndpoint.mock.calls[index] as unknown[])[1] as RequestInit | undefined
  const body = JSON.parse(String(init?.body)) as { message: { parts: Array<{ text?: unknown }> } }
  return body.message.parts[0]?.text
}

beforeEach(() => vi.stubGlobal('fetch', chatEndpoint))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  streams.length = 0
})

describe("useChatWithReceipts' reload over the real useChat", () => {
  it('after a turn that failed, sends the last message again in its place', async () => {
    const { result } = renderHook(() => useChatWithReceipts({ id: 'c1' }))
    await act(async () => {
      void result.current.sendMessage(ASK)
    })
    await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(1))
    await write(FAILED)
    await end()
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    act(() => result.current.reload())
    await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(2))
    expect(asks(result.current.messages)).toHaveLength(1)
    await write('0:"Done."')
    await end()
    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current.messages.map((m) => m.role)).toEqual(['user', 'assistant'])
  })

  it('on a reopened chat whose last turn failed (stored rows: parts, no content), sends the stored message again (M30)', async () => {
    const failed = stored('s2', 'assistant', '⚠️ The model is unavailable', { origin: 'turn_error', label: 'Error' })
    const initialMessages = [stored('s1', 'user', ASK), failed]
    const { result } = renderHook(() => useChatWithReceipts({ id: 'c1', initialMessages }))

    act(() => result.current.reload())
    await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(1))
    expect(sentText(0)).toBe(ASK)
    expect(asks(result.current.messages)).toHaveLength(1)
    expect(result.current.messages.map((m) => m.id)).not.toContain('s2')
    await end()
    await waitFor(() => expect(result.current.isLoading).toBe(false))
  })

  it('while a turn runs, does nothing: the message stays and nothing more is sent', async () => {
    const { result } = renderHook(() => useChatWithReceipts({ id: 'c1' }))
    await act(async () => {
      void result.current.sendMessage(ASK)
    })
    await waitFor(() => expect(chatEndpoint).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(result.current.isLoading).toBe(true))

    act(() => result.current.reload())
    expect(chatEndpoint).toHaveBeenCalledTimes(1)
    expect(asks(result.current.messages)).toHaveLength(1)
    await end()
    await waitFor(() => expect(result.current.isLoading).toBe(false))
  })
})
