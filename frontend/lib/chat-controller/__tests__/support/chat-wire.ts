/**
 * PRD-261 A-19e1 — the wire under the chat controller cases (../controller-cases.test.ts): useChat's
 * data stream (`0:` words, `d:` data, `e:` a failure), one stream per request to /api/chat, which the
 * test writes into; `wire.refusal` answers a send with that status and body instead. The test stubs
 * the global fetch with `chatEndpoint` and empties `turns` and `posts` after each case.
 */
import { act } from '@testing-library/react'
import { vi } from 'vitest'

export interface Turn {
  readonly push: (line: string) => void
  readonly end: () => void
}

/** The body useChat posts: the words of the message are its first part's text. */
export type ChatPost = { readonly message?: { readonly parts?: ReadonlyArray<{ readonly text?: string }> } }

export type Refusal = { readonly status: number; readonly body: string }

const CHAT_PATH = '/api/chat'
export const turns: Turn[] = []
export const posts: ChatPost[] = []
export const wire: { refusal: Refusal | null } = { refusal: null }

/** A response body the test writes into; an abort (Stop, an unmount) fails the read as a browser's does. */
function streamResponse(signal: AbortSignal | null | undefined): Response {
  const encoder = new TextEncoder()
  let controller: ReadableStreamDefaultController<Uint8Array> | null = null
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      controller = c
    },
  })
  signal?.addEventListener('abort', () => controller?.error(new DOMException('The user aborted a request.', 'AbortError')))
  turns.push({ push: (line) => controller?.enqueue(encoder.encode(`${line}\n`)), end: () => controller?.close() })
  return { ok: true, status: 200, headers: { get: () => null }, body } as unknown as Response
}

function refusedResponse({ status, body }: Refusal): Response {
  return { ok: false, status, headers: { get: () => null }, text: async () => body } as unknown as Response
}

export const chatEndpoint = vi.fn(async (url: string, init?: RequestInit): Promise<Response> => {
  if (url !== CHAT_PATH) throw new Error(`unexpected request: ${url}`)
  posts.push(JSON.parse(String(init?.body)) as ChatPost)
  return wire.refusal ? refusedResponse(wire.refusal) : streamResponse(init?.signal)
})

export const tick = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
export const dataLine = (type: string, data: Record<string, unknown>) => `d:${JSON.stringify({ type, data })}`
export const wordsLine = (text: string) => `0:${JSON.stringify(text)}`

/** Write lines into the newest request's stream, and let useChat read them; endStream closes it. */
export async function stream(...lines: string[]) {
  await act(async () => {
    lines.forEach((line) => turns[turns.length - 1].push(line))
    await tick()
  })
}

export const endStream = () => act(async () => {
  turns[turns.length - 1].end()
  await tick()
})
