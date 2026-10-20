import type { Page } from '@playwright/test'
import { fulfillBackendJson } from './backend-route'

// PRD-261 A-11b, A-17b: the app's own POST /api/chat answered in the page with recorded turns, in
// the stream format lib/chat/hooks.ts reads ('0:' text lines and 'd:' data lines). No model is
// called. A held turn keeps its stream open until the browser aborts it, as a reply being written
// does; a turn with a status other than 200 answers that status with no stream, as a refused
// request does. specs/sdk-chat.spec.ts (the hello theme's chat) and specs/play-parity/ (Play's
// chat) both serve their turns through here.

const TEXT_LINE = '0:'
const DATA_LINE = 'd:'
const HISTORY_PATH = '/api/chat/history'
const CANCEL_PATH = /^\/api\/chat\/[^/]+\/cancel$/
const OK = 200

/** One recorded frame: a text line or a data line. */
export interface Frame {
  readonly text?: string
  readonly data?: unknown
}

/** A turn as the page serves it: its stream lines, and whether the stream stays open after them. */
export interface ServedTurn {
  readonly lines: readonly string[]
  readonly hold: boolean
  /** The response's status; absent is 200 with the stream. */
  readonly status?: number
}

export interface ChatScript {
  readonly path: string
  readonly postsKey: string
  /** The turn each POST /api/chat gets, in order (the last one repeats). */
  readonly turns: readonly ServedTurn[]
}

export interface ChatPost {
  readonly id?: string
  readonly message?: { readonly role?: string; readonly parts?: ReadonlyArray<{ readonly text?: string }> }
}

type PostsWindow = Window & Record<string, ChatPost[] | undefined>

/** A recorded frame as the line lib/chat/hooks.ts reads. */
export function lineOf(frame: Frame): string {
  return frame.text !== undefined ? `${TEXT_LINE}${JSON.stringify(frame.text)}` : `${DATA_LINE}${JSON.stringify(frame.data)}`
}

/**
 * Runs in the page before its own scripts: the app's POST /api/chat (same origin) gets the next recorded
 * turn as a stream, every other request goes out as it would. A held turn's stream stays open after its
 * lines and fails with an AbortError when the request's signal aborts, as fetch's own stream does.
 * Self-contained: Playwright sends its source to the page.
 */
export function stubChatStream(script: ChatScript): void {
  const realFetch = window.fetch.bind(window)
  const posts: ChatPost[] = []
  Object.defineProperty(window, script.postsKey, { value: posts })
  let next = 0
  const holdOpen = (controller: ReadableStreamDefaultController<Uint8Array>, signal: AbortSignal | null | undefined) => {
    const abort = () => controller.error(new DOMException('The user aborted a request.', 'AbortError'))
    if (signal?.aborted) abort()
    else signal?.addEventListener('abort', abort, { once: true })
  }
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const url = new URL(raw, window.location.href)
    const method = (init?.method ?? (input instanceof Request ? input.method : 'GET')).toUpperCase()
    if (url.origin !== window.location.origin || url.pathname !== script.path || method !== 'POST') return realFetch(input, init)
    posts.push(typeof init?.body === 'string' ? (JSON.parse(init.body) as ChatPost) : {})
    const turn = script.turns[Math.min(next, script.turns.length - 1)]
    next += 1
    if (turn.status !== undefined && turn.status !== 200) {
      return Promise.resolve(new Response('The chat request was refused.', { status: turn.status }))
    }
    const encoder = new TextEncoder()
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        turn.lines.forEach((line) => controller.enqueue(encoder.encode(`${line}\n`)))
        if (turn.hold) holdOpen(controller, init?.signal)
        else controller.close()
      },
    })
    return Promise.resolve(new Response(body, { status: 200, headers: { 'content-type': 'text/plain; charset=utf-8' } }))
  }
}

/** Answers the conversations list with none, so the cold start opens a new chat whatever the backend holds. */
export async function noStoredConversations(page: Page): Promise<void> {
  await page.route((url) => url.pathname === HISTORY_PATH, (route) => fulfillBackendJson(route, OK, []))
}

/** Answers Stop's cancel and records the path of each one. */
export async function recordCancels(page: Page): Promise<string[]> {
  const cancelled: string[] = []
  await page.route((url) => CANCEL_PATH.test(url.pathname), async (route) => {
    if (route.request().method() === 'POST') cancelled.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, { cancelled: true })
  })
  return cancelled
}

/** Every POST /api/chat body the page sent, in order. */
export function chatPosts(page: Page, postsKey: string): Promise<ChatPost[]> {
  return page.evaluate((key) => [...((window as unknown as PostsWindow)[key] ?? [])], postsKey)
}

/** The words of the message a post sent. */
export function askedWords(post: ChatPost | undefined): string | undefined {
  return post?.message?.parts?.[0]?.text
}
