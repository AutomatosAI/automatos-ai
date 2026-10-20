import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-261 A-11b (@host): a theme's chat on the SDK's chat area (lib/theme-sdk/chat.ts).
//
// The SDK-chat hello theme (fixtures/hello-a11b) registers its own page on /chat and draws it from
// useChatController() alone: the conversation on screen, then that conversation's stream in a keyed
// child, with the host's ToolApproval card for each approval a reply raises. The app's own POST /api/chat
// is answered in the page with recorded turns (fixtures/hello-a11b/chat-turns.json, test data, no model
// is called); a held turn keeps its stream open until the browser aborts it, as a reply being written does.
//
// 1. The stream renders: the reply's words arrive and the turn ends.
// 2. Stop keeps the partial reply (P260-FX-30), marked stopped, and cancels the turn on the server.
// 3. Try again sends the owner's last message again, on the same conversation, and the new reply replaces the old.
// 4. A tool approval files under its reply as the host's card, and Approve & run posts the decision.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const HELLO_A11B: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a11b0', fixture: 'hello-a11b' }
const TURNS_FILE = path.resolve(__dirname, '..', 'fixtures', HELLO_A11B.fixture, 'chat-turns.json')
const CHAT_PATH = '/chat'
const CHAT_STREAM_PATH = '/api/chat'
const HISTORY_PATH = '/api/chat/history'
const CANCEL_PATH = /^\/api\/chat\/[^/]+\/cancel$/
const GRANT_ROUTE = '**/api/v1/approval-grants/*/grant'
const POSTS_KEY = '__helloA11bChatPosts'
const STATUS_ATTR = 'data-theme-host'
const TEXT_LINE = '0:'
const DATA_LINE = 'd:'
const OK = 200

const ASK = 'Find me a laptop under 1,000 euros'
const ANSWER = 'Here are three laptops under 1,000 euros.'
const AGAIN = 'Here are three more laptops.'
const PARTIAL = 'I am still writing the comparison'
const APPROVAL_TEXT = 'This action needs your approval before it runs.'
const APPROVE_BUTTON = 'Approve & run'
const APPROVED = 'Approved.'
const TOOL_CARD = '[data-protected-surface="tool-approval"]'
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build/i

type TurnName = 'answer' | 'again' | 'partial' | 'approval'

interface Frame {
  readonly text?: string
  readonly data?: unknown
}

interface RecordedTurns {
  readonly chatId: string
  readonly turns: Readonly<Record<TurnName, { readonly frames: readonly Frame[]; readonly hold?: boolean }>>
}

/** A turn as the page serves it: its stream lines, and whether the stream stays open after them. */
interface ServedTurn {
  readonly lines: readonly string[]
  readonly hold: boolean
}

interface ChatScript {
  readonly path: string
  readonly postsKey: string
  /** The turn each POST /api/chat gets, in order (the last one repeats). */
  readonly turns: readonly ServedTurn[]
}

interface ChatPost {
  readonly id?: string
  readonly message?: { readonly role?: string; readonly parts?: ReadonlyArray<{ readonly text?: string }> }
}

type PostsWindow = Window & Record<string, ChatPost[] | undefined>

function readTurns(): RecordedTurns {
  const turns = JSON.parse(fs.readFileSync(TURNS_FILE, 'utf8')) as RecordedTurns
  if (typeof turns.chatId !== 'string' || typeof turns.turns !== 'object') throw new Error(`${TURNS_FILE} is not a set of recorded turns`)
  return turns
}

/** A recorded frame as the line lib/chat/hooks.ts reads. */
function lineOf(frame: Frame): string {
  return frame.text !== undefined ? `${TEXT_LINE}${JSON.stringify(frame.text)}` : `${DATA_LINE}${JSON.stringify(frame.data)}`
}

function served(recorded: RecordedTurns, name: TurnName): ServedTurn {
  const turn = recorded.turns[name]
  return { lines: turn.frames.map(lineOf), hold: turn.hold === true }
}

/**
 * Runs in the page before its own scripts: the app's POST /api/chat (same origin) gets the next recorded
 * turn as a stream, every other request goes out as it would. A held turn's stream stays open after its
 * lines and fails with an AbortError when the request's signal aborts, as fetch's own stream does.
 */
function stubChatStream(script: ChatScript): void {
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
async function noStoredConversations(page: Page): Promise<void> {
  await page.route((url) => url.pathname === HISTORY_PATH, (route) => fulfillBackendJson(route, OK, []))
}

/** Answers Stop's cancel and records the path of each one. */
async function recordCancels(page: Page): Promise<string[]> {
  const cancelled: string[] = []
  await page.route((url) => CANCEL_PATH.test(url.pathname), async (route) => {
    if (route.request().method() === 'POST') cancelled.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, { cancelled: true })
  })
  return cancelled
}

/** Opens /chat under the hello theme, every POST /api/chat answered with `turns` in order. */
async function openHelloChat(page: Page, turns: readonly TurnName[]): Promise<Locator> {
  const recorded = readTurns()
  const script: ChatScript = { path: CHAT_STREAM_PATH, postsKey: POSTS_KEY, turns: turns.map((name) => served(recorded, name)) }
  await page.addInitScript(stubChatStream, script)
  await noStoredConversations(page)
  const theme = await serveInstalledTheme(page, HELLO_A11B)
  await chooseStyle(page, theme.style, APP_URL)
  await page.goto(CHAT_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByTestId('hello-a11b-frame')).toBeVisible()
  const chat = page.getByTestId('hello-a11b-chat')
  await expect(chat).toBeVisible()
  return chat
}

async function ask(chat: Locator, words: string): Promise<void> {
  await chat.getByRole('textbox', { name: 'Message', exact: true }).fill(words)
  await chat.getByRole('button', { name: 'Send', exact: true }).click()
}

function chatPosts(page: Page): Promise<ChatPost[]> {
  return page.evaluate((key) => [...((window as unknown as PostsWindow)[key] ?? [])], POSTS_KEY)
}

function replies(chat: Locator): Locator {
  return chat.locator('[data-testid="hello-a11b-message"][data-role="assistant"]')
}

function askedWords(post: ChatPost | undefined): string | undefined {
  return post?.message?.parts?.[0]?.text
}

test.describe('the SDK chat area in the browser', () => {
  test('the stream renders through useChatController', { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const chat = await openHelloChat(page, ['answer'])
    await ask(chat, ASK)

    await expect(replies(chat)).toHaveCount(1)
    await expect(replies(chat).getByTestId('hello-a11b-text')).toHaveText(ANSWER)
    await expect(chat).toHaveAttribute('data-status', 'idle')
    await expect(replies(chat)).toHaveAttribute('data-writing', 'false')
    await expect(chat).toHaveAttribute('data-chat-id', readTurns().chatId)
    expect((await chatPosts(page)).map(askedWords)).toEqual([ASK])

    expect(watch.pageErrors).toEqual([])
    expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
  })

  test('Stop keeps the partial reply and cancels the turn (FX-30)', { tag: '@host' }, async ({ page }) => {
    const cancelled = await recordCancels(page)
    const chat = await openHelloChat(page, ['partial'])
    await ask(chat, ASK)

    const reply = replies(chat)
    await expect(reply.getByTestId('hello-a11b-text')).toHaveText(PARTIAL)
    await expect(reply).toHaveAttribute('data-writing', 'true')
    await expect(chat).toHaveAttribute('data-status', 'streaming')

    await chat.getByRole('button', { name: 'Stop', exact: true }).click()
    await expect(chat).toHaveAttribute('data-status', 'idle')
    await expect(reply).toHaveCount(1)
    await expect(reply).toHaveAttribute('data-stopped', 'true')
    await expect(reply.getByTestId('hello-a11b-text')).toHaveText(PARTIAL)
    await expect.poll(() => cancelled).toEqual([`/api/chat/${readTurns().chatId}/cancel`])
  })

  test('Try again sends the last message again on the same conversation', { tag: '@host' }, async ({ page }) => {
    const chat = await openHelloChat(page, ['answer', 'again'])
    await ask(chat, ASK)
    await expect(replies(chat).getByTestId('hello-a11b-text')).toHaveText(ANSWER)

    await chat.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect(replies(chat)).toHaveCount(1)
    await expect(replies(chat).getByTestId('hello-a11b-text')).toHaveText(AGAIN)
    const posts = await chatPosts(page)
    expect(posts.map(askedWords)).toEqual([ASK, ASK])
    expect(posts[1]?.id).toBe(readTurns().chatId)
    await expect(chat.locator('[data-testid="hello-a11b-message"][data-role="user"]')).toHaveCount(1)
  })

  test("a tool approval files as the host's card, and approving it posts the decision", { tag: '@host' }, async ({ page }) => {
    const decided: string[] = []
    await page.route(GRANT_ROUTE, async (route) => {
      if (route.request().method() === 'POST') decided.push(new URL(route.request().url()).pathname)
      await fulfillBackendJson(route, OK, { grant: { id: 11201, status: 'granted', details: {} } })
    })
    const chat = await openHelloChat(page, ['approval'])
    await ask(chat, ASK)

    const reply = replies(chat)
    await expect(reply.getByTestId('hello-a11b-text')).toHaveText(APPROVAL_TEXT)
    const approval = reply.getByTestId('hello-a11b-approval')
    await expect(approval).toHaveCount(1)
    await expect(approval).toHaveAttribute('data-status', 'waiting')
    const card = approval.locator(TOOL_CARD)
    await expect(card).toBeVisible()

    await card.getByRole('button', { name: APPROVE_BUTTON }).click()
    await expect.poll(() => decided).toEqual(['/api/v1/approval-grants/11201/grant'])
    await expect(card.getByRole('status')).toHaveText(APPROVED)
    await expect(approval).toHaveAttribute('data-status', 'answered')
  })
})
