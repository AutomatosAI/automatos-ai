import { expect, test, type Page, type Route } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-268 §6.4, US-W02 (@host): the day's brief arrives as a message in Auto's thread and never takes the chat away.
//
// Under an installed theme that attacks nothing (fixtures/hello-a04b: a frame that only wraps the page), a fresh
// browser opens /chat on its newest conversation (PRD-237's cold start, app/chat/page.tsx), here the person's Auto
// thread, which holds today's brief. The thread draws the brief card, with Listen in its protected surface
// `brief-voice`, and the composer stays on screen beside it and takes words. specs/protected-surfaces.spec.ts says
// why its hostile sheet starts on a new chat instead.
//
// The backend answers are the spec's: the conversation list holds only the Auto thread, the thread and its one
// message (the brief) are answered here, and the brief's open names that message, as when another tab or the
// scheduled digest posted it first. The suggestions list is empty. Nothing is sent and nothing plays.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const STATUS_ATTR = 'data-theme-host'
const OK = 200

const HELLO_A04B: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000a04b0', fixture: 'hello-a04b' }
const FRAME_TEST_ID = 'hello-a04b-frame'
/** components/proactive/thread-cards.tsx: the cards under an Auto-thread message. */
const THREAD_CARDS_TEST_ID = 'thread-cards'
const COMPOSER_PLACEHOLDER = 'Send a message...'
/** components/voice/brief-voice.tsx LISTEN_TO_THE_BRIEF. */
const LISTEN = 'Listen to the brief'
const BRIEF_VOICE_SURFACE = '[data-protected-surface="brief-voice"]'
const DRAFT = 'Thanks, what about the flights?'

const BRIEF_REF = { chat_id: '00000000-0000-4000-8000-0000000b0201', message_id: '00000000-0000-4000-8000-0000000b0202' }
const HISTORY_PATH = '/api/chat/history'
const THREAD_PATH = `/api/chat/${BRIEF_REF.chat_id}`
const MESSAGES_PATH = `${THREAD_PATH}/messages`
const OPEN_PATH = '/api/proactive/brief/open'
const PROACTIVE_PREFIX = '/api/proactive/'

const POSTED_AT = '2026-10-30T07:00:00Z'
const BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Morning, Sam.',
  prose: 'One thing needs you before noon.',
  spoken: 'Morning, Sam. 1. Approve the Lisbon flights.',
  sections: { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 },
  labels: { sections: {}, more: {} },
  state_hash: 'e2e-brief-thread',
  generated_at: POSTED_AT,
}
/** The person's Auto thread (PRD-205 kind 'auto'), as GET /api/chat/history lists it. */
const AUTO_THREAD = {
  id: BRIEF_REF.chat_id,
  userId: 1,
  title: 'Auto',
  createdAt: POSTED_AT,
  updatedAt: POSTED_AT,
  visibility: 'private',
  lastContext: {},
  lastMessagePreview: BRIEF.prose,
  kind: 'auto',
}
/** The day's brief message: its text part, then its brief@1 part, posted via open. */
const BRIEF_MESSAGES = [
  {
    id: BRIEF_REF.message_id,
    role: 'assistant',
    parts: [{ type: 'text', text: BRIEF.prose }, { type: 'brief@1', brief: BRIEF }],
    attachments: [],
    source: { origin: 'daily_digest', via: 'open', local_date: BRIEF.date, label: 'Auto · background' },
    createdAt: POSTED_AT,
  },
]

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

/** The proactive routes: the open names the day's message (and is recorded); the Suggested list lists nothing. */
async function answerProactive(route: Route, opened: string[]): Promise<void> {
  const request = route.request()
  const pathname = new URL(request.url()).pathname
  if (pathname === OPEN_PATH && request.method() === 'POST') opened.push(pathname)
  await fulfillBackendJson(route, OK, pathname === OPEN_PATH ? { ...BRIEF_REF, posted: false } : { suggestions: [] })
}

/** Answers the conversation list, the Auto thread and its brief, and the proactive routes; returns the opens. */
async function answerTheAutoThread(page: Page): Promise<string[]> {
  const opened: string[] = []
  await page.route(onPath(HISTORY_PATH), (route) => fulfillBackendJson(route, OK, [AUTO_THREAD]))
  await page.route(onPath(THREAD_PATH), (route) => fulfillBackendJson(route, OK, { ...AUTO_THREAD, turnInFlight: false }))
  await page.route(onPath(MESSAGES_PATH), (route) => fulfillBackendJson(route, OK, BRIEF_MESSAGES))
  await page.route((url) => url.pathname.startsWith(PROACTIVE_PREFIX), (route) => answerProactive(route, opened))
  return opened
}

test.describe("the day's brief in Auto's thread", () => {
  test("under an installed theme, /chat on Auto's thread shows today's brief card and the composer together", { tag: '@host' }, async ({ page }) => {
    const watch = watchPage(page)
    const opened = await answerTheAutoThread(page)
    const served = await serveInstalledTheme(page, HELLO_A04B)
    await chooseStyle(page, served.style, APP_URL)
    await page.goto(CHAT_PATH)

    await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
    const frame = page.getByTestId(FRAME_TEST_ID)
    await expect(frame).toBeVisible()
    const card = frame.getByTestId(THREAD_CARDS_TEST_ID).getByRole('article', { name: BRIEF.greeting })
    await expect(card).toBeVisible()
    await expect(card.locator(BRIEF_VOICE_SURFACE)).toHaveCount(1)
    await expect(card.getByRole('button', { name: LISTEN })).toBeVisible()

    // The composer is on screen beside the brief, and takes words without the brief going.
    const composer = frame.getByPlaceholder(COMPOSER_PLACEHOLDER).filter({ visible: true }).first()
    await expect(composer).toBeVisible()
    await composer.fill(DRAFT)
    await expect(composer).toHaveValue(DRAFT)
    await expect(card).toBeVisible()

    await expect.poll(() => opened.length).toBeGreaterThan(0)
    expect(watch.pageErrors).toEqual([])
  })
})
