import { expect, test, type Locator, type Page, type Route, type TestInfo } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { askedWords, chatPosts, lineOf, stubChatStream, type ChatScript } from '../support/chat-stream'
import { CLASSIC_STYLE, STUDIO_STYLE, chooseCoreStyle, type CoreStyle } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-268 §6.12.4, US-W03c (@host): W3's web surfaces in a real browser, under Studio light, Studio dark and Classic.
//
// The host draws each in the core chat, with no installed theme: thread-cards.tsx draws, under an assistant message,
// its calibration card (a `calibration@1` part), its follow-up line (a `follow_up@1` part, read against the caller's
// own list) and its reply chips (a `reply_chips@1` part, while it is the chat's newest message), and
// composer-with-suggestion.tsx draws the consent line above the composer while the server offers calibration.
// Classic draws the same chat (components/chatbot/chat.tsx), so every look is rendered here.
//
// The backend answers are the spec's: the conversation list holds one private chat, which /chat opens on a cold
// start (app/chat/page.tsx), and its messages, the calibration view (CALIBRATION_ENABLED ships off, so the backend
// itself offers none), the follow-up list and its cancel are answered here and recorded. The day's brief is off
// (brief_off) and the Suggested list is empty. A tapped chip goes through the app's own POST /api/chat, answered in
// the page with one recorded turn (support/chat-stream.ts); no model is called. Under each look the spec checks
// Auto's words carry data-ai-generated and no control does, attaches a screenshot of each surface, and then: a chip
// tap sends its text as the person's message and the chips go; Cancel cancels the follow-up; Go ahead answers the
// consent line once, the line goes and the calibration Auto posted into this chat arrives as its card.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const OK = 200
const NOT_FOUND = 404
const STUDIO_CLASS = /(^|\s)studio(\s|$)/
const DARK_CLASS = /(^|\s)dark(\s|$)/
/** next-themes' stored tone (components/providers.tsx storageKey). */
const TONE_KEY = 'automatos-theme'
/** Classic's "Auto now" aside (app/chat/page.tsx useAutoNowOpen): closed, so it lies over none of the chat's right edge. */
const CLASSIC_AUTO_NOW_KEY = 'classicChatAutoNowOpen'
const CLOSED = '0'
const AI_WORDS = '[data-ai-generated="true"]'
const AI_CAPTION = '[data-proactive-ai-caption]'
/** components/chatbot/message-with-receipts.tsx: the block that holds an assistant message's own words. */
const ASSISTANT_MESSAGE_TEST_ID = 'ai-generated-message'
/** components/proactive/follow-up-line.tsx followUpTime: HH:MM in the viewer's own locale and zone. */
const TIME_OPTIONS: Intl.DateTimeFormatOptions = { hour: '2-digit', minute: '2-digit' }
const FOLLOW_UP_IN_MS = 15 * 60_000
const SHOT = { animations: 'disabled' } as const

const CHAT_ID = '00000000-0000-4000-8000-0000000b0301'
const ASK_ID = '00000000-0000-4000-8000-0000000b0302'
const REPLY_ID = '00000000-0000-4000-8000-0000000b0303'
const CALIBRATION_MESSAGE_ID = '00000000-0000-4000-8000-0000000b0304'
const CALIBRATION_ID = '00000000-0000-4000-8000-0000000b0305'
const FOLLOW_UP_ID = 4301

const HISTORY_PATH = '/api/chat/history'
const THREAD_PATH = `/api/chat/${CHAT_ID}`
const MESSAGES_PATH = `${THREAD_PATH}/messages`
const CALIBRATION_PATH = '/api/auto/calibration'
const PROACTIVE_PREFIX = '/api/proactive/'
const FOLLOW_UPS_PATH = '/api/proactive/follow-ups'
const CANCEL_PATH = `${FOLLOW_UPS_PATH}/${FOLLOW_UP_ID}/cancel`
const BRIEF_OPEN_PATH = '/api/proactive/brief/open'
/** api/proactive_brief.py's refusal where the workspace's proactive mode is off, the answer brief-on-open.tsx expects. */
const BRIEF_OFF = { code: 'brief_off', detail: 'The brief is off in this workspace.' }
const NOT_EXPECTED = { code: 'e2e_not_expected', detail: 'Rendering calls no answer route.' }
const CONFLICT = 409

const CHAT_STREAM_PATH = '/api/chat'
const POSTS_KEY = '__proactiveW3ChatPosts'

const POSTED_AT = '2026-10-31T09:00:00Z'
const ASK_TEXT = 'Book the Lisbon flights for Thursday.'
const REPLY_TEXT = "Window or aisle for the Lisbon flights? I'll check back on the fare in 15 minutes."
const PLAIN_TEXT = 'The Lisbon flights are booked for Thursday.'
const CHIPS = ['Window', 'Aisle'] as const
const CHIP_ANSWER = 'Window it is. Seat 4A is held for you.'
/** Model-written in the person's chat: the follow-up line never draws it. */
const FOLLOW_UP_REASON = 'whether the Lisbon fare dropped below 180 euros'
/** services/calibration/strings.py, en: the server's consent line. */
const CONSENT_LINE = 'Auto reads the titles of your recent work to say what it knows.'
const CALIBRATION_TITLE = 'What Auto knows so far'

const CALIBRATION = {
  v: 1,
  id: CALIBRATION_ID,
  lines: [
    {
      id: 'line-1',
      text: 'You plan trips for a small team.',
      sources: [
        { type: 'deliverable', id: 'e2e-deliverable-1', label: 'Lisbon offsite plan' },
        { type: 'deliverable', id: 'e2e-deliverable-2', label: 'Porto offsite plan' },
      ],
    },
    { id: 'line-2', text: 'You write short, plain updates.', sources: [{ type: 'conversation', id: 'e2e-chat-1', label: 'Weekly update' }] },
  ],
  offers: ['Draft the monthly update', 'Compare flight options', 'Summarise a document'],
  empty: false,
}
const SOURCE_COUNTS = ['from 2 Deliverables', 'from 1 Conversation']

/** The person's private chat, as GET /api/chat/history lists it. */
const CHAT = {
  id: CHAT_ID,
  userId: 1,
  title: 'Lisbon trip',
  createdAt: POSTED_AT,
  updatedAt: POSTED_AT,
  visibility: 'private',
  lastContext: {},
  lastMessagePreview: REPLY_TEXT,
  kind: 'user',
}

const ASK = { id: ASK_ID, role: 'user', parts: [{ type: 'text', text: ASK_TEXT }], attachments: [], createdAt: POSTED_AT }
const PLAIN_REPLY = { id: REPLY_ID, role: 'assistant', parts: [{ type: 'text', text: PLAIN_TEXT }], attachments: [], createdAt: POSTED_AT }
/** The calibration Auto posted on the person's Go ahead (services/calibration/offer.py): its text part, then its card. */
const CALIBRATION_MESSAGE = {
  id: CALIBRATION_MESSAGE_ID,
  role: 'assistant',
  parts: [
    { type: 'text', text: 'Here is what I think I know about your work so far, from the titles of your recent work:' },
    { type: 'calibration@1', calibration: CALIBRATION },
  ],
  attachments: [],
  source: { origin: 'calibration', calibration_id: CALIBRATION_ID },
  createdAt: POSTED_AT,
}

/** The reply whose turn offered the chips and scheduled one follow-up, its parts after its text (US-C12c). */
function replyWithTurnParts(at: string) {
  return {
    id: REPLY_ID,
    role: 'assistant',
    parts: [
      { type: 'text', text: REPLY_TEXT },
      { type: 'reply_chips@1', replies: [...CHIPS] },
      { type: 'follow_up@1', follow_up_id: FOLLOW_UP_ID, at },
    ],
    attachments: [],
    createdAt: POSTED_AT,
  }
}

interface Look {
  readonly name: 'studio-light' | 'studio-dark' | 'classic'
  readonly label: string
  readonly core: CoreStyle
  readonly tone: 'light' | 'dark'
}

const LOOKS: readonly Look[] = [
  { name: 'studio-light', label: 'Studio light', core: STUDIO_STYLE, tone: 'light' },
  { name: 'studio-dark', label: 'Studio dark', core: STUDIO_STYLE, tone: 'dark' },
  { name: 'classic', label: 'Classic', core: CLASSIC_STYLE, tone: 'light' },
]

/** The chat's messages before the person answers the consent line, and once Go ahead has posted the calibration. */
interface Scene {
  readonly before: readonly unknown[]
  readonly after: readonly unknown[]
  readonly consented: boolean
  /** The newest assistant message's words: on screen once the chat has loaded. */
  readonly ready: string
}

/** What the spec's backend holds, and what the page sent it. */
interface Backend {
  consented: boolean
  cancelled: boolean
  readonly at: string
  readonly consents: unknown[]
  readonly cancels: string[]
  readonly answers: string[]
}

function onPath(pathname: string): (url: URL) => boolean {
  return (url) => url.pathname === pathname
}

/** GET /api/auto/calibration's answer (api/auto_calibration.py _read_view): offered until Go ahead, then open. */
function calibrationView(consented: boolean) {
  return {
    available: true,
    offer: !consented,
    consent_line: CONSENT_LINE,
    status: consented ? 'open' : null,
    calibration: consented ? CALIBRATION : null,
    rejected_lines: [],
    accepted_lines: [],
  }
}

/** One of the caller's follow-ups, as GET /api/proactive/follow-ups lists it (services/proactive/follow_ups.py). */
function followUpRow(at: string, cancelled: boolean) {
  return { id: FOLLOW_UP_ID, at, reason: FOLLOW_UP_REASON, status: cancelled ? 'cancelled' : 'pending', chat_id: CHAT_ID }
}

/** The consent POST posts the calibration into this chat (offer.py Offered.to_dict); GET reads the view. */
async function answerCalibration(route: Route, backend: Backend): Promise<void> {
  const request = route.request()
  if (request.method() !== 'POST') return fulfillBackendJson(route, OK, calibrationView(backend.consented))
  backend.consents.push(request.postDataJSON())
  backend.consented = true
  return fulfillBackendJson(route, OK, { status: 'open', calibration: CALIBRATION, chat_id: CHAT_ID, message_id: CALIBRATION_MESSAGE_ID })
}

/** Reject and Sounds right: no surface sends either on render, so each is recorded and refused. */
async function recordAnswer(route: Route, backend: Backend): Promise<void> {
  const request = route.request()
  if (request.method() === 'POST') backend.answers.push(new URL(request.url()).pathname)
  return fulfillBackendJson(route, CONFLICT, NOT_EXPECTED)
}

/** The follow-up list and its cancel; the brief is off and the Suggested list is empty. */
async function answerProactive(route: Route, backend: Backend): Promise<void> {
  const request = route.request()
  const { pathname } = new URL(request.url())
  if (pathname === CANCEL_PATH && request.method() === 'POST') {
    backend.cancels.push(pathname)
    backend.cancelled = true
    return fulfillBackendJson(route, OK, { follow_up: followUpRow(backend.at, true) })
  }
  if (pathname === FOLLOW_UPS_PATH) return fulfillBackendJson(route, OK, { follow_ups: [followUpRow(backend.at, backend.cancelled)] })
  if (pathname === BRIEF_OPEN_PATH) return fulfillBackendJson(route, NOT_FOUND, BRIEF_OFF)
  return fulfillBackendJson(route, OK, { suggestions: [] })
}

/** Answers the conversation list, the chat and its messages, calibration and the proactive routes. */
async function answerTheBackend(page: Page, scene: Scene, at: string): Promise<Backend> {
  const backend: Backend = { consented: scene.consented, cancelled: false, at, consents: [], cancels: [], answers: [] }
  await page.route(onPath(HISTORY_PATH), (route) => fulfillBackendJson(route, OK, [CHAT]))
  await page.route(onPath(THREAD_PATH), (route) => fulfillBackendJson(route, OK, { ...CHAT, turnInFlight: false }))
  await page.route(onPath(MESSAGES_PATH), (route) => fulfillBackendJson(route, OK, backend.consented ? scene.after : scene.before))
  await page.route(onPath(CALIBRATION_PATH), (route) => answerCalibration(route, backend))
  await page.route((url) => url.pathname.startsWith(`${CALIBRATION_PATH}/`), (route) => recordAnswer(route, backend))
  await page.route((url) => url.pathname.startsWith(PROACTIVE_PREFIX), (route) => answerProactive(route, backend))
  return backend
}

async function expectLook(page: Page, look: Look): Promise<void> {
  const html = page.locator('html')
  if (look.core.studio) await expect(html).toHaveClass(STUDIO_CLASS)
  else await expect(html).not.toHaveClass(STUDIO_CLASS)
  if (look.tone === 'dark') await expect(html).toHaveClass(DARK_CLASS)
  else await expect(html).not.toHaveClass(DARK_CLASS)
}

/** The assistant message holding `words`, marked as Auto's own (PRD-264 EUAI-09). */
async function expectAutosMessage(page: Page, words: string): Promise<Locator> {
  const message = page.getByTestId(ASSISTANT_MESSAGE_TEST_ID).filter({ hasText: words })
  await expect(message).toBeVisible()
  await expect(message).toHaveAttribute('data-ai-generated', 'true')
  return message
}

/** Opens /chat under `look`, on the chat's cold start, once its newest reply is on screen. */
async function openTheChat(page: Page, look: Look, scene: Scene): Promise<void> {
  await page.addInitScript(({ toneKey, tone, asideKey, closed }) => {
    window.localStorage.setItem(toneKey, tone)
    window.localStorage.setItem(asideKey, closed)
  }, { toneKey: TONE_KEY, tone: look.tone, asideKey: CLASSIC_AUTO_NOW_KEY, closed: CLOSED })
  await chooseCoreStyle(page, look.core, APP_URL)
  await page.goto(CHAT_PATH)
  await expectLook(page, look)
  await expectAutosMessage(page, scene.ready)
  await page.evaluate(() => document.fonts.ready.then(() => undefined))
}

/** Each of `words` sits inside Auto's data-ai-generated words, and no control in `box` carries the mark. */
async function expectAutosWords(box: Locator, words: readonly string[]): Promise<void> {
  for (const text of words) await expect(box.locator(AI_WORDS).filter({ hasText: text }).first()).toBeVisible()
  await expect(box.locator(`button${AI_WORDS}`)).toHaveCount(0)
  await expect(box.locator(AI_WORDS).getByRole('button')).toHaveCount(0)
}

async function attachShot(box: Locator, name: string, look: Look, testInfo: TestInfo): Promise<void> {
  await testInfo.attach(`${look.name}-${name}.png`, { body: await box.screenshot(SHOT), contentType: 'image/png' })
}

/** The fire time as the follow-up line writes it, its spaces collapsed as Playwright reads text. */
async function viewerTime(page: Page, at: string): Promise<string> {
  const time = await page.evaluate(({ iso, options }) => new Date(iso).toLocaleTimeString(undefined, options), { iso: at, options: TIME_OPTIONS })
  return time.replace(/\s+/g, ' ')
}

/** The card under the calibration message: its lines with their source counts and Reject, its offers, its answers. */
async function expectCalibrationCard(page: Page, look: Look, testInfo: TestInfo): Promise<void> {
  const box = page.getByTestId('thread-calibration')
  const card = box.getByRole('article', { name: CALIBRATION_TITLE })
  await expect(card).toBeVisible()
  await expect(card.locator('[data-line-id]')).toHaveCount(CALIBRATION.lines.length)
  await expectAutosWords(card, [...CALIBRATION.lines.map((line) => line.text), ...CALIBRATION.offers])
  for (const count of SOURCE_COUNTS) await expect(card.getByText(count, { exact: true })).toBeVisible()
  for (const line of CALIBRATION.lines) await expect(card.getByRole('button', { name: `Reject: ${line.text}`, exact: true })).toBeEnabled()
  await expect(card.getByRole('button', { name: 'Sounds right', exact: true })).toBeEnabled()
  await expect(card.getByRole('button', { name: "Something's off", exact: true })).toBeEnabled()
  await expect(card.locator(AI_CAPTION)).toHaveCount(1)
  await expect(card.locator(AI_WORDS).locator(AI_CAPTION)).toHaveCount(0)
  await expect(card).not.toContainText('%')
  await attachShot(box, 'calibration-card', look, testInfo)
}

/** The line under the reply that scheduled the follow-up: its time in the viewer's zone and Cancel, never the reason. */
async function expectFollowUpLine(page: Page, at: string, look: Look, testInfo: TestInfo): Promise<Locator> {
  const box = page.getByTestId('follow-up-lines')
  const line = box.getByTestId('follow-up-line')
  await expect(line).toHaveCount(1)
  await expect(line).toHaveAttribute('data-follow-up-status', 'pending')
  await expect(line).toContainText(`Follow-up at ${await viewerTime(page, at)}`)
  await expect(line.getByRole('button', { name: /^Cancel the follow-up at / })).toBeEnabled()
  await expect(box.locator(AI_WORDS)).toHaveCount(0)
  await expect(page.getByText(FOLLOW_UP_REASON)).toHaveCount(0)
  await attachShot(box, 'follow-up-line', look, testInfo)
  return line
}

/** The chips once, below the newest reply, each named "Reply: <chip>" with Auto's words inside. */
async function expectChips(page: Page, look: Look, testInfo: TestInfo): Promise<Locator> {
  const chips = page.getByRole('group', { name: 'Replies', exact: true })
  await expect(chips).toHaveCount(1)
  await expect(chips.getByRole('button', { name: /^Reply: / })).toHaveCount(CHIPS.length)
  for (const chip of CHIPS) await expect(chips.getByRole('button', { name: `Reply: ${chip}`, exact: true })).toBeEnabled()
  await expectAutosWords(chips, CHIPS)
  await attachShot(chips, 'reply-chips', look, testInfo)
  // Both boxes read in one frame, after the screenshot has run the chips' entrance to its end.
  const handle = await chips.elementHandle()
  const gap = await (await expectAutosMessage(page, REPLY_TEXT)).evaluate(
    (reply, below) => (below ? below.getBoundingClientRect().top - reply.getBoundingClientRect().bottom : Number.NaN),
    handle,
  )
  expect(gap).toBeGreaterThanOrEqual(-1)
  return chips
}

/** Cancel posts the follow-up's cancel once, and the line says it was cancelled. */
async function cancelTheFollowUp(line: Locator, backend: Backend): Promise<void> {
  await line.getByRole('button', { name: /^Cancel the follow-up at / }).click()
  await expect.poll(() => backend.cancels).toEqual([CANCEL_PATH])
  await expect(line).toHaveAttribute('data-follow-up-status', 'cancelled')
  await expect(line).toContainText('Cancelled')
  await expect(line.getByRole('button')).toHaveCount(0)
}

/** A tap sends the chip's own words as the person's message in this chat, and the chips are gone after. */
async function tapAChip(page: Page, chips: Locator): Promise<void> {
  await chips.getByRole('button', { name: `Reply: ${CHIPS[0]}`, exact: true }).click()
  await expect.poll(async () => (await chatPosts(page, POSTS_KEY)).map(askedWords)).toEqual([CHIPS[0]])
  expect((await chatPosts(page, POSTS_KEY))[0]?.id).toBe(CHAT_ID)
  await expectAutosMessage(page, CHIP_ANSWER)
  await expect(chips).toHaveCount(0)
}

/** The consent line above the composer: the server's words (not Auto's), Go ahead and Not now, nothing sent yet. */
async function expectConsentLine(page: Page, backend: Backend, look: Look, testInfo: TestInfo): Promise<Locator> {
  const line = page.getByTestId('calibration-offer')
  await expect(line).toHaveCount(1)
  await expect(line).toHaveAccessibleName(CONSENT_LINE)
  await expect(line).toContainText(CONSENT_LINE)
  await expect(line.getByRole('button', { name: 'Go ahead', exact: true })).toBeEnabled()
  await expect(line.getByRole('button', { name: 'Not now', exact: true })).toBeEnabled()
  await expect(line.locator(AI_WORDS)).toHaveCount(0)
  expect(backend.consents).toEqual([])
  await attachShot(line, 'consent-line', look, testInfo)
  return line
}

/** Go ahead answers once with this chat's id; the line goes and the calibration arrives in this chat as its card. */
async function goAhead(page: Page, line: Locator, backend: Backend): Promise<void> {
  await line.getByRole('button', { name: 'Go ahead', exact: true }).click()
  await expect.poll(() => backend.consents.length).toBe(1)
  expect(backend.consents[0]).toMatchObject({ consent: true, chat_id: CHAT_ID })
  await expect(page.getByTestId('calibration-offer')).toHaveCount(0)
  const card = page.getByTestId('thread-calibration').getByRole('article', { name: CALIBRATION_TITLE })
  await expect(card).toBeVisible()
  await expect(card.getByRole('button', { name: 'Sounds right', exact: true })).toBeEnabled()
  expect(backend.consents).toHaveLength(1)
}

test.describe("W3's surfaces in every built-in look", () => {
  for (const look of LOOKS) {
    test(`under ${look.label}: the calibration card, the follow-up line with Cancel, and chips a tap sends`, { tag: '@host' }, async ({ page }, testInfo) => {
      const watch = watchPage(page)
      const at = new Date(Date.now() + FOLLOW_UP_IN_MS).toISOString()
      const thread = [CALIBRATION_MESSAGE, ASK, replyWithTurnParts(at)]
      const scene: Scene = { before: thread, after: thread, consented: true, ready: REPLY_TEXT }
      const turn = { lines: [lineOf({ text: CHIP_ANSWER }), lineOf({ data: { type: 'finish' } })], hold: false }
      const script: ChatScript = { path: CHAT_STREAM_PATH, postsKey: POSTS_KEY, turns: [turn] }
      await page.addInitScript(stubChatStream, script)
      const backend = await answerTheBackend(page, scene, at)
      await openTheChat(page, look, scene)

      await expectCalibrationCard(page, look, testInfo)
      const line = await expectFollowUpLine(page, at, look, testInfo)
      const chips = await expectChips(page, look, testInfo)
      await cancelTheFollowUp(line, backend)
      await tapAChip(page, chips)

      expect(backend.answers).toEqual([])
      expect(watch.pageErrors).toEqual([])
    })

    test(`under ${look.label}: the consent line, answered once, brings the calibration card`, { tag: '@host' }, async ({ page }, testInfo) => {
      const watch = watchPage(page)
      const scene: Scene = { before: [ASK, PLAIN_REPLY], after: [ASK, PLAIN_REPLY, CALIBRATION_MESSAGE], consented: false, ready: PLAIN_TEXT }
      const backend = await answerTheBackend(page, scene, new Date(Date.now() + FOLLOW_UP_IN_MS).toISOString())
      await openTheChat(page, look, scene)

      const line = await expectConsentLine(page, backend, look, testInfo)
      await goAhead(page, line, backend)

      expect(backend.answers).toEqual([])
      expect(watch.pageErrors).toEqual([])
    })
  }
})
