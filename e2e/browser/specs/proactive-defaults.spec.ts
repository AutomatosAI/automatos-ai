import { expect, test, type Page, type TestInfo } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, type InstalledTheme } from '../support/hello-theme'
import { watchPage } from '../support/watch'

// PRD-268 US-C21f (@host): the proactive defaults in a real browser, under Studio light and Studio dark.
//
// The proactive hello theme (fixtures/hello-proactive) places every SDK 1.3.0 proactive default on its
// route /x/hello-proactive with fixture data: BriefCard, SuggestionCard (the card and the pinned line),
// StatusLabel, ReadinessList, CalibrationCard and ReplyChips. Under each tone the spec checks each one
// draws the host's words, glyphs and controls, that Auto's words carry data-ai-generated and each card
// holding them shows the "Auto is an AI" caption once, and attaches a screenshot of each to the report.
//
// The backend answers are the spec's. Listen reads the brief the host holds, so the spec tells the host
// where the day's brief is (as an open would have: the workspace's key in localStorage, written before
// the second load) and answers that message with the fixture's brief. The proactive and readiness
// routes, the brief's feedback and the server voice are answered here and recorded, never by the
// backend: no default makes the call a press makes (Start, Hide, Check again, the thumbs, Listen) on
// render, and nothing plays.
//
// Classic is not rendered here: the host runs every installed theme on Studio (lib/ui-style.ts
// rendersStudio; lib/theme-host/manifest.ts accepts only base "studio"), and under Classic it serves
// no theme route (app/x/[...path]/page.tsx sends the member to the chat). No Classic surface places
// these defaults before US-W01a, US-W01b, US-W02, US-W04 and US-W05. axe is the owner's call (K18).

const HELLO_PROACTIVE: InstalledTheme = { installId: '00000000-0000-4000-8000-0000000c21f0', fixture: 'hello-proactive' }
const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const ROUTE_PATH = '/x/hello-proactive'
const STATUS_ATTR = 'data-theme-host'
const THEME_CLASS = /(^|\s)hello-proactive(\s|$)/
const STUDIO_CLASS = /(^|\s)studio(\s|$)/
const DARK_CLASS = /(^|\s)dark(\s|$)/
const AI_WORDS = '[data-ai-generated="true"]'
const AI_CAPTION = '[data-proactive-ai-caption]'
/** lib/ai-label/copy.ts aiLabelText(null). */
const AI_CAPTION_TEXT = 'Auto is an AI'
/** next-themes' stored tone (components/providers.tsx storageKey). */
const TONE_KEY = 'automatos-theme'
/** The workspace WorkspaceProvider last loaded (components/workspace-provider.tsx rememberWorkspace). */
const WORKSPACE_KEY = 'last_active_workspace'
/** Where the day's brief is, per workspace and local date (lib/proactive/brief-day.ts). */
const BRIEF_KEY_PREFIX = 'automatos:brief:'
const BRIEF_REF = { chat_id: '00000000-0000-4000-8000-0000000c21f2', message_id: '00000000-0000-4000-8000-0000000c21f3' }
const BRIEF_MESSAGES_PATH = `/api/chat/${BRIEF_REF.chat_id}/messages`
/** What Listen reads: the brief the host holds, as its message carries it. */
const HELD_BRIEF = {
  v: 1,
  date: '2026-10-30',
  greeting: 'Morning, Sam.',
  prose: 'One thing needs you before noon.',
  spoken: 'Morning, Sam. 1. Approve the Lisbon flights.',
  sections: { needs_you: [], suggestions: [], today: [], since: [], unchecked: [] },
  counts: { needs_you: 0, suggestions: 0, today: 0, since: 0 },
  labels: { sections: {}, more: {} },
  state_hash: 'e2e-proactive-held',
  generated_at: '2026-10-30T07:00:00Z',
}
const BRIEF_MESSAGES = [
  { id: BRIEF_REF.message_id, role: 'assistant', parts: [{ type: 'text', text: HELD_BRIEF.prose }, { type: 'brief@1', brief: HELD_BRIEF }] },
]
/** The proactive routes, the readiness routes, the brief's feedback and the server voice: answered here, never by the backend. */
const PROACTIVE_ROUTES = /\/api\/(proactive|readiness|voice\/speech|activity\/digest\/feedback)(\/|\?|$)/
const PROACTIVE_REFUSAL = { code: 'e2e_not_expected', detail: 'Rendering a default calls no route.' }
/** What a default calls once pressed (Start, Hide, Check again, the thumbs, Listen): none is called on render. */
const PRESS_CALLS = /^POST \/api\/(proactive\/suggestions\/[^/]+\/(start|hide)|readiness\/check|activity\/digest\/feedback|voice\/speech)$/
const CONFLICT = 409
const OK = 200
const SHARING_FAILURE = /Invalid hook call|Minified React error|not in this SDK build|Automatos theme SDK:/i
const LISTEN = 'Listen to the brief'

interface Tone {
  readonly name: 'studio-light' | 'studio-dark'
  readonly label: string
  readonly stored: 'light' | 'dark'
}

const TONES: readonly Tone[] = [
  { name: 'studio-light', label: 'Studio light', stored: 'light' },
  { name: 'studio-dark', label: 'Studio dark', stored: 'dark' },
]

/** A placed default: its box on the route, the words of Auto's it holds, and its AI captions. */
interface Placed {
  readonly name: string
  readonly testId: string
  readonly aiWords: readonly string[]
  readonly captions: number
}

const PLACED: readonly Placed[] = [
  {
    name: 'brief-card',
    testId: 'hello-proactive-brief',
    aiWords: ['Morning, Sam.', 'the Lisbon flights hold lapses then', 'Approve the Lisbon flights', 'Weekly metrics report'],
    captions: 1,
  },
  {
    name: 'suggestion-card',
    testId: 'hello-proactive-suggestion',
    aiWords: ['Draft the October update', 'the last one went out in September'],
    captions: 1,
  },
  { name: 'suggestion-pinned', testId: 'hello-proactive-pinned', aiWords: ['Draft the October update'], captions: 1 },
  { name: 'status-label', testId: 'hello-proactive-status', aiWords: [], captions: 0 },
  { name: 'readiness-list', testId: 'hello-proactive-readiness', aiWords: [], captions: 0 },
  {
    name: 'calibration-card',
    testId: 'hello-proactive-calibration',
    aiWords: ['You write short, plain updates', 'Draft the monthly update'],
    captions: 1,
  },
  { name: 'reply-chips', testId: 'hello-proactive-chips', aiWords: ['Yes, book it', 'Not this week', 'Show me options'], captions: 0 },
]

/** Answers every proactive route with a refusal, and records each call (preflights aside) as `METHOD /path`. */
async function recordProactiveCalls(page: Page): Promise<string[]> {
  const called: string[] = []
  await page.route(PROACTIVE_ROUTES, async (route) => {
    const request = route.request()
    if (request.method() !== 'OPTIONS') called.push(`${request.method()} ${new URL(request.url()).pathname}`)
    await fulfillBackendJson(route, CONFLICT, PROACTIVE_REFUSAL)
  })
  return called
}

/** Answers the day's brief message with the held brief, and records each read of it. */
async function answerTheBrief(page: Page): Promise<string[]> {
  const read: string[] = []
  await page.route(`**${BRIEF_MESSAGES_PATH}`, async (route) => {
    if (route.request().method() === 'GET') read.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, BRIEF_MESSAGES)
  })
  return read
}

/** Before each load: the tone, and, once a load has named the workspace, where its brief is today. */
async function prepareStorage(page: Page, tone: Tone): Promise<void> {
  await page.addInitScript(({ toneKey, tone: stored, workspaceKey, prefix, ref }) => {
    window.localStorage.setItem(toneKey, stored)
    const workspace = window.localStorage.getItem(workspaceKey)
    if (!workspace) return
    const now = new Date()
    const pad = (part: number): string => String(part).padStart(2, '0')
    const day = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
    window.localStorage.setItem(`${prefix}${workspace}:${day}`, JSON.stringify(ref))
  }, { toneKey: TONE_KEY, tone: tone.stored, workspaceKey: WORKSPACE_KEY, prefix: BRIEF_KEY_PREFIX, ref: BRIEF_REF })
}

/** Opens the route twice: the first load names the workspace, the second holds the day's brief. */
async function openRoute(page: Page): Promise<void> {
  await page.goto(ROUTE_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect.poll(() => page.evaluate((key) => window.localStorage.getItem(key), WORKSPACE_KEY)).not.toBeNull()
  await page.goto(ROUTE_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByTestId('hello-proactive-route')).toBeVisible()
  await page.evaluate(() => document.fonts.ready.then(() => undefined))
}

async function expectTone(page: Page, tone: Tone): Promise<void> {
  const html = page.locator('html')
  await expect(html).toHaveClass(STUDIO_CLASS)
  await expect(html).toHaveClass(THEME_CLASS)
  if (tone.stored === 'dark') await expect(html).toHaveClass(DARK_CLASS)
  else await expect(html).not.toHaveClass(DARK_CLASS)
}

/** Auto's words inside data-ai-generated, the caption once where it belongs, then the screenshot. */
async function expectPlaced(page: Page, placed: Placed, tone: Tone, testInfo: TestInfo): Promise<void> {
  const box = page.getByTestId(placed.testId)
  await expect(box).toBeVisible()
  for (const words of placed.aiWords) await expect(box.locator(AI_WORDS).filter({ hasText: words }).first()).toBeVisible()
  await expect(box.locator(AI_CAPTION)).toHaveCount(placed.captions)
  if (placed.captions > 0) await expect(box.locator(AI_CAPTION)).toHaveText(AI_CAPTION_TEXT)
  await expect(box.locator(AI_WORDS).locator(AI_CAPTION)).toHaveCount(0)
  const shot = await box.screenshot({ animations: 'disabled' })
  await testInfo.attach(`${tone.name}-${placed.name}.png`, { body: shot, contentType: 'image/png' })
}

async function expectBriefCard(page: Page): Promise<void> {
  const card = page.getByTestId('hello-proactive-brief').getByRole('article', { name: 'Morning, Sam.' })
  await expect(card.locator('[data-section]')).toHaveCount(5)
  await expect(card.getByRole('button', { name: 'Answer →: Approve the Lisbon flights' })).toBeVisible()
  await expect(card.getByRole('button', { name: 'Show 2 more' })).toBeVisible()
  await expect(card.getByRole('group', { name: 'Rate this brief' }).getByRole('button')).toHaveCount(2)
  await expect(card.getByRole('button', { name: LISTEN })).toBeEnabled()
  await expect(card.locator('[data-brief-voice="root"]')).not.toHaveAttribute('data-voice-speaking', 'true')
}

async function expectSuggestionCards(page: Page): Promise<void> {
  for (const testId of ['hello-proactive-suggestion', 'hello-proactive-pinned']) {
    const box = page.getByTestId(testId)
    await expect(box.locator('[data-state="open"]')).toHaveCount(1)
    await expect(box.getByRole('button', { name: /^Start: Draft the October update/ })).toBeEnabled()
    await expect(box.getByRole('button', { name: 'Hide: Draft the October update' })).toBeEnabled()
  }
  await expect(page.getByTestId('hello-proactive-pinned').locator('[data-variant="pinned"]')).toHaveCount(1)
}

/** Each state's glyph and word in Studio's pill; Try again only on the failed row. */
async function expectStatusLabels(page: Page): Promise<void> {
  const box = page.getByTestId('hello-proactive-status')
  const words: readonly [string, string][] = [
    ['needs_you', '◆Needs you'],
    ['working', '●Working'],
    ['ready', '✓Ready to look at'],
    ['done', '✓Done'],
    ['failed', "✕Couldn't finish"],
  ]
  for (const [state, glyphAndWord] of words) {
    await expect(box.locator(`[data-display-state="${state}"] .studio-pill`)).toHaveText(glyphAndWord)
  }
  await expect(box.locator('[data-display-state="working"]')).toContainText("▸ Reading last week's numbers")
  await expect(box.getByRole('button', { name: /^Try again/ })).toHaveCount(1)
  await expect(box.getByRole('button', { name: 'Try again: Newsletter send' })).toBeVisible()
}

/** Each item reads as one sentence, the browser's row after the server's, and the fix is an in-app link. */
async function expectReadinessList(page: Page): Promise<void> {
  const box = page.getByTestId('hello-proactive-readiness')
  const rows = box.locator('[data-readiness-id]')
  await expect(rows).toHaveCount(4)
  await expect(rows.last()).toHaveAttribute('data-readiness-id', 'microphone')
  await expect(box.getByRole('listitem', { name: 'Server voice: needs setup. Set up voice.' })).toBeVisible()
  await expect(box.getByRole('listitem', { name: 'Phone alerts: not checked.' })).toBeVisible()
  await expect(box.getByRole('link', { name: 'Set up voice' })).toHaveAttribute('href', '/settings')
  await expect(box.getByRole('button', { name: 'Check again' })).toBeEnabled()
}

async function expectCalibrationAndChips(page: Page): Promise<void> {
  const card = page.getByTestId('hello-proactive-calibration').getByRole('article', { name: 'What Auto knows so far' })
  await expect(card.locator('[data-line-id]')).toHaveCount(3)
  await expect(card.getByRole('button', { name: 'Reject: Most of your customers are small hotels.' })).toBeEnabled()
  await expect(card.getByRole('button', { name: 'Sounds right' })).toBeEnabled()
  const chips = page.getByTestId('hello-proactive-chips').getByRole('group', { name: 'Replies' })
  await expect(chips.getByRole('button', { name: /^Reply: / })).toHaveCount(3)
}

test.describe('the proactive defaults in every built-in Studio tone', () => {
  for (const tone of TONES) {
    test(`a theme places each default under ${tone.label}, Auto's words marked, with a screenshot of each`, { tag: '@host' }, async ({ page }, testInfo) => {
      const watch = watchPage(page)
      const proactiveCalls = await recordProactiveCalls(page)
      const briefReads = await answerTheBrief(page)
      await prepareStorage(page, tone)
      const served = await serveInstalledTheme(page, HELLO_PROACTIVE)
      await chooseStyle(page, served.style, APP_URL)
      await openRoute(page)
      await expectTone(page, tone)

      for (const placed of PLACED) await expectPlaced(page, placed, tone, testInfo)
      await expectBriefCard(page)
      await expectSuggestionCards(page)
      await expectStatusLabels(page)
      await expectReadinessList(page)
      await expectCalibrationAndChips(page)

      expect(briefReads).toContain(BRIEF_MESSAGES_PATH)
      await expect(page.getByTestId('hello-proactive-heard')).toHaveText('waiting')
      expect(proactiveCalls.filter((call) => PRESS_CALLS.test(call))).toEqual([])
      expect(watch.pageErrors).toEqual([])
      expect(watch.consoleErrors.filter((text) => SHARING_FAILURE.test(text))).toEqual([])
    })
  }
})
