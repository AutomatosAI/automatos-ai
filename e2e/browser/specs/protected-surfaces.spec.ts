import fs from 'fs'
import path from 'path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { fulfillBackendJson } from '../support/backend-route'
import { chooseStyle, serveInstalledTheme, THEME_E2E_PATH } from '../support/hello-theme'
import { describeParity, openPlay } from '../support/play-parity'

// PRD-261 A-12c (@host): protected surfaces and the CSS layer model (PA-15).
//
// 1. Under an installed theme whose stylesheet attacks the approval cards (fixtures/hello-a12c/
//    theme.css: display:none, opacity:0, visibility:hidden, off-screen, pointer-events:none, and
//    translate, scale, rotate, offset-path, margin, mask, content-visibility, contain, zoom, size
//    and z-index on the cards' two protected elements, and visibility and clicks taken from every
//    ancestor), a tool approval card and a Mission approval card are still drawn, on screen, and
//    their Approve & run button still posts the decision. The cards arrive the way they do in the
//    app: a chat turn (recorded, fixtures/hello-a12c/chat-turns.json) carries the tool-data that
//    chat.tsx turns into the widget on the canvas. Each test starts on a new chat, with no protected
//    surface on the page until the turn draws its card: see startOnANewChat.
// 2. Under installed Play (the Design Kit's package, support/play-parity.ts; A-19f4: since W-A8
//    the only Play), the body takes Play's page background (the kit's
//    themes/play/src/app/play-theme.css, its `:where(html.play) body` rule), not Studio's
//    bg-background: theme CSS is unlayered, so a theme rule meets core's unlayered Tailwind output
//    on specificity and order. In a cascade layer it would lose to it outright. It needs the
//    package and stores the member's choice on the lane's one backend, so it is @package and runs
//    in the package step, one test at a time.
//
// This proves what CSS can reach. For code a theme runs inside the page it is a review rule, not
// a guarantee (PRD §4.4): see fixtures/hello-a12c/README.md.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const FIXTURE = 'hello-a12c'
const A12C_INSTALL_ID = '00000000-0000-4000-8000-0000000a12c0'
const TURNS_FILE = path.resolve(__dirname, '..', 'fixtures', FIXTURE, 'chat-turns.json')
const CHAT_PATH = '/chat'
const CHAT_STREAM_PATH = '/api/chat'
const CHAT_HISTORY_PATH = '/api/chat/history'
const NO_CONVERSATIONS: readonly unknown[] = []
const STATUS_ATTR = 'data-theme-host'
const THEME_STYLE_LINK = 'link[data-theme-host-style]'
const FRAME_TEST_ID = 'hello-a12c-frame'
const HOST_ATTR = 'data-protected-surface'
const INNER_SELECTOR = '[data-protected-surface-inner]'
const COMPOSER_PLACEHOLDER = 'Send a message...'
const ASK = 'Show me what needs my approval'
const APPROVE_BUTTON = 'Approve & run'
const STREAM_CONTENT_TYPE = 'text/plain; charset=utf-8'
const TEXT_LINE = '0:'
const DATA_LINE = 'd:'
const OK = 200

const PLAY_PAGE_BACKGROUND = '--pl-page-background'
const TRANSPARENT = 'rgba(0, 0, 0, 0)'
const SENTINEL = 'rgb(1, 2, 3)'

/** What each protected element keeps, computed, under the hostile sheet (CSS property → value). */
const KEPT: Readonly<Record<string, string>> = {
  display: 'block',
  visibility: 'visible',
  opacity: '1',
  'pointer-events': 'auto',
  position: 'static',
  transform: 'none',
  translate: 'none',
  scale: 'none',
  rotate: 'none',
  'offset-path': 'none',
  filter: 'none',
  'clip-path': 'none',
  'mask-image': 'none',
  'content-visibility': 'visible',
  contain: 'none',
  zoom: '1',
  'margin-top': '0px',
  'margin-left': '0px',
  'z-index': 'auto',
}
/** What the sheet takes from every ancestor of a card: proof its parent rule landed. */
const ANCESTOR_TAKEN = { visibility: 'hidden', pointerEvents: 'none' }
/** The properties the sheet's rules declare on each protected element: proof they matched. */
const ATTACKED = [
  'display', 'opacity', 'filter', 'visibility', 'position', 'left', 'top', 'transform', 'clip-path', 'pointer-events',
  'translate', 'scale', 'rotate', 'offset-path', 'margin-left', 'content-visibility', 'contain', 'zoom', 'width', 'height', 'z-index',
]
/** mask and -webkit-mask expand to mask-image in one engine and -webkit-mask-image in the other. */
const MASK_IMAGE = /(^|-)mask-image$/

type SurfaceName = 'tool-approval' | 'mission-approval'

interface CardCase {
  readonly surface: SurfaceName
  /** The decision the Approve & run button posts, on the backend's origin. */
  readonly decisionRoute: string
  readonly decisionBody: unknown
  /** The line the card shows once approved, or null when it closes itself (the Mission card). */
  readonly outcome: string | null
}

const CARDS: readonly CardCase[] = [
  {
    surface: 'tool-approval',
    decisionRoute: '**/api/v1/approval-grants/*/grant',
    decisionBody: { grant: { id: 12001, status: 'granted', details: {} } },
    outcome: 'Approved.',
  },
  {
    surface: 'mission-approval',
    decisionRoute: '**/api/missions/*/approve',
    decisionBody: { id: 'e2e-a12c-mission', state: 'executing' },
    outcome: null,
  },
]

interface RecordedTurn {
  readonly text: string
  readonly events: readonly unknown[]
}

function isRecordedTurn(value: unknown): value is RecordedTurn {
  const turn = value as Partial<RecordedTurn> | null
  return typeof turn === 'object' && turn !== null && typeof turn.text === 'string' && Array.isArray(turn.events)
}

function readTurn(surface: SurfaceName): RecordedTurn {
  const turns = JSON.parse(fs.readFileSync(TURNS_FILE, 'utf8')) as Record<string, unknown>
  const turn = turns[surface]
  if (!isRecordedTurn(turn)) throw new Error(`${TURNS_FILE} has no recorded turn for ${surface}`)
  return turn
}

/** The turn in the stream format lib/chat/hooks.ts reads: one text line, then the data lines. */
function streamBody(turn: RecordedTurn): string {
  const lines = [`${TEXT_LINE}${JSON.stringify(turn.text)}`, ...turn.events.map((event) => `${DATA_LINE}${JSON.stringify(event)}`)]
  return `${lines.join('\n')}\n`
}

/** Answers the app's own POST /api/chat with the recorded turn (no model is called). */
async function answerChatWith(page: Page, turn: RecordedTurn): Promise<void> {
  await page.route((url) => url.pathname === CHAT_STREAM_PATH, async (route) => {
    if (route.request().method() !== 'POST') {
      await route.continue()
      return
    }
    await route.fulfill({ status: OK, contentType: STREAM_CONTENT_TYPE, headers: { 'cache-control': 'no-store' }, body: streamBody(turn) })
  })
}

/** Answers the card's decision and records the path of every POST of it. */
async function recordDecisions(page: Page, card: CardCase): Promise<string[]> {
  const decided: string[] = []
  await page.route(card.decisionRoute, async (route) => {
    if (route.request().method() === 'POST') decided.push(new URL(route.request().url()).pathname)
    await fulfillBackendJson(route, OK, card.decisionBody)
  })
  return decided
}

/**
 * The precondition, stated: /chat opens on a new chat. A fresh browser's /chat lands on the newest conversation
 * (PRD-237's cold start, app/chat/page.tsx). Since PRD-268 W2 the lane's one shared workspace has one: the first
 * page any spec loads opens the day's brief (BriefOnOpen), which posts it into a new Auto thread, and that thread's
 * brief card carries Listen in the protected surface `brief-voice` (US-C21d). The hostile sheet's `:has` rule
 * takes visibility from every ancestor of any protected surface, so landing there would leave Listen alone on the
 * page, with no composer to ask from, before the turn's card is drawn. Answering the conversation list with none
 * keeps the page as the empty workspace left it. specs/proactive-brief-thread.spec.ts holds the Auto thread itself
 * (its brief card beside the composer) under an installed theme that attacks nothing.
 */
async function startOnANewChat(page: Page): Promise<void> {
  await page.route((url) => url.pathname === CHAT_HISTORY_PATH, (route) => fulfillBackendJson(route, OK, NO_CONVERSATIONS))
}

/** Opens the chat on a new chat under the hostile theme, once its stylesheet is on the page. */
async function openChatUnderHostileTheme(page: Page): Promise<void> {
  await startOnANewChat(page)
  const served = await serveInstalledTheme(page, { installId: A12C_INSTALL_ID, fixture: FIXTURE })
  await chooseStyle(page, served.style, APP_URL)
  await page.goto(CHAT_PATH)
  await expect(page.locator('html')).toHaveAttribute(STATUS_ATTR, 'active')
  await expect(page.getByTestId(FRAME_TEST_ID)).toBeVisible()
  await expect(page.locator(THEME_STYLE_LINK)).toHaveCount(1)
}

async function ask(page: Page): Promise<void> {
  const composer = page.getByPlaceholder(COMPOSER_PLACEHOLDER).filter({ visible: true }).first()
  await composer.fill(ASK)
  await composer.press('Enter')
}

function keptStyle(element: Locator): Promise<Record<string, string>> {
  return element.evaluate((node, names) => {
    const style = getComputedStyle(node)
    return Object.fromEntries(names.map((name) => [name, style.getPropertyValue(name)]))
  }, Object.keys(KEPT))
}

function parentStyle(element: Locator): Promise<typeof ANCESTOR_TAKEN | null> {
  return element.evaluate((node) => {
    const parent = node.parentElement
    if (parent === null) return null
    const style = getComputedStyle(parent)
    return { visibility: style.visibility, pointerEvents: style.pointerEvents }
  })
}

/** Every property the theme sheet's style rules declare on the element (the rules it matches). */
function attackedProperties(element: Locator): Promise<string[]> {
  return element.evaluate((node, linkSelector) => {
    const link = document.querySelector<HTMLLinkElement>(linkSelector)
    const rules = Array.from<CSSRule>(link?.sheet?.cssRules ?? [])
    const matching = rules.filter((rule): rule is CSSStyleRule => rule instanceof CSSStyleRule && node.matches(rule.selectorText))
    return matching.flatMap((rule) => Array.from(rule.style))
  }, THEME_STYLE_LINK)
}

/** At least part of the element is inside the viewport. */
async function expectOnScreen(page: Page, element: Locator): Promise<void> {
  const box = await element.boundingBox()
  const viewport = page.viewportSize()
  expect(box).not.toBeNull()
  expect(viewport).not.toBeNull()
  if (box === null || viewport === null) return
  const onScreen = box.x + box.width > 0 && box.y + box.height > 0 && box.x < viewport.width && box.y < viewport.height
  expect({ box, onScreen }).toEqual({ box, onScreen: true })
}

/** The card's host and inner wrapper were attacked by every rule, and kept every guarded value. */
async function expectCardStands(page: Page, card: Locator): Promise<void> {
  const inner = card.locator(INNER_SELECTOR)
  await expect(card).toBeVisible()
  await expect(inner).toBeVisible()
  for (const element of [card, inner]) {
    const attacked = await attackedProperties(element)
    expect(attacked).toEqual(expect.arrayContaining(ATTACKED))
    expect(attacked.some((name) => MASK_IMAGE.test(name))).toBe(true)
    expect(await keptStyle(element)).toEqual(KEPT)
  }
  expect(await parentStyle(card)).toEqual(ANCESTOR_TAKEN)
  await expectOnScreen(page, card)
}

for (const card of CARDS) {
  test(`the ${card.surface} card stays visible and clickable under a hostile theme stylesheet`, { tag: '@host' }, async ({ page }) => {
    await answerChatWith(page, readTurn(card.surface))
    const decided = await recordDecisions(page, card)
    await openChatUnderHostileTheme(page)
    await ask(page)

    const surface = page.locator(`[${HOST_ATTR}="${card.surface}"]`)
    await expect(surface).toHaveCount(1)
    await expectCardStands(page, surface)

    // A real click: Playwright clicks only an element that is visible, stable, and the one the
    // browser's hit test finds at the click point.
    const approve = surface.getByRole('button', { name: APPROVE_BUTTON })
    await expectOnScreen(page, approve)
    await approve.click()
    await expect.poll(() => decided.length).toBe(1)
    if (card.outcome === null) await expect(surface).toHaveCount(0)
    else await expect(surface.getByRole('status')).toHaveText(card.outcome)
  })
}

/** The computed background colour of the body, and of a probe painted with `var(<name>)`. */
function backgrounds(page: Page, variable: string): Promise<{ body: string; variable: string }> {
  return page.evaluate((name) => {
    const probe = document.createElement('div')
    probe.style.backgroundColor = `var(${name})`
    document.body.appendChild(probe)
    const resolved = getComputedStyle(probe).backgroundColor
    probe.remove()
    return { body: getComputedStyle(document.body).backgroundColor, variable: resolved }
  }, variable)
}

describeParity('the CSS layer model under installed Play', (variant) => {
  test("under installed Play, the body takes Play's page background, not Studio's bg-background", { tag: '@package' }, async ({ page, request }) => {
    await openPlay(page, request, variant, THEME_E2E_PATH)

    // Play's token is defined (not the transparent of an unset variable) and the body wears it.
    await expect
      .poll(async () => {
        const seen = await backgrounds(page, PLAY_PAGE_BACKGROUND)
        return seen.variable !== TRANSPARENT && seen.body === seen.variable
      })
      .toBe(true)

    // The look writes Studio's --background from the same ground, so the colours alone could
    // agree by design. A sentinel on Play's token, which core's body rule never reads, shows which
    // rule paints the body: it follows the sentinel only if Play's rule wins the cascade.
    await page.evaluate(
      ({ name, value }) => document.documentElement.style.setProperty(name, value),
      { name: PLAY_PAGE_BACKGROUND, value: SENTINEL },
    )
    await expect.poll(async () => (await backgrounds(page, PLAY_PAGE_BACKGROUND)).body).toBe(SENTINEL)
  })
})
