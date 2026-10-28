import { expect, test, type APIRequestContext, type ElementHandle, type Page } from '@playwright/test'
import { CAPTION_SELECTOR, copyState, recordCopyRoots, watchWarnings } from '../support/ai-label-copy'
import { noStoredConversations } from '../support/chat-stream'
import { chooseKitPackage, installKitPackage, resetKitPackage } from '../support/kit-package'
import { messageBox, sendButton, voiceButton } from '../support/play-chat'
import { THEME_LOAD_TIMEOUT_MS } from '../support/play-parity'
import { watchPage } from '../support/watch'

// PRD-261 A-19e1 (@package), PRD-264 EUAI-11a (EU AI Act Art. 50): installed Play's /chat carries the
// host's AI label, and no style can hide it: a style that hides the label brings the host's own copy of it,
// and Play's chat keeps running. Play is the Design Kit's package, uploaded, opted in to and chosen as
// installed-play.spec.ts does (support/kit-package.ts); the conversations list is answered with none, so the
// chat opens on a new chat, and nothing is sent.
//
// The label is the host's (components/ai-label/ai-label.tsx, drawn by app/chat/layout.tsx after the
// theme's page): role="note", "Auto is an AI", once, outside Play's page, frame and composer, a tiny caption
// with no border, background or rounding (never a pill). Since Play 1.2.0 its message box (the kit's
// themes/play/src/components/play/chat/message-box.tsx) holds the words, then the host's voice button (the
// SDK's voice.VoiceButton), then Send, the words and Send each in the SDK's cards.AiLabelFence, which now
// draws its children alone. While a style keeps the label from being seen, by any of the ways the host's
// check reads (Play's frame under the label's corner painted the words' own colour, by colour or by a gradient,
// though it is not the label's ancestor; display, transparent words, a blank ::first-line or ::first-letter, a
// ::before of the words, a text stroke, a clipped box, an indent off-screen, a clip-path, something drawn over
// it, click-through or not),
// or its check throws, the host draws the same words in its own layer (components/ai-label/protected-caption.tsx): a
// closed shadow root on a guarded element of the host's at the end of <body>, the words a popover in the
// browser's top layer, and logs a theme-host warning that names how. A second style aimed at that copy,
// and a popover of the page's shown after it, leave it on screen and on top (support/ai-label-copy.ts reads
// the closed root). The words, Send and the voice button stay enabled throughout. This is the browser half
// of lib/theme-host/__tests__/ai-label.test.ts (jsdom has no layout, hit-testing or top layer, and matches
// a document's rules inside shadow roots), on installed Play, the only Play since W-A8 took builtin Play out
// of core.
//
// The specs store the member's choice on the lane's one backend, so they run in the @package step,
// one test at a time; each starts and ends on Studio with no install of Play's package.

const APP_URL = process.env.E2E_APP_URL ?? 'http://localhost:3000'
const CHAT_PATH = '/chat'
const STATUS_ATTR = 'data-theme-host'
const PLAY_CLASS = /\bplay\b/
/** The host's label (lib/ai-label/copy.ts aiLabelText, with no agent picked). */
const AI_LABEL = 'Auto is an AI'
/** components/ai-label/ai-label.tsx AI_LABEL_ATTR. */
const LABEL_SELECTOR = '[data-ai-label]'
/** lib/ai-label/visibility.ts AI_LABEL_PROBE_ATTR: the probe's transient sheet, never left on the page. */
const PROBE_SELECTOR = '[data-ai-label-probe]'
/** Play's frame: its sidebar (the kit's themes/play/src/components/play/shell/play-sidebar.tsx). */
const PLAY_FRAME = 'aside[aria-label="Sidebar"]'
/** A stylesheet that hides the host's label under installed Play's own <html> class, as a hostile theme's would. */
const HIDING_STYLE = `html.play ${LABEL_SELECTOR} { display: none !important; }`
/** A stylesheet aimed at the host's copy: its element by name, and anything a generic rule could catch it by. */
const AIMED_AT_COPY =
  `html.play ${CAPTION_SELECTOR}, html.play ${CAPTION_SELECTOR} *, html.play [popover], html.play p[role="note"] {` +
  ' display: none !important; visibility: hidden !important; opacity: 0 !important;' +
  ' transform: translateY(-200vh) !important; }'
/** Words in the box, so Send is enabled (Send stays disabled on an empty box). */
const DRAFT = 'Hello Auto'

/** Installs Play's package, chooses it, and opens /chat on a new chat; the message box is on screen. */
async function openInstalledPlayChat(page: Page, request: APIRequestContext): Promise<void> {
  const install = await installKitPackage(request)
  await chooseKitPackage(page, install, APP_URL)
  await noStoredConversations(page)
  await page.goto(CHAT_PATH)
  const html = page.locator('html')
  await expect(html).toHaveAttribute(STATUS_ATTR, 'active', { timeout: THEME_LOAD_TIMEOUT_MS })
  await expect(html).toHaveClass(PLAY_CLASS)
  await expect(messageBox(page)).toBeVisible({ timeout: THEME_LOAD_TIMEOUT_MS })
}

/** The copy on screen, whole, on top and in the top layer, saying the label's words. */
const COPY_ON_TOP = { text: AI_LABEL, open: true, inViewport: true, painted: true, topmost: true }

/** The theme-host warning use-ai-label-check.ts logs for the page label hidden by `reason`. */
function hiddenWarning(reason: string): RegExp {
  return new RegExp(`\\[theme-host\\] the active theme hides the page AI label \\(${reason}\\)`)
}

/**
 * Ways a theme's style can hide the label that the host's check reads, by generic selectors the server's
 * scope check would let through (role=note), or by name (refused at upload, injected here directly), each
 * with the reason the check gives.
 */
const HIDING_WAYS: ReadonlyArray<readonly [string, string, string]> = [
  // Play's frame lies under the fixed label's corner but is not its ancestor: the ground is read from the paint stack
  [
    "Play's frame under the label painted the words' own colour",
    `html.play body > div:has(> ${PLAY_FRAME}) { background-color: hsl(var(--muted-foreground)) !important; }`,
    'color',
  ],
  [
    "a gradient of the words' own colour on Play's frame under the label",
    `html.play body > div:has(> ${PLAY_FRAME}) { background-image: linear-gradient(hsl(var(--muted-foreground)), hsl(var(--muted-foreground))) !important; }`,
    'color',
  ],
  ['transparent words', 'html.play [role="note"] { color: transparent !important; }', 'color'],
  ['a 4 px box that clips its words', `html.play ${LABEL_SELECTOR} { width: 4px !important; height: 4px !important; overflow: hidden !important; }`, 'text'],
  ['words indented off-screen', `html.play ${LABEL_SELECTOR} { text-indent: -9999px !important; }`, 'text'],
  ['a clip-path', `html.play ${LABEL_SELECTOR} { clip-path: inset(50%) !important; }`, 'paint'],
  ['a blank first line', 'html.play [role="note"]::first-line { color: transparent !important; }', 'paint'],
  ['a first letter with no fill', 'html.play p[role="note"]::first-letter { -webkit-text-fill-color: transparent !important; }', 'paint'],
  [
    "a ::before of the words drawn over them (it hit-tests as the words)",
    'html.play [role="note"] span::before { content: "" !important; position: absolute !important; inset: 0 !important; background: Canvas !important; }',
    'paint',
  ],
  ['a text stroke in the ground colour that blots the glyphs', 'html.play [role="note"] { -webkit-text-stroke: 3px Canvas !important; }', 'paint'],
]

/**
 * Draws an element of the page's over the label's corner, above its z-index, click-through or not (a theme
 * could make its cover `pointer-events: none` so hit-testing skips it); returns what takes it away.
 */
async function coverTheLabel(page: Page, clickThrough: boolean): Promise<() => Promise<void>> {
  const cover = await page.evaluateHandle(({ selector, through }) => {
    const box = document.querySelector(selector)?.getBoundingClientRect()
    const element = document.createElement('div')
    element.setAttribute('data-test-cover', '')
    const style = `position: fixed; left: ${(box?.left ?? 0) - 8}px; top: ${(box?.top ?? 0) - 8}px;` +
      ` width: ${(box?.width ?? 0) + 16}px; height: ${(box?.height ?? 0) + 16}px; z-index: 2147483000; background: rgb(250, 0, 0);` +
      (through ? ' pointer-events: none;' : '')
    element.setAttribute('style', style)
    document.body.appendChild(element)
    return element
  }, { selector: LABEL_SELECTOR, through: clickThrough })
  return async () => {
    await cover.evaluate((element: Element) => element.remove())
  }
}

/** Shows a popover of the page's over the bottom of the viewport, after the copy; returns what hides it. */
async function showPagePopover(page: Page): Promise<() => Promise<void>> {
  const popover = await page.evaluateHandle(() => {
    const element = document.createElement('div')
    element.setAttribute('popover', 'manual')
    element.setAttribute('style', 'position: fixed; inset: auto 0 0 0; width: 100vw; height: 160px; margin: 0; background: rgb(0, 0, 250);')
    document.body.appendChild(element)
    element.showPopover()
    return element
  })
  return async () => {
    await popover.evaluate((element: Element) => element.remove())
  }
}

async function removeStyle(style: ElementHandle<Node>): Promise<void> {
  await style.evaluate((node: Node) => {
    node.parentNode?.removeChild(node)
  })
}

/** The label's computed box under Play: a caption draws no border, background or rounding (never a pill). */
async function labelBox(page: Page): Promise<readonly string[]> {
  return page.locator(LABEL_SELECTOR).evaluate((node: Element) => {
    const style = window.getComputedStyle(node)
    return [style.borderTopWidth, style.borderRadius, style.backgroundColor, style.boxShadow]
  })
}

/** Play's words, Send and the voice button all run, the draft still in the box. */
async function expectChatRuns(page: Page): Promise<void> {
  await expect(messageBox(page)).toBeEnabled()
  await expect(messageBox(page)).toHaveValue(DRAFT)
  await expect(sendButton(page)).toBeEnabled()
  await expect(voiceButton(page)).toBeEnabled()
}

test.describe("PRD-261 A-19e1 — installed Play's /chat carries the host's AI label (PRD-264)", () => {
  test.beforeEach(async ({ page, request }) => {
    await resetKitPackage(request)
    await recordCopyRoots(page)
  })

  test.afterEach(async ({ request }) => {
    await resetKitPackage(request)
  })

  test("shows the label once, outside Play's page, frame and composer, and the message box runs", { tag: '@package' }, async ({ page, request }) => {
    const watch = watchPage(page)
    await openInstalledPlayChat(page, request)

    const label = page.getByRole('note', { name: AI_LABEL, exact: true })
    await expect(label).toBeVisible()
    await expect(label).toHaveText(AI_LABEL)
    await expect(label).toBeInViewport()
    await expect(page.locator(LABEL_SELECTOR)).toHaveCount(1)
    expect(await labelBox(page)).toEqual(['0px', '0px', 'rgba(0, 0, 0, 0)', 'none'])
    await expect(page.locator(PLAY_FRAME)).toBeVisible()
    await expect(page.locator(`${PLAY_FRAME} ${LABEL_SELECTOR}`)).toHaveCount(0)
    await expect(page.locator(`main ${LABEL_SELECTOR}`)).toHaveCount(0)
    // the host's check reads the caption as seen (painted, readable, nothing over it): no copy in its own layer
    await expect(page.locator(CAPTION_SELECTOR)).toHaveCount(0)
    expect(await copyState(page)).toBeNull()
    await messageBox(page).fill(DRAFT)
    await expectChatRuns(page)
    expect(watch.pageErrors).toEqual([])
  })

  test("a style that hides the label brings the host's own copy, which a style aimed at it can't hide; the chat runs throughout", { tag: '@package' }, async ({ page, request }) => {
    const warnings = watchWarnings(page)
    await openInstalledPlayChat(page, request)
    await messageBox(page).fill(DRAFT)
    await expectChatRuns(page)

    const hiding = await page.addStyleTag({ content: HIDING_STYLE })
    await expect(page.locator(LABEL_SELECTOR)).toBeHidden()
    await expect.poll(() => copyState(page)).toMatchObject(COPY_ON_TOP)
    await expect.poll(() => warnings.some((warning) => hiddenWarning('display').test(warning))).toBe(true)
    await expectChatRuns(page)

    const aimed = await page.addStyleTag({ content: AIMED_AT_COPY })
    await expect.poll(() => copyState(page)).toMatchObject(COPY_ON_TOP)
    // the host's element keeps its inline guard against the important rules aimed at it
    const guard = await page.locator(CAPTION_SELECTOR).evaluate((node: Element) => {
      const style = window.getComputedStyle(node)
      return [style.display, style.visibility, style.opacity, style.transform]
    })
    expect(guard).toEqual(['block', 'visible', '1', 'none'])
    await expectChatRuns(page)
    await removeStyle(aimed)

    // a popover of the page's shown after the copy: the copy rises above it again
    const hidePopover = await showPagePopover(page)
    await expect.poll(() => copyState(page)).toMatchObject(COPY_ON_TOP)
    await hidePopover()

    await removeStyle(hiding)
    await expect(page.locator(LABEL_SELECTOR)).toBeVisible()
    await expect(page.locator(CAPTION_SELECTOR)).toHaveCount(0)
    await expectChatRuns(page)
  })

  test('every way of hiding the label the check reads brings the copy and a warning; taking it away takes the copy away', { tag: '@package' }, async ({ page, request }) => {
    const warnings = watchWarnings(page)
    await openInstalledPlayChat(page, request)
    await messageBox(page).fill(DRAFT)

    for (const [name, css, reason] of HIDING_WAYS) {
      const style = await page.addStyleTag({ content: css })
      await expect.poll(() => copyState(page), { message: name }).toMatchObject(COPY_ON_TOP)
      await expect.poll(() => warnings.some((warning) => hiddenWarning(reason).test(warning)), { message: name }).toBe(true)
      await expectChatRuns(page)
      await removeStyle(style)
      await expect.poll(() => copyState(page), { message: `${name}, taken away` }).toBeNull()
    }

    for (const clickThrough of [false, true]) {
      const name = clickThrough ? 'a click-through element drawn over the label' : 'an element drawn over the label'
      const uncover = await coverTheLabel(page, clickThrough)
      await expect.poll(() => copyState(page), { message: name }).toMatchObject(COPY_ON_TOP)
      await expect.poll(() => warnings.some((warning) => hiddenWarning('covered').test(warning))).toBe(true)
      await expect(page.locator(PROBE_SELECTOR)).toHaveCount(0)
      await expectChatRuns(page)
      await uncover()
      await expect.poll(() => copyState(page), { message: `${name}, taken away` }).toBeNull()
    }

    // a check that throws (a page script patched getComputedStyle, for the label's elements only, so the rest of
    // the page keeps working) counts the label as hidden
    await page.evaluate((selector) => {
      const real = window.getComputedStyle
      Object.defineProperty(window, '__realGetComputedStyle', { value: real, configurable: true })
      window.getComputedStyle = (element: Element, pseudo?: string | null) => {
        if (element.closest(selector)) throw new Error('getComputedStyle patched by the page')
        return real.call(window, element, pseudo)
      }
    }, LABEL_SELECTOR)
    await expect.poll(() => copyState(page), { message: 'a check that throws' }).toMatchObject(COPY_ON_TOP)
    await expect.poll(() => warnings.some((warning) => /the page AI label could not be checked \(unchecked/.test(warning))).toBe(true)
    await page.evaluate(() => {
      window.getComputedStyle = (window as unknown as { __realGetComputedStyle: typeof window.getComputedStyle }).__realGetComputedStyle
    })
    await expect.poll(() => copyState(page), { message: 'the check restored' }).toBeNull()
  })
})
