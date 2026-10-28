/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — every in-app chat shows "Auto is an AI", and no
 * theme can remove it.
 *
 * app/chat/layout.tsx is rendered for real, with the default chat (its input, the one Mission
 * Mode uses too: chat.tsx draws MultimodalInput), with an installed theme whose boot carried
 * no entry that registers /chat through the theme host (ThemeSlot draws its page in MainLayout: the
 * theme's own markup and stylesheet around a reply's card and its own composer inside the SDK's
 * cards.AiLabelFence, as Play's package draws its message box), and (PRD-261 A-19e1) under the
 * neutral fixture theme of A-19b (fixture-theme-entry.ts) as the active installed theme, its /chat
 * page a theme's own composer inside cards.AiLabelFence. No theme is compiled into the app (A-19f1),
 * so every theme's chat reaches the page through the theme host.
 * The label renders once, after ThemeSlot, outside the theme's page, with role="note" and the
 * same words. Then hostile fixture themes hide it: `display:none`, `visibility:hidden`, `opacity:0`,
 * zero size, an off-screen move, a fade on an ancestor, transparent or faint words (by colour or by
 * `-webkit-text-fill-color`), a blank ::first-line or ::first-letter, a clip-path, a mask, a filter on an
 * ancestor, a tiny font, words the
 * browser would lay out squeezed or off-screen, something drawn over them, and a ground under them the
 * page lays out that isn't the label's ancestor (an element, a gradient or a patch under one point in the
 * words' own colour, an image no check can read; an opaque element faded out fakes no ground, and HUD's
 * backdrop as built is no harm). Each time the label
 * reports itself hidden, the host draws the same words in its own layer (components/ai-label/
 * protected-caption.tsx: a closed shadow root on an element of the host's at the end of <body>,
 * guarded inline), a theme-host warning names what the theme did, and the chat keeps working: its
 * input and a card's button stay enabled, and nothing says the chat is paused. A sheet aimed at
 * that copy itself leaves it on screen. A theme switched in after mount (the observer on <html>
 * and <head>) brings the copy, and switching back takes it away.
 *
 * jsdom lays nothing out, so ai-label-layout.ts installs what a browser would hand the check
 * (element and text boxes, hit-testing, the copy's closed shadow roots, a popover stand-in). jsdom
 * also matches a document's rules against elements inside a shadow root, which a browser never
 * does, so its verdict on the host's copy is harsher than a browser's; the top layer, real text
 * layout and real hit-testing are e2e/browser/specs/installed-play-ai-label.spec.ts's.
 */
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { createElement as h, type ReactElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ThemeEntry } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({
  entry: null as ThemeEntry | null,
  layout: 'fallback' as 'fallback' | 'theme' | 'pending',
}))

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
vi.mock('@/hooks/use-studio-theme', () => ({
  useIsStudio: () => state.layout === 'theme',
  useActiveTheme: () => state.entry,
  useThemeLayout: () => state.layout,
}))
/** The theme's frame around an installed theme's page: its content alone here. */
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { readonly children?: ReactNode }) => children,
}))
vi.mock('@/components/chatbot/discussion-bar', () => ({ DiscussionBar: () => null }))
vi.mock('@/components/chatbot/agent-selector', () => ({ AgentSelector: () => null }))
vi.mock('@/lib/api-client', () => ({ apiClient: { uploadAttachment: vi.fn() } }))
vi.mock('@/hooks/use-agent-api', () => ({
  useAgents: () => ({ data: [{ id: 7, name: 'Research Desk' }, { id: 8, name: '  ' }] }),
}))

import ChatLayout from '@/app/chat/layout'
import { AI_LABEL_ATTR, AiLabelGate } from '@/components/ai-label/ai-label'
import { aiLabelHiddenWarning } from '@/components/ai-label/use-ai-label-check'
import { MultimodalInput } from '@/components/chatbot/multimodal-input'
import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { aiLabelText } from '@/lib/ai-label/copy'
import { useAiLabelStore, usePublishAnsweringAgent } from '@/lib/ai-label/store'
import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'
import { AiLabelFence as ThemeAiLabelFence } from '@/lib/theme-sdk/host-cards/ai-label-fence'
import { CHAT_HREF } from '@/lib/ticket-links'

import { type LabelLayout, captionHost, installLabelLayout, installPopover, theCaption } from './ai-label-layout'
import { FIXTURE_THEME_ENTRY, FIXTURE_THEME_TEST_IDS, fixtureThemeHost } from './fixture-theme-entry'

type Page = 'default' | 'theme' | 'fixture'

const PAGES: readonly Page[] = ['default', 'theme', 'fixture']
const AUTO_LABEL = 'Auto is an AI'
const THEME_CLASS = 'hello-chat'
const HOSTILE_CLASS = 'hostile-theme'
const CARD_BUTTON = 'Approve & run'
/** The entry-less installed theme's own composer: its box and its Send. */
const THEME_COMPOSER = 'Message Auto in the hello theme'
const THEME_SEND = 'Send to the hello theme'
/** The fixture theme's own composer: its box and its Send. */
const FIXTURE_COMPOSER = 'Message the fixture theme'
const FIXTURE_SEND = 'Send to the fixture theme'
/** Where a label-sized box sits in jsdom's 1024 x 768 window when no style moves or sizes it. */
const LABEL_BOX = { left: 880, top: 740, width: 110, height: 20 }
const LABEL = `[${AI_LABEL_ATTR}]`
const CAPTION_HOST = '[data-ai-label-caption]'
const THEME_HOST_LOG = '[theme-host]'

function addThemeSheet(css: string): void {
  const sheet = document.createElement('style')
  sheet.setAttribute('data-test-theme', '')
  sheet.textContent = css
  document.head.appendChild(sheet)
}

/** The default chat's input, as chat.tsx draws it (Mission Mode's too). */
function defaultChat(): ReactElement {
  return h(MultimodalInput, {
    chatId: 'ai-label-test', status: 'ready', stop: vi.fn(), sendMessage: vi.fn(), selectedVisibilityType: 'private',
  })
}

/** A reply's card: a protected surface, outside the composer. */
function card(): ReactElement {
  return h('section', { 'data-testid': 'card' }, h('button', { type: 'button' }, CARD_BUTTON))
}

/** A theme's own composer, its box and its Send, inside the SDK's cards.AiLabelFence (its composer only). */
function themeComposer(box: string, send: string): ReactElement {
  const composer = h('form', null, h('textarea', { 'aria-label': box }), h('button', { type: 'submit' }, send))
  return h(ThemeAiLabelFence, null, composer)
}

/** An installed theme that registers /chat: its own page markup around a reply's card and its own composer. */
function themeChat(): ReactElement {
  return h('div', { 'data-testid': 'theme-page', 'data-theme-page': '' }, h('header', null, 'Hello theme'), card(), themeComposer(THEME_COMPOSER, THEME_SEND))
}

/**
 * The A-19b fixture theme's /chat page as an installed theme draws it with the SDK alone: its own
 * markup, a reply's card, and its own composer inside cards.AiLabelFence.
 */
function fixtureChat(): ReactElement {
  return h('div', { 'data-testid': FIXTURE_THEME_TEST_IDS.chat }, h('header', null, 'Fixture theme'), card(), themeComposer(FIXTURE_COMPOSER, FIXTURE_SEND))
}

function FixtureFrame({ children }: { readonly children?: ReactNode }): ReactElement {
  return h('div', { 'data-testid': 'theme-frame' }, children)
}

/** The theme host with an installed theme active (its boot carried no entry), its /chat page registered (registerPage). */
const FIXTURE_THEME_HOST: ThemeHostContextValue = {
  status: 'active',
  reason: null,
  frame: FixtureFrame,
  slots: new Map(),
  pages: new Map<string, unknown>([[CHAT_HREF, themeChat]]),
  routes: new Map(),
  reportRenderFailure: () => undefined,
}

/** The active theme's entry under each page: the A-19b fixture theme's, or none. */
const ENTRY_OF: Readonly<Record<Page, ThemeEntry | null>> = { default: null, theme: null, fixture: FIXTURE_THEME_ENTRY }
/** The element each page's chat is drawn in. */
const PAGE_ROOT: Readonly<Record<Page, string>> = { default: 'core-chat', theme: 'theme-page', fixture: FIXTURE_THEME_TEST_IDS.chat }
/** The name of each page's chat input. */
const COMPOSER_OF: Readonly<Record<Exclude<Page, 'default'>, string>> = { theme: THEME_COMPOSER, fixture: FIXTURE_COMPOSER }

/** The theme host's value for a page an installed theme draws; null for the default chat. */
function hostOf(page: Page): ThemeHostContextValue | null {
  if (page === 'theme') return FIXTURE_THEME_HOST
  if (page === 'fixture') return fixtureThemeHost('active', { pages: new Map<string, unknown>([[CHAT_HREF, fixtureChat]]) })
  return null
}

function renderChat(page: Page) {
  state.entry = ENTRY_OF[page]
  state.layout = page === 'default' ? 'fallback' : 'theme'
  const chat = h(ChatLayout, null, h('div', { 'data-testid': 'core-chat' }, card(), defaultChat()))
  const host = hostOf(page)
  return render(host ? h(ThemeHostContext.Provider, { value: host }, chat) : chat)
}

function inputOf(page: Page): HTMLElement {
  if (page === 'default') return screen.getByRole('textbox')
  return screen.getByRole('textbox', { name: COMPOSER_OF[page] })
}

function cardButton(): HTMLElement {
  return screen.getByRole('button', { name: CARD_BUTTON, hidden: true })
}

function theLabel(): HTMLElement {
  const labels = document.querySelectorAll<HTMLElement>(LABEL)
  expect(labels).toHaveLength(1)
  return labels[0]
}

/** The class names on `element` that would draw a pill: a rounding, a border, a background, a shadow or a ring. */
function pillClasses(element: Element): string[] {
  return [...element.classList].filter((name) => /^(rounded|border|bg-|shadow|ring)/.test(name))
}

/** A theme hides the label with `css`; resolves once the host's copy is on screen. */
async function hideWith(css: string): Promise<HTMLElement> {
  addThemeSheet(css)
  document.documentElement.classList.add(HOSTILE_CLASS)
  return waitFor(theCaption)
}

/** The theme-host warnings logged so far: what a theme that hides the label leaves in the console. */
function hostWarnings(): unknown[][] {
  return vi.mocked(console.warn).mock.calls.filter(([prefix]) => prefix === THEME_HOST_LOG)
}

let layout: LabelLayout
let uninstallLayout: () => void

/**
 * The copy's own styles keep it on screen: displayed, visible, opaque, a readable size. Its hit-test
 * isn't probed here: the probe writes to the copy, and the copy rebuilds itself on any write.
 */
function expectCopyShown(caption: HTMLElement): void {
  const style = window.getComputedStyle(caption)
  expect([style.display, style.visibility, style.opacity]).toEqual(['block', 'visible', '1'])
  expect(caption.isConnected).toBe(true)
}

beforeEach(() => {
  ;({ layout, uninstall: uninstallLayout } = installLabelLayout(LABEL_BOX))
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  uninstallLayout()
  vi.restoreAllMocks()
  document.head.querySelectorAll('[data-test-theme]').forEach((sheet) => sheet.remove())
  document.documentElement.classList.remove(THEME_CLASS, HOSTILE_CLASS)
  useAiLabelStore.setState({ agentId: null, hiddenLabels: [], shownLabels: [] })
  state.entry = null
  state.layout = 'fallback'
})

describe('the AI label on every in-app chat (PRD-264 EUAI-11a)', () => {
  it.each(PAGES)('shows once on the %s chat, outside the page, as a note', (page) => {
    if (page === 'theme') {
      document.documentElement.classList.add(THEME_CLASS)
      addThemeSheet(`html.${THEME_CLASS} [data-theme-page] { color: rebeccapurple; padding: 12px; }`)
    }
    renderChat(page)
    const note = screen.getByRole('note', { name: AUTO_LABEL })
    expect(note).toHaveTextContent(AUTO_LABEL)
    expect(theLabel()).toBe(note)
    // a tiny muted caption, never a pill, and the check still reads it as seen
    expect(pillClasses(note)).toEqual([])
    expect(note).toHaveClass('text-[11px]', 'text-muted-foreground')
    expect(aiLabelHiddenReason(note, window)).toBeNull()
    // after ThemeSlot, outside the page and the theme's markup: no page or theme draws it
    const pageRoot = screen.getByTestId(PAGE_ROOT[page])
    expect(pageRoot.contains(note)).toBe(false)
    if (page !== 'default') expect(screen.queryByTestId('core-chat')).toBeNull()
    // seen where it is: no copy in the host's layer, no warning, the composer runs
    expect(inputOf(page)).toBeEnabled()
    expect(captionHost()).toBeNull()
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: [], shownLabels: ['page'] })
    expect(hostWarnings()).toEqual([])
  })

  it.each(PAGES)('on the %s chat no input renders a second copy of the label', (page) => {
    renderChat(page)
    expect(document.querySelectorAll(LABEL)).toHaveLength(1)
    expect(screen.getAllByRole('note')).toHaveLength(1)
  })
})

const HOSTILE_RULES: ReadonlyArray<readonly [string, string, string]> = [
  ['display:none', `html.${HOSTILE_CLASS} ${LABEL} { display: none; }`, 'display'],
  ['visibility:hidden', `html.${HOSTILE_CLASS} ${LABEL} { visibility: hidden; }`, 'visibility'],
  ['opacity:0', `html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`, 'opacity'],
  ['zero size', `html.${HOSTILE_CLASS} ${LABEL} { width: 0px; height: 0px; }`, 'size'],
  ['an off-screen move', `html.${HOSTILE_CLASS} ${LABEL} { position: fixed; top: -500px; }`, 'viewport'],
  ['a fade on an ancestor', `html.${HOSTILE_CLASS} body { opacity: 0.2; }`, 'opacity'],
  // paint the check reads, by generic selectors the server's scope check lets through
  ['transparent words', `html.${HOSTILE_CLASS} [role="note"] { color: transparent; }`, 'color'],
  ['a transparent text fill', `html.${HOSTILE_CLASS} [role="note"] { -webkit-text-fill-color: transparent; }`, 'color'],
  ['faint words', `html.${HOSTILE_CLASS} [role="note"] { color: rgb(236, 236, 236); }`, 'color'],
  ['words the colour of their ground', `html.${HOSTILE_CLASS} body { background-color: rgb(20, 20, 20); color: rgb(28, 28, 28); }`, 'color'],
  ['a clip-path', `html.${HOSTILE_CLASS} [role="note"] { clip-path: inset(50%); }`, 'paint'],
  ['a mask', `html.${HOSTILE_CLASS} [role="note"] { mask-image: linear-gradient(transparent, transparent); }`, 'paint'],
  ['a filter on an ancestor', `html.${HOSTILE_CLASS} body { filter: opacity(0); }`, 'paint'],
  ['a tiny font', `html.${HOSTILE_CLASS} [role="note"] { font-size: 4px; }`, 'font'],
  // a 3px stroke in the ground's own colour on 24px type: not thick enough to fill the glyphs, but it blots them
  [
    'a text stroke in the colour of the ground',
    `html.${HOSTILE_CLASS} [role="note"] { font-size: 24px; -webkit-text-stroke-width: 3px; -webkit-text-stroke-color: Canvas; }`,
    'paint',
  ],
  ['a text stroke thick enough to fill the glyphs (0.3em), in any colour', `html.${HOSTILE_CLASS} [role="note"] { font-size: 12px; -webkit-text-stroke-width: 0.3em; -webkit-text-stroke-color: rgb(0, 0, 0); }`, 'paint'],
  // ::before and ::after of the words or the label, read from the rules (jsdom reports no pseudo-element styles)
  ['a ::before of the words, by its rule', `html.${HOSTILE_CLASS} [role="note"] span::before { content: ""; position: absolute; inset: 0; }`, 'paint'],
  ['an :after of the label that adds words, by its rule', `html.${HOSTILE_CLASS} p[role="note"]:after { content: " (not really)"; }`, 'paint'],
  // ::first-line and ::first-letter paint the words without showing in getComputedStyle(element)
  ['a blank first line', `html.${HOSTILE_CLASS} [role="note"]::first-line { color: transparent; }`, 'paint'],
  ['a first letter with no fill', `html.${HOSTILE_CLASS} p[role="note"]::first-letter { -webkit-text-fill-color: transparent; }`, 'paint'],
]

describe('no theme can hide the AI label (PRD-264 EUAI-11a)', () => {
  const cases: Array<[Page, string, string, string]> = PAGES.flatMap((page) =>
    HOSTILE_RULES.map(([name, css, reason]): [Page, string, string, string] => [page, name, css, reason]))

  it.each(cases)("on the %s chat, a theme that sets %s on it gets the label drawn in the host's own layer, and the chat runs", async (page, _name, css, reason) => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(css)
    renderChat(page)
    const label = theLabel()
    expect(label).toHaveTextContent(AUTO_LABEL)
    expect(aiLabelHiddenReason(label, window)).toBe(reason)
    const caption = await waitFor(theCaption)
    expect(caption).toHaveTextContent(AUTO_LABEL)
    expect(caption).toHaveAttribute('aria-label', AUTO_LABEL)
    expectCopyShown(caption)
    // the chat is never paused: the composer and a reply's card stay enabled, and nothing says otherwise
    expect(inputOf(page)).toBeEnabled()
    expect(cardButton()).toBeEnabled()
    expect(screen.queryByRole('alert', { hidden: true })).toBeNull()
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: ['page'], shownLabels: ['page'] })
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', reason)]])
  })

  it("draws the copy outside the theme's page, on an element of the host's at the end of <body>, guarded inline", async () => {
    renderChat('fixture')
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    const host = captionHost() as HTMLElement
    expect(host.parentElement).toBe(document.body)
    expect(screen.getByTestId(FIXTURE_THEME_TEST_IDS.chat).contains(host)).toBe(false)
    for (const property of ['display', 'visibility', 'opacity', 'position', 'pointer-events']) {
      expect(host.style.getPropertyPriority(property)).toBe('important')
    }
    // closed: a theme's code gets no shadowRoot, and a document's selectors don't reach inside it
    expect(host.shadowRoot).toBeNull()
    expect(document.querySelectorAll(`[aria-label="${AUTO_LABEL}"]`)).toHaveLength(1)
    expect(theCaption().getAttribute('popover')).toBe('manual')
    for (const property of ['display', 'visibility', 'opacity', 'color', 'font', 'white-space', 'max-width']) {
      expect(theCaption().style.getPropertyPriority(property)).toBe('important')
    }
  })

  it("keeps the copy on screen against a sheet aimed at it, its host and everything around it", async () => {
    renderChat('theme')
    await hideWith([
      `html.${HOSTILE_CLASS} ${LABEL} { display: none; }`,
      `${CAPTION_HOST}, ${CAPTION_HOST} p, p[role="note"], [popover] {` +
        ' display: none !important; visibility: hidden !important; opacity: 0 !important;' +
        ' width: 0px !important; height: 0px !important; top: -500px !important; left: -500px !important; }',
      'body { opacity: 0.2 !important; visibility: hidden !important; }',
    ].join('\n'))
    const host = captionHost() as HTMLElement
    const hostStyle = window.getComputedStyle(host)
    expect([hostStyle.display, hostStyle.visibility, hostStyle.opacity, hostStyle.position]).toEqual(['block', 'visible', '1', 'fixed'])
    expectCopyShown(theCaption())
    expect(theCaption()).toHaveTextContent(AUTO_LABEL)
    // the sheet hides the page with <body>, so nothing in it has an accessible name (a hidden element's is
    // empty) and a role query can't find the composer by name; found by its label instead, it is not disabled
    expect(screen.getByLabelText(THEME_COMPOSER)).toBeEnabled()
    expect(screen.getByText(THEME_SEND)).toBeEnabled()
  })

  it('raises the copy into the browser\'s top layer where the browser has one', async () => {
    const popover = installPopover()
    try {
      renderChat('default')
      const caption = await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
      expect(popover.raised).toEqual([caption])
    } finally {
      popover.uninstall()
    }
  })

  it.each([
    ['a line the browser lays out squeezed', { left: 880, top: 740, width: 12, height: 20 }],
    ['a line the browser lays out too short', { left: 880, top: 740, width: 110, height: 5 }],
    ['words indented off-screen', { left: -9119, top: 740, width: 110, height: 20 }],
  ])("brings the copy for %s (the words' own box, not the label's)", async (_name, textBox) => {
    renderChat('default')
    expect(captionHost()).toBeNull()
    layout.textBox = textBox
    const caption = await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { color: rgb(80, 80, 80); }`)
    expect(aiLabelHiddenReason(theLabel(), window)).toBe('text')
    expect(caption).toHaveTextContent(AUTO_LABEL)
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'text')]])
    expect(inputOf('default')).toBeEnabled()
  })

  it('brings the copy when the label clips its own words (a 4 px box with overflow hidden)', async () => {
    renderChat('default')
    layout.textBox = { left: 880, top: 740, width: 70, height: 13 }
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { width: 4px; height: 4px; overflow: hidden; }`)
    expect(aiLabelHiddenReason(theLabel(), window)).toBe('text')
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'text')]])
  })

  it("brings the copy when something is drawn over the words, the label's own pseudo-element included", async () => {
    renderChat('theme')
    expect(captionHost()).toBeNull()
    layout.cover = theLabel()
    const caption = await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { color: rgb(80, 80, 80); }`)
    expect(aiLabelHiddenReason(theLabel(), window)).toBe('covered')
    expect(caption).toHaveTextContent(AUTO_LABEL)
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'covered')]])
    expect(inputOf('theme')).toBeEnabled()
    // the probe leaves nothing behind: no inline style on the label, no probe sheet
    expect(theLabel().style.getPropertyValue('pointer-events')).toBe('')
    expect(document.querySelector('[data-ai-label-probe]')).toBeNull()
  })

  it('brings the copy for a click-through overlay drawn over the words: pointer-events: none hides nothing from the probe', async () => {
    renderChat('theme')
    const overlay = document.body.appendChild(document.createElement('div'))
    overlay.setAttribute('data-test-overlay', '')
    layout.cover = overlay
    try {
      await hideWith(`html.${HOSTILE_CLASS} [data-test-overlay] { pointer-events: none; }`)
      expect(window.getComputedStyle(overlay).pointerEvents).toBe('none')
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('covered')
      expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'covered')]])
      expect(document.querySelector('[data-ai-label-probe]')).toBeNull()
      expect(inputOf('theme')).toBeEnabled()
    } finally {
      overlay.remove()
    }
  })

  it("reads the computed ::first-line where the browser reports it: a transparent first line brings the copy", async () => {
    const realStyle = window.getComputedStyle.bind(window)
    const blankLine = document.createElement('div').style
    blankLine.setProperty('color', 'rgba(0, 0, 0, 0)')
    Object.defineProperty(window, 'CSS', { value: { supports: () => true }, configurable: true })
    vi.spyOn(window, 'getComputedStyle').mockImplementation((element: Element, pseudo?: string | null) =>
      pseudo === '::first-line' && element.hasAttribute('data-ai-label-text') ? blankLine : realStyle(element))
    try {
      renderChat('default')
      expect(await waitFor(theCaption)).toHaveTextContent(AUTO_LABEL)
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('color')
      expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'color')]])
    } finally {
      delete (window as { CSS?: unknown }).CSS
    }
  })

  it.each([
    [
      'a thin outline in a colour that stands out from the ground',
      `html.${HOSTILE_CLASS} [role="note"] { font-size: 12px; -webkit-text-stroke-width: 1px; -webkit-text-stroke-color: rgb(0, 0, 0); }`,
    ],
    [
      // Studio's own compiled Tailwind (shadcn's resizable handle): an escaped `\:after` inside a class is no
      // pseudo-element (CI run 40760817606 read it as one and drew the copy on Studio's default /chat)
      "Tailwind's before: and after: variant utilities, their escaped colons included",
      String.raw`.data-\[panel-group-direction\=vertical\]\:after\:left-0[data-panel-group-direction="vertical"]::after { content: var(--tw-content); left: 0px; }` +
        '\n' +
        String.raw`.hover\:before\:opacity-100:hover::before { content: var(--tw-content); opacity: 1; }`,
    ],
    [
      'pseudo-elements that draw nothing, or belong to the page',
      `html.${HOSTILE_CLASS} *, html.${HOSTILE_CLASS} ::before, html.${HOSTILE_CLASS} ::after { box-sizing: border-box; }\n` +
        `html.${HOSTILE_CLASS} [role="note"] span::before { color: red; }\n` +
        `html.${HOSTILE_CLASS} body::before { content: ""; position: fixed; inset: 0; }`,
    ],
  ])('reads %s as no harm: the label is seen, with no copy and no warning', (_name, css) => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(css)
    renderChat('default')
    expect(aiLabelHiddenReason(theLabel(), window)).toBeNull()
    expect(captionHost()).toBeNull()
    expect(hostWarnings()).toEqual([])
  })

  it.each([
    ['a ::before of the words', '[data-ai-label-text]', '::before'],
    ['an ::after of the label', '[data-ai-label]', '::after'],
  ])("brings the copy for %s: it hit-tests as what it belongs to, so its content is what shows it", async (_name, selector, pseudo) => {
    const realStyle = window.getComputedStyle.bind(window)
    const drawn = document.createElement('div').style
    drawn.setProperty('content', '""')
    Object.defineProperty(window, 'CSS', { value: { supports: () => true }, configurable: true })
    vi.spyOn(window, 'getComputedStyle').mockImplementation((element: Element, requested?: string | null) =>
      requested === pseudo && element.matches(selector) ? drawn : realStyle(element))
    try {
      renderChat('default')
      expect(await waitFor(theCaption)).toHaveTextContent(AUTO_LABEL)
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('paint')
      expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'paint')]])
    } finally {
      delete (window as { CSS?: unknown }).CSS
    }
  })

  it.each([
    // the words' line in jsdom's layout runs from 880 to 990: the old start, middle and end points were 882, 935 and 988
    ['a stripe between the start, middle and end of the words', null, (x: number) => x >= 890 && x <= 900],
    ['a stripe only a random point lands on', 0.05, (x: number) => x >= 886 && x <= 889],
  ])('brings the copy for %s: more points each line, and two new at random each check', async (_name, random, stripe) => {
    renderChat('default')
    expect(captionHost()).toBeNull()
    layout.cover = theLabel()
    layout.coverAt = (x) => stripe(x)
    if (random !== null) vi.spyOn(Math, 'random').mockReturnValue(random)
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { color: rgb(80, 80, 80); }`)
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'covered')]])
    expect(inputOf('default')).toBeEnabled()
  })

  // The ground under the fixed label is whatever the page lays under its corner, not only the label's ancestors:
  // a dark page (body) with light words, and an element of the page's under the corner (layout.under) that a theme paints.
  const DARK_PAGE = `html.${HOSTILE_CLASS} body { background-color: rgb(20, 20, 20); } html.${HOSTILE_CLASS} [role="note"] { color: rgb(236, 236, 236); }`

  /** An element of the page's laid out under the label (only at the points `at` says, if it says); returns it. */
  function layUnder(at: ((x: number, y: number) => boolean) | null = null, parent: Element = document.body): HTMLElement {
    const under = parent.appendChild(document.createElement('div'))
    under.setAttribute('data-test-under', '')
    layout.under = under
    layout.underAt = at
    return under
  }

  it.each([
    ["an element under the corner in the words' own colour", 'background-color: rgb(236, 236, 236);', null],
    ["a gradient in the words' own colour on an element under the corner", 'background-image: linear-gradient(to right, rgb(236, 236, 236), rgb(236, 236, 236));', null],
    ['an image under the corner, which no check can read', 'background-image: url("ground.png");', null],
    // the words' line in jsdom's layout runs from 880 to 990: one of the evenly spaced points sits at about 895
    ["a patch in the words' own colour under one point of them", 'background-color: rgb(236, 236, 236);', (x: number) => x >= 894 && x <= 897],
  ])('brings the copy for %s: the ground is read from what paints under the words', async (_name, paint, at) => {
    renderChat('default')
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(DARK_PAGE)
    expect(aiLabelHiddenReason(theLabel(), window)).toBeNull()
    const under = layUnder(at)
    try {
      await hideWith(`html.${HOSTILE_CLASS} [data-test-under] { ${paint} }`)
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('color')
      expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'color')]])
      expect(inputOf('default')).toBeEnabled()
    } finally {
      under.remove()
    }
  })

  it("brings the copy for a ramp under the words whose middle is their own grey, though both its ends stand out from them", async () => {
    renderChat('default')
    const under = layUnder()
    try {
      // rgb(118, 118, 118) keeps 4:1 against the dark page, and against black and white each, but not against the grey half-way
      await hideWith(
        `html.${HOSTILE_CLASS} body { background-color: rgb(20, 20, 20); } html.${HOSTILE_CLASS} [role="note"] { color: rgb(118, 118, 118); }\n` +
          `html.${HOSTILE_CLASS} [data-test-under] { background-image: linear-gradient(to right, rgb(0, 0, 0), rgb(255, 255, 255)); }`,
      )
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('color')
    } finally {
      under.remove()
    }
  })

  it.each([
    ['its own opacity', (under: HTMLElement) => under.setAttribute('style', 'opacity: 0')],
    ["an ancestor's opacity", (under: HTMLElement) => under.parentElement?.setAttribute('style', 'opacity: 0')],
  ])("doesn't let an opaque element faded out by %s fake a ground the words stand out from", (_name, fade) => {
    renderChat('default')
    const wrapper = document.body.appendChild(document.createElement('div'))
    const under = layUnder(null, wrapper)
    try {
      // the page is nearly the words' own light colour; the black element under them makes them readable while it is painted
      document.documentElement.classList.add(HOSTILE_CLASS)
      addThemeSheet(
        `html.${HOSTILE_CLASS} body { background-color: rgb(236, 236, 236); } html.${HOSTILE_CLASS} [role="note"] { color: rgb(250, 250, 250); }\n` +
          `html.${HOSTILE_CLASS} [data-test-under] { background-color: rgb(0, 0, 0); }`,
      )
      expect(aiLabelHiddenReason(theLabel(), window)).toBeNull()
      fade(under)
      expect(aiLabelHiddenReason(theLabel(), window)).toBe('color')
    } finally {
      wrapper.remove()
    }
  })

  it("reads HUD's backdrop as built under the corner (its glow, grid and scan lines, on its ground) as no harm", () => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    // the ghost accent on the void ground at its brightest grid and glow (themes/hud: glow capped where its second text reads)
    // its scan lines over its glow and grid (one element here: the first layer paints on top)
    addThemeSheet(
      `html.${HOSTILE_CLASS} body { background-color: rgb(3, 5, 7); } html.${HOSTILE_CLASS} [role="note"] { color: rgb(147, 166, 175); }\n` +
        `html.${HOSTILE_CLASS} [data-test-under] { background-image:` +
        ' repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.18) 0px, rgba(0, 0, 0, 0.18) 1px, transparent 1px, transparent 3px),' +
        ' radial-gradient(640px 520px at 50% 46%, rgba(227, 236, 240, 0.2), transparent 70%),' +
        ' linear-gradient(rgba(227, 236, 240, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(227, 236, 240, 0.1) 1px, transparent 1px); }',
    )
    renderChat('default')
    const under = layUnder()
    try {
      expect(window.getComputedStyle(under).backgroundImage).toContain('radial-gradient')
      expect(aiLabelHiddenReason(theLabel(), window)).toBeNull()
      expect(captionHost()).toBeNull()
      expect(hostWarnings()).toEqual([])
    } finally {
      under.remove()
    }
  })

  it('treats the label as hidden when its check throws (a getComputedStyle a theme patched), and shows the copy', async () => {
    const patched = new Error('getComputedStyle patched by a theme')
    vi.spyOn(window, 'getComputedStyle').mockImplementation(() => {
      throw patched
    })
    render(h(AiLabelGate, { children: h('div', null, 'a chat') }))
    expect(await waitFor(theCaption)).toHaveTextContent(AUTO_LABEL)
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: ['page'], shownLabels: ['page'] })
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning('page', 'unchecked'), patched]])
  })

  it("the SDK's cards.AiLabelFence disables nothing: the fixture theme's own box and Send keep working while the label is hidden", async () => {
    renderChat('fixture')
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    expect(inputOf('fixture')).toBeEnabled()
    expect(screen.getByRole('button', { name: FIXTURE_SEND, hidden: true })).toBeEnabled()
    expect(screen.getByTestId(FIXTURE_THEME_TEST_IDS.chat).querySelector('fieldset')).toBeNull()
    expect(screen.getByTestId(FIXTURE_THEME_TEST_IDS.chat).contains(theLabel())).toBe(false)
  })

  it('brings the copy when a hiding theme is switched in after mount, warns once, and takes it away when the theme goes', async () => {
    renderChat('theme')
    expect(captionHost()).toBeNull()
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    expect(inputOf('theme')).toBeEnabled()
    window.dispatchEvent(new Event('resize'))
    expect(hostWarnings()).toHaveLength(1)
    document.documentElement.classList.remove(HOSTILE_CLASS)
    await waitFor(() => expect(captionHost()).toBeNull())
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: [], shownLabels: ['page'] })
    expect(inputOf('theme')).toBeEnabled()
  })

  it('takes its copy with it when the chat route goes, and leaves no label shown (EUAI-11b)', async () => {
    const view = renderChat('fixture')
    await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: ['page'], shownLabels: ['page'] })
    view.unmount()
    expect(captionHost()).toBeNull()
    // the theme SDK's chat starts no turn without a label: a gate that goes leaves none shown
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: [], shownLabels: [] })
  })

  it("shows the label on a theme route such as Conversations (its own gate, on the left), and its copy on the left too", async () => {
    render(h(AiLabelGate, { placement: 'bottom-left', children: h('ul', { 'aria-label': 'Conversations' }) }))
    expect(screen.getByRole('note', { name: AUTO_LABEL })).toHaveClass('left-3')
    const caption = await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { visibility: hidden; }`)
    expect(caption.style.left).not.toBe('auto')
    expect(caption.style.right).toBe('auto')
  })
})

describe('the label names the agent answering (PRD-264 EUAI-11a)', () => {
  function PicksAgent({ agentId }: { readonly agentId: number }) {
    usePublishAnsweringAgent(agentId)
    return defaultChat()
  }

  it("says the picked agent's name while it answers, and Auto again once the chat goes", () => {
    const view = render(h(ChatLayout, null, h(PicksAgent, { agentId: 7 })))
    expect(screen.getByRole('note', { name: 'Research Desk is an AI' })).toHaveTextContent('Research Desk is an AI')
    view.rerender(h(ChatLayout, null, h('div', null, 'no chat')))
    expect(screen.getByRole('note', { name: AUTO_LABEL })).toBeInTheDocument()
  })

  it.each([8, 99])('still says it is an AI when agent %i has no name to show', (agentId) => {
    render(h(ChatLayout, null, h(PicksAgent, { agentId })))
    expect(screen.getByRole('note', { name: 'This agent is an AI' })).toBeInTheDocument()
  })

  it('reads its words from one place', () => {
    expect(aiLabelText(null)).toBe(AUTO_LABEL)
    expect(aiLabelText('  ')).toBe(AUTO_LABEL)
    expect(aiLabelText('Research Desk')).toBe('Research Desk is an AI')
    expect(aiLabelHiddenWarning('page', 'display')).toBe(
      'the active theme hides the page AI label (display); the host draws it in its own layer instead',
    )
    expect(aiLabelHiddenReason(null, window)).toBe('detached')
    expect(aiLabelHiddenReason(document.createElement('p'), window)).toBe('detached')
  })
})
