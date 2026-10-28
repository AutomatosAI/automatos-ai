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
 * same words. Then hostile fixture themes set `display:none`, `visibility:hidden`, `opacity:0`,
 * zero size or an off-screen position on the label (and a fade on an ancestor): each time the label
 * reports itself hidden, the host draws the same words in its own layer (components/ai-label/
 * protected-caption.tsx: a shadow root on an element of the host's at the end of <body>, guarded
 * inline), a theme-host warning names what the theme did, and the chat keeps working: its input and
 * a card's button stay enabled, and nothing says the chat is paused. A sheet aimed at that copy
 * itself leaves it on screen. A theme switched in after mount (the observer on <html> and <head>)
 * brings the copy, and switching back takes it away.
 *
 * jsdom applies <style> rules and inline styles in getComputedStyle (display, visibility,
 * opacity, width, height, top, left) but lays nothing out, so getBoundingClientRect returns
 * zeros. It is mocked explicitly: each element's box comes from its computed left, top, width
 * and height, a fixed label-sized box where those are unset. A theme's `width: 0px` or
 * `top: -500px` therefore reaches the check the way a browser's layout would hand it over.
 * jsdom also matches a document's rules against elements inside a shadow root, which a browser
 * never does, so its verdict on the host's copy is harsher than a browser's; jsdom has no top
 * layer, so that half is e2e/browser/specs/installed-play-ai-label.spec.ts's.
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
import { AI_LABEL_CAPTION_ATTR } from '@/components/ai-label/protected-caption'
import { aiLabelHiddenWarning } from '@/components/ai-label/use-ai-label-check'
import { MultimodalInput } from '@/components/chatbot/multimodal-input'
import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { aiLabelText } from '@/lib/ai-label/copy'
import { useAiLabelStore, usePublishAnsweringAgent } from '@/lib/ai-label/store'
import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'
import { AiLabelFence as ThemeAiLabelFence } from '@/lib/theme-sdk/host-cards/ai-label-fence'
import { CHAT_HREF } from '@/lib/ticket-links'

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
const CAPTION_HOST = `[${AI_LABEL_CAPTION_ATTR}]`
const THEME_HOST_LOG = '[theme-host]'

/** A computed length in px, or the fallback when the style leaves it unset. */
function px(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value)
  return value.trim().endsWith('px') && Number.isFinite(parsed) ? parsed : fallback
}

function boxOf(element: Element): DOMRect {
  const style = window.getComputedStyle(element)
  const left = px(style.left, LABEL_BOX.left)
  const top = px(style.top, LABEL_BOX.top)
  const width = px(style.width, LABEL_BOX.width)
  const height = px(style.height, LABEL_BOX.height)
  return { x: left, y: top, left, top, width, height, right: left + width, bottom: top + height, toJSON: () => ({}) } as DOMRect
}

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

/** The host's element that holds the label's copy in its own layer, or null while there is none. */
function captionHost(): HTMLElement | null {
  const hosts = document.querySelectorAll<HTMLElement>(CAPTION_HOST)
  expect(hosts.length).toBeLessThanOrEqual(1)
  return hosts[0] ?? null
}

/** The label's words in the host's own layer: inside the host element's shadow root. */
function theCaption(): HTMLElement {
  const caption = captionHost()?.shadowRoot?.querySelector<HTMLElement>('[role="note"]')
  if (!caption) throw new Error("the host drew no copy of the label in its own layer")
  return caption
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

beforeEach(() => {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    return boxOf(this)
  })
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
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
    expect(aiLabelHiddenReason(caption, window)).toBeNull()
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
    // a document's selectors don't reach inside the host's shadow root: only the theme's hidden label answers
    expect(document.querySelectorAll(`[aria-label="${AUTO_LABEL}"]`)).toHaveLength(1)
    expect(theCaption().getAttribute('popover')).toBe('manual')
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
    expect(aiLabelHiddenReason(theCaption(), window)).toBeNull()
    expect(theCaption()).toHaveTextContent(AUTO_LABEL)
    // the sheet hides the page with <body>; the composer is still not disabled
    expect(screen.getByRole('textbox', { name: THEME_COMPOSER, hidden: true })).toBeEnabled()
  })

  it('raises the copy into the browser\'s top layer where the browser has one', async () => {
    const raised: Element[] = []
    const showPopover = function (this: Element) {
      raised.push(this)
    }
    Object.defineProperty(HTMLElement.prototype, 'showPopover', { value: showPopover, configurable: true })
    try {
      renderChat('default')
      const caption = await hideWith(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
      expect(raised).toEqual([caption])
    } finally {
      delete (HTMLElement.prototype as { showPopover?: unknown }).showPopover
    }
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
