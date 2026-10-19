/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — every in-app chat shows "Auto is an AI", and no
 * theme can remove it.
 *
 * app/chat/layout.tsx is rendered for real, with the default chat (its input, the one Mission
 * Mode uses too: chat.tsx draws FencedMultimodalInput), with builtin Play's chat (its message box,
 * through the lazy slot builtin Play registers for /chat, lib/theme-host/builtin/play-slots.tsx),
 * and with an installed fixture theme that registers /chat through the theme host (ThemeSlot draws
 * its page, the host's chat screen inside the theme's own markup and stylesheet, in MainLayout).
 * The label renders once, after ThemeSlot, outside the theme's page, with role="note" and the
 * same words. Then hostile fixture themes set `display:none`,
 * `visibility:hidden`, `opacity:0`, zero size or an off-screen position on the label (and a fade
 * on an ancestor): each time the label reports itself hidden, the chat's input is disabled with
 * the PRD's words, and a card's button outside the composer stays clickable (approval cards are
 * protected surfaces). A theme switched in after mount (the observer on <html> and <head>)
 * pauses the chat too, and switching back lifts the pause.
 *
 * jsdom applies <style> rules and inline styles in getComputedStyle (display, visibility,
 * opacity, width, height, top, left) but lays nothing out, so getBoundingClientRect returns
 * zeros. It is mocked explicitly: each element's box comes from its computed left, top, width
 * and height, a fixed label-sized box where those are unset. A theme's `width: 0px` or
 * `top: -500px` therefore reaches the check the way a browser's layout would hand it over.
 */
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { createElement as h, type ReactElement, type ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ThemeEntry } from '@/lib/theme-host/registry-types'

const state = vi.hoisted(() => ({
  entry: null as ThemeEntry | null,
  layout: 'fallback' as 'fallback' | 'theme' | 'pending',
  /** What builtin Play's /chat slot draws: Play's chat. */
  slot: null as null | (() => ReactElement),
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
vi.mock('@/components/play/chat/play-chat-lazy', () => ({
  PlayChat: () => (state.slot ? state.slot() : null),
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
import { AI_LABEL_FENCE_ATTR } from '@/components/ai-label/ai-label-fence'
import { FencedMultimodalInput } from '@/components/ai-label/fenced-multimodal-input'
import { MessageBox, PLACEHOLDER as PLAY_PLACEHOLDER } from '@/components/play/chat/message-box'
import { ThemeHostContext, type ThemeHostContextValue } from '@/components/theme-host/theme-host-context'
import { AI_LABEL_PAUSED_TEXT, aiLabelText } from '@/lib/ai-label/copy'
import { useAiLabelStore, usePublishAnsweringAgent } from '@/lib/ai-label/store'
import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'
import { BUILTIN_PLAY_ENTRY } from '@/lib/theme-host/builtin/play-entry'
import { CHAT_HREF } from '@/lib/ticket-links'

type Page = 'default' | 'play' | 'theme'

const PAGES: readonly Page[] = ['default', 'play', 'theme']
const AUTO_LABEL = 'Auto is an AI'
const THEME_CLASS = 'hello-chat'
const HOSTILE_CLASS = 'hostile-theme'
const CARD_BUTTON = 'Approve & run'
/** Where a label-sized box sits in jsdom's 1024 x 768 window when no style moves or sizes it. */
const LABEL_BOX = { left: 880, top: 740, width: 110, height: 20 }
const LABEL = `[${AI_LABEL_ATTR}]`

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
  return h(FencedMultimodalInput, {
    chatId: 'ai-label-test', status: 'ready', stop: vi.fn(), sendMessage: vi.fn(), selectedVisibilityType: 'private',
  })
}

/** A reply's card: a protected surface, outside the composer. */
function card(): ReactElement {
  return h('section', { 'data-testid': 'card' }, h('button', { type: 'button' }, CARD_BUTTON))
}

/** Builtin Play's chat, as the play slot draws it: a reply's card and Play's own message box. */
function playChat(): ReactElement {
  return h('main', { 'data-testid': 'play-chat' }, card(), h(MessageBox, { streaming: false, onSend: () => true, onStop: () => undefined }))
}

/** A fixture theme that registers /chat: the host's chat screen inside the theme's own page markup. */
function themeChat(): ReactElement {
  return h('div', { 'data-testid': 'theme-page', 'data-theme-page': '' }, h('header', null, 'Hello theme'), playChat())
}

function FixtureFrame({ children }: { readonly children?: ReactNode }): ReactElement {
  return h('div', { 'data-testid': 'theme-frame' }, children)
}

/** The theme host with an installed fixture theme active, its /chat page registered (registerPage). */
const FIXTURE_THEME_HOST: ThemeHostContextValue = {
  status: 'active',
  reason: null,
  frame: FixtureFrame,
  slots: new Map(),
  pages: new Map<string, unknown>([[CHAT_HREF, themeChat]]),
  routes: new Map(),
  reportRenderFailure: () => undefined,
}

function renderChat(page: Page) {
  state.entry = page === 'play' ? BUILTIN_PLAY_ENTRY : null
  state.layout = page === 'default' ? 'fallback' : 'theme'
  state.slot = page === 'play' ? playChat : null
  const chat = h(ChatLayout, null, h('div', { 'data-testid': 'core-chat' }, card(), defaultChat()))
  return render(page === 'theme' ? h(ThemeHostContext.Provider, { value: FIXTURE_THEME_HOST }, chat) : chat)
}

function inputOf(page: Page): HTMLElement {
  return page === 'default' ? screen.getByRole('textbox') : screen.getByRole('textbox', { name: PLAY_PLACEHOLDER })
}

function cardButton(): HTMLElement {
  return screen.getByRole('button', { name: CARD_BUTTON, hidden: true })
}

function theLabel(): HTMLElement {
  const labels = document.querySelectorAll<HTMLElement>(LABEL)
  expect(labels).toHaveLength(1)
  return labels[0]
}

beforeEach(() => {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    return boxOf(this)
  })
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  document.head.querySelectorAll('[data-test-theme]').forEach((sheet) => sheet.remove())
  document.documentElement.classList.remove(THEME_CLASS, HOSTILE_CLASS)
  useAiLabelStore.setState({ agentId: null, hiddenLabels: [] })
  state.entry = null
  state.layout = 'fallback'
  state.slot = null
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
    // after ThemeSlot, outside the page and the theme's markup: no page or theme draws it
    const pageRoot = screen.getByTestId(page === 'default' ? 'core-chat' : page === 'play' ? 'play-chat' : 'theme-page')
    expect(pageRoot.contains(note)).toBe(false)
    expect(note.closest(`[${AI_LABEL_FENCE_ATTR}]`)).toBeNull()
    if (page !== 'default') expect(screen.queryByTestId('core-chat')).toBeNull()
    // the composer sits in the label's fence, and runs while the label can be seen
    expect(inputOf(page).closest(`[${AI_LABEL_FENCE_ATTR}]`)).not.toBeNull()
    expect(inputOf(page)).toBeEnabled()
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it.each(['default', 'play'] as const)('on the %s chat no input renders a second copy of the label', (page) => {
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

  it.each(cases)('on the %s chat, a theme that sets %s on it pauses the chat', async (page, _name, css, reason) => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(css)
    renderChat(page)
    const label = theLabel()
    expect(label).toHaveTextContent(AUTO_LABEL)
    expect(aiLabelHiddenReason(label, window)).toBe(reason)
    await waitFor(() => expect(inputOf(page)).toBeDisabled())
    expect(screen.getByRole('alert', { hidden: true })).toHaveTextContent(AI_LABEL_PAUSED_TEXT)
    // a reply's card is not the composer: it stays clickable (a protected surface)
    expect(cardButton()).toBeEnabled()
  })

  it('pauses the chat when a hiding theme is switched in after mount, and lifts the pause when it goes', async () => {
    renderChat('theme')
    expect(inputOf('theme')).toBeEnabled()
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    document.documentElement.classList.add(HOSTILE_CLASS)
    await waitFor(() => expect(inputOf('theme')).toBeDisabled())
    expect(screen.getByRole('alert', { hidden: true })).toHaveTextContent(AI_LABEL_PAUSED_TEXT)
    document.documentElement.classList.remove(HOSTILE_CLASS)
    await waitFor(() => expect(inputOf('theme')).toBeEnabled())
    expect(screen.queryByRole('alert', { hidden: true })).toBeNull()
  })

  it('takes its pause with it when the chat route goes', async () => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
    const view = renderChat('play')
    await waitFor(() => expect(useAiLabelStore.getState().hiddenLabels).toEqual(['page']))
    view.unmount()
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
  })

  it('shows the label on a theme route such as Conversations (its own gate, on the left) and says so when it is hidden', async () => {
    render(h(AiLabelGate, { placement: 'bottom-left', children: h('ul', { 'aria-label': 'Conversations' }) }))
    expect(screen.getByRole('note', { name: AUTO_LABEL })).toHaveClass('left-3')
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { visibility: hidden; }`)
    document.documentElement.classList.add(HOSTILE_CLASS)
    await waitFor(() => expect(screen.getByRole('alert', { hidden: true })).toHaveTextContent(AI_LABEL_PAUSED_TEXT))
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
    expect(AI_LABEL_PAUSED_TEXT).toBe('This theme hides the AI notice, so chat is paused. Switch theme in Settings.')
    expect(aiLabelHiddenReason(null, window)).toBe('detached')
    expect(aiLabelHiddenReason(document.createElement('p'), window)).toBe('detached')
  })
})
