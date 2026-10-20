/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — Auto's floating widget, the in-app chat MainLayout
 * draws on every page but /chat (theme-drawn pages among them), shows "Auto is an AI" too, and no
 * theme can remove it.
 *
 * The real AutoWidget renders here, its data sources stubbed as in
 * components/layout/__tests__/main-layout-widget-page.test.tsx. Opened, its panel shows the host's
 * label above the conversation, outside the label's fence, with the input and the starter chips
 * inside it. Hostile fixture sheets then hide the label (display, visibility, opacity, size, an
 * off-screen move, a fade on an ancestor): each time the widget's input and chips are disabled and
 * the panel says why, while a page chat's composer, fenced by the page's label, keeps running.
 * Closing the panel lifts its pause.
 *
 * jsdom lays nothing out, so getBoundingClientRect is mocked from each element's computed left,
 * top, width and height, as in ai-label.test.ts.
 */
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next/navigation', () => ({
  usePathname: () => '/deliverables',
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
  AnimatePresence: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ user: null }) }))
vi.mock('@/hooks/use-bug-report-api', () => ({ useSubmitBugReport: () => ({ mutate: vi.fn() }) }))
vi.mock('@/lib/chat/hooks', () => ({
  useChat: () => ({ messages: [], isLoading: false, sendMessage: vi.fn(), stop: vi.fn() }),
}))
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'unknown-page', route: '/deliverables' }) }))
vi.mock('@/contexts/role-context', () => ({ useSystemRole: () => ({ isAdmin: false }) }))
vi.mock('@/lib/chat/api', () => ({
  getChat: vi.fn(),
  getChatHistory: vi.fn(() => Promise.resolve([])),
  getChatMessages: vi.fn(),
}))
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))
vi.mock('@/stores/chat-session-store', async () => {
  const { EMPTY_CHAT_SESSION } = await import('@/lib/chat/chat-session')
  // Stable actions: the widget's effects depend on them.
  const store = {
    session: EMPTY_CHAT_SESSION,
    openChat: () => {},
    newDraft: () => {},
    closeTab: () => {},
    chatIdAssigned: () => {},
    setTitles: () => {},
  }
  return { useChatSessionStore: (select: (s: typeof store) => unknown) => select(store) }
})

import { AI_LABEL_ATTR } from '@/components/ai-label/ai-label'
import { AI_LABEL_FENCE_ATTR, AiLabelFence } from '@/components/ai-label/ai-label-fence'
import { WIDGET_LABEL_SETTLE_MS } from '@/components/ai-label/widget-ai-label'
import { AutoWidget } from '@/components/chatbot/chat-widget'
import { AI_LABEL_PAUSED_TEXT } from '@/lib/ai-label/copy'
import { useAiLabelStore, WIDGET_AI_LABEL } from '@/lib/ai-label/store'
import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'

const AUTO_LABEL = 'Auto is an AI'
const WIDGET_PLACEHOLDER = 'Ask Auto anything...'
const STARTER_CHIP = 'Where did we leave off?'
const PAGE_INPUT = 'Page chat input'
const HOSTILE_CLASS = 'hostile-theme'
const LABEL = `[${AI_LABEL_ATTR}]`
const FENCE = `[${AI_LABEL_FENCE_ATTR}]`
/** Past the panel's settle delay, with room for the check to land. */
const SETTLED = { timeout: WIDGET_LABEL_SETTLE_MS + 1500 }
/** Where a label-sized box sits in jsdom's 1024 x 768 window when no style moves or sizes it. */
const LABEL_BOX = { left: 40, top: 300, width: 110, height: 20 }

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

/** Auto's widget beside a page chat's composer (fenced by the page's label, which isn't mounted here). */
async function openWidget(): Promise<void> {
  render(
    <>
      <AiLabelFence>
        <input aria-label={PAGE_INPUT} />
      </AiLabelFence>
      <AutoWidget position="bottom-right" currentPage="dashboard" />
    </>,
  )
  fireEvent.click(await screen.findByTitle('Ask Auto'))
}

function widgetInput(): HTMLElement {
  return screen.getByPlaceholderText(WIDGET_PLACEHOLDER)
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
  document.documentElement.classList.remove(HOSTILE_CLASS)
  useAiLabelStore.setState({ agentId: null, hiddenLabels: [], shownLabels: [] })
})

describe("the AI label in Auto's floating widget (PRD-264 EUAI-11a)", () => {
  it('shows "Auto is an AI" in the open panel, outside the fence, with the input inside it', async () => {
    await openWidget()
    const note = await screen.findByRole('note', { name: AUTO_LABEL })
    expect(note).toHaveTextContent(AUTO_LABEL)
    expect(theLabel()).toBe(note)
    expect(note.closest(FENCE)).toBeNull()
    expect(widgetInput().closest(FENCE)).not.toBeNull()
    expect(screen.getByRole('button', { name: STARTER_CHIP }).closest(FENCE)).not.toBeNull()
    // held until the label has been checked once the panel settles, then running
    await waitFor(() => expect(widgetInput()).toBeEnabled(), SETTLED)
    expect(screen.getByRole('button', { name: STARTER_CHIP })).toBeEnabled()
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('holds the input until the first check has run', async () => {
    await openWidget()
    expect(widgetInput()).toBeDisabled()
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

describe("no theme can hide the widget's AI label (PRD-264 EUAI-11a)", () => {
  it.each(HOSTILE_RULES)('a theme that sets %s on it pauses the widget only', async (_name, css, reason) => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(css)
    await openWidget()
    const label = theLabel()
    expect(label).toHaveTextContent(AUTO_LABEL)
    expect(aiLabelHiddenReason(label, window)).toBe(reason)
    await waitFor(() => expect(useAiLabelStore.getState().hiddenLabels).toEqual([WIDGET_AI_LABEL]), SETTLED)
    expect(widgetInput()).toBeDisabled()
    expect(screen.getByRole('button', { name: STARTER_CHIP, hidden: true })).toBeDisabled()
    expect(screen.getByRole('alert', { hidden: true })).toHaveTextContent(AI_LABEL_PAUSED_TEXT)
    // the page's chat is fenced by the page's label, not the widget's
    expect(screen.getByRole('textbox', { name: PAGE_INPUT, hidden: true })).toBeEnabled()
  })

  it('pauses the widget when a hiding theme is switched in after it opened, and lifts the pause when it goes', async () => {
    await openWidget()
    await waitFor(() => expect(widgetInput()).toBeEnabled(), SETTLED)
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    document.documentElement.classList.add(HOSTILE_CLASS)
    await waitFor(() => expect(widgetInput()).toBeDisabled())
    expect(screen.getByRole('alert', { hidden: true })).toHaveTextContent(AI_LABEL_PAUSED_TEXT)
    document.documentElement.classList.remove(HOSTILE_CLASS)
    await waitFor(() => expect(widgetInput()).toBeEnabled())
    expect(screen.queryByRole('alert', { hidden: true })).toBeNull()
  })

  it('takes its pause with it when the panel closes', async () => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
    await openWidget()
    await waitFor(() => expect(useAiLabelStore.getState().hiddenLabels).toEqual([WIDGET_AI_LABEL]), SETTLED)
    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(document.querySelector(LABEL)).toBeNull())
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    // released, not marked seen: a closed panel's label is not on screen
    expect(useAiLabelStore.getState().shownLabels).toEqual([])
  })
})
