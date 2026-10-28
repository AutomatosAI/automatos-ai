/**
 * PRD-264 EUAI-11a (EU AI Act Art. 50(1), (5)) — Auto's floating widget, the in-app chat MainLayout
 * draws on every page but /chat (theme-drawn pages among them), shows "Auto is an AI" too, and no
 * theme can remove it.
 *
 * The real AutoWidget renders here, its data sources stubbed as in
 * components/layout/__tests__/main-layout-widget-page.test.tsx. Opened, its panel shows the host's
 * label above the conversation, with the input and the starter chips enabled. Hostile fixture
 * sheets then hide the label (display, visibility, opacity, size, an off-screen move, a fade on an
 * ancestor, transparent words, a clip-path): each time the host draws the same words at the top of the panel in its own layer
 * (components/ai-label/protected-caption.tsx), a theme-host warning names what the theme did, and
 * the widget keeps working: its input and chips stay enabled, and nothing says it is paused.
 * Closing the panel takes the copy away.
 *
 * jsdom lays nothing out, so ai-label-layout.ts installs what a browser would hand the check, as in
 * ai-label.test.ts.
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
import { aiLabelHiddenWarning } from '@/components/ai-label/use-ai-label-check'
import { WIDGET_LABEL_SETTLE_MS } from '@/components/ai-label/widget-ai-label'
import { AutoWidget } from '@/components/chatbot/chat-widget'
import { useAiLabelStore, WIDGET_AI_LABEL } from '@/lib/ai-label/store'
import { aiLabelHiddenReason } from '@/lib/ai-label/visibility'

import { captionHost, installLabelLayout, theCaption } from './ai-label-layout'

const AUTO_LABEL = 'Auto is an AI'
const WIDGET_PLACEHOLDER = 'Ask Auto anything...'
const STARTER_CHIP = 'Where did we leave off?'
const PAGE_INPUT = 'Page chat input'
const HOSTILE_CLASS = 'hostile-theme'
const LABEL = `[${AI_LABEL_ATTR}]`
const THEME_HOST_LOG = '[theme-host]'
/** Past the panel's settle delay, with room for the check to land. */
const SETTLED = { timeout: WIDGET_LABEL_SETTLE_MS + 1500 }
/** Where a label-sized box sits in jsdom's 1024 x 768 window when no style moves or sizes it. */
const LABEL_BOX = { left: 40, top: 300, width: 110, height: 20 }

function addThemeSheet(css: string): void {
  const sheet = document.createElement('style')
  sheet.setAttribute('data-test-theme', '')
  sheet.textContent = css
  document.head.appendChild(sheet)
}

/** Auto's widget beside a page chat's composer. */
async function openWidget(): Promise<void> {
  render(
    <>
      <input aria-label={PAGE_INPUT} />
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

/** The class names on `element` that would draw a pill: a rounding, a border, a background, a shadow or a ring. */
function pillClasses(element: Element): string[] {
  return [...element.classList].filter((name) => /^(rounded|border|bg-|shadow|ring)/.test(name))
}

/** The theme-host warnings logged so far. */
function hostWarnings(): unknown[][] {
  return vi.mocked(console.warn).mock.calls.filter(([prefix]) => prefix === THEME_HOST_LOG)
}

let uninstallLayout: () => void

beforeEach(() => {
  ;({ uninstall: uninstallLayout } = installLabelLayout(LABEL_BOX))
  vi.spyOn(console, 'warn').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  uninstallLayout()
  vi.restoreAllMocks()
  document.head.querySelectorAll('[data-test-theme]').forEach((sheet) => sheet.remove())
  document.documentElement.classList.remove(HOSTILE_CLASS)
  useAiLabelStore.setState({ agentId: null, hiddenLabels: [], shownLabels: [] })
})

describe("the AI label in Auto's floating widget (PRD-264 EUAI-11a)", () => {
  it('shows "Auto is an AI" in the open panel, above a conversation that runs from the start', async () => {
    await openWidget()
    const note = await screen.findByRole('note', { name: AUTO_LABEL })
    expect(note).toHaveTextContent(AUTO_LABEL)
    expect(theLabel()).toBe(note)
    // a tiny muted caption, never a pill
    expect(pillClasses(note)).toEqual([])
    expect(note).toHaveClass('text-[11px]', 'text-muted-foreground')
    // never held: the input and the chips run before the first check and after it
    expect(widgetInput()).toBeEnabled()
    expect(screen.getByRole('button', { name: STARTER_CHIP })).toBeEnabled()
    expect(note.closest('fieldset')).toBeNull()
    expect(widgetInput().closest('fieldset')).toBeNull()
    await waitFor(() => expect(useAiLabelStore.getState().shownLabels).toEqual([WIDGET_AI_LABEL]), SETTLED)
    // the caption's own styles never read as hidden
    expect(aiLabelHiddenReason(note, window)).toBeNull()
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    expect(captionHost()).toBeNull()
    expect(hostWarnings()).toEqual([])
  })
})

const HOSTILE_RULES: ReadonlyArray<readonly [string, string, string]> = [
  ['display:none', `html.${HOSTILE_CLASS} ${LABEL} { display: none; }`, 'display'],
  ['visibility:hidden', `html.${HOSTILE_CLASS} ${LABEL} { visibility: hidden; }`, 'visibility'],
  ['opacity:0', `html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`, 'opacity'],
  ['zero size', `html.${HOSTILE_CLASS} ${LABEL} { width: 0px; height: 0px; }`, 'size'],
  ['an off-screen move', `html.${HOSTILE_CLASS} ${LABEL} { position: fixed; top: -500px; }`, 'viewport'],
  ['a fade on an ancestor', `html.${HOSTILE_CLASS} body { opacity: 0.2; }`, 'opacity'],
  ['transparent words', `html.${HOSTILE_CLASS} [role="note"] { color: transparent; }`, 'color'],
  ['a clip-path', `html.${HOSTILE_CLASS} [role="note"] { clip-path: inset(50%); }`, 'paint'],
]

describe("no theme can hide the widget's AI label (PRD-264 EUAI-11a)", () => {
  it.each(HOSTILE_RULES)("a theme that sets %s on it gets the label drawn in the host's own layer, and the widget runs", async (_name, css, reason) => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(css)
    await openWidget()
    const label = theLabel()
    expect(label).toHaveTextContent(AUTO_LABEL)
    expect(aiLabelHiddenReason(label, window)).toBe(reason)
    const caption = await waitFor(theCaption, SETTLED)
    expect(caption).toHaveTextContent(AUTO_LABEL)
    expect(window.getComputedStyle(caption).display).toBe('block')
    // at the top of the panel's conversation, where the label sits, inside the viewport
    expect(caption.style.top).toMatch(/px$/)
    expect(caption.style.left).toMatch(/px$/)
    expect(widgetInput()).toBeEnabled()
    expect(screen.getByRole('button', { name: STARTER_CHIP, hidden: true })).toBeEnabled()
    expect(screen.queryByRole('alert', { hidden: true })).toBeNull()
    expect(useAiLabelStore.getState()).toMatchObject({ hiddenLabels: [WIDGET_AI_LABEL], shownLabels: [WIDGET_AI_LABEL] })
    expect(hostWarnings()).toEqual([[THEME_HOST_LOG, aiLabelHiddenWarning(WIDGET_AI_LABEL, reason)]])
    expect(screen.getByRole('textbox', { name: PAGE_INPUT, hidden: true })).toBeEnabled()
  })

  it('brings the copy when a hiding theme is switched in after the panel opened, and takes it away when the theme goes', async () => {
    await openWidget()
    await waitFor(() => expect(useAiLabelStore.getState().shownLabels).toEqual([WIDGET_AI_LABEL]), SETTLED)
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { display: none; }`)
    document.documentElement.classList.add(HOSTILE_CLASS)
    await waitFor(theCaption)
    expect(widgetInput()).toBeEnabled()
    document.documentElement.classList.remove(HOSTILE_CLASS)
    await waitFor(() => expect(captionHost()).toBeNull())
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    expect(widgetInput()).toBeEnabled()
  })

  it('takes its copy with it when the panel closes', async () => {
    document.documentElement.classList.add(HOSTILE_CLASS)
    addThemeSheet(`html.${HOSTILE_CLASS} ${LABEL} { opacity: 0; }`)
    await openWidget()
    await waitFor(theCaption, SETTLED)
    fireEvent.keyDown(window, { key: 'Escape' })
    await waitFor(() => expect(document.querySelector(LABEL)).toBeNull())
    expect(captionHost()).toBeNull()
    expect(useAiLabelStore.getState().hiddenLabels).toEqual([])
    // released, not marked seen: a closed panel's label is not on screen
    expect(useAiLabelStore.getState().shownLabels).toEqual([])
  })
})
