/**
 * PRD-246 US-003 — the Studio chat's compact form.
 *
 * On a phone the chat is the conversation and nothing else: the threads list
 * and the Auto now rail are the SAME components, moved behind the bar's two
 * controls into the app's one `Sheet`. Width is mocked through
 * `@/hooks/use-mobile` for the 1024 fork, and through `window.matchMedia` for
 * the rail's own 1280 threshold (PRD-244 D5's, which the stylesheet mirrors).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { readdirSync, readFileSync } from 'fs'
import path from 'path'

const width = vi.hoisted(() => ({ compact: false, railInSheet: false }))

vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => width.compact,
  useIsTabletOrBelow: () => width.compact,
}))
vi.mock('@/hooks/use-auto-now', () => ({
  AUTO_NOW_RAIL_MIN_WIDTH: 1280,
  useAutoNow: () => ({ questionCount: 0, decisionsTotal: 0 }),
}))
vi.mock('../auto-now-rail', () => ({
  AutoNowRail: () => <div data-testid="auto-now-rail" />,
}))
vi.mock('@/hooks/use-missions-api', () => ({
  useMission: () => ({ data: null, isLoading: false }),
}))
vi.mock('@/lib/chat/api', () => ({
  getChat: vi.fn(),
  getChatMessages: vi.fn(),
  getChatHistory: vi.fn().mockResolvedValue([]),
}))

import { StudioChatShell } from '../studio-chat-shell'

const shell = () => (
  <StudioChatShell
    selectedChatId="chat-1"
    selectedChat={null}
    onSelectChat={vi.fn()}
    onNewChat={vi.fn()}
    titles={{ 'chat-1': 'Quarterly close' }}
  >
    <div data-testid="conversation" />
  </StudioChatShell>
)

beforeEach(() => {
  width.compact = false
  width.railInSheet = false
  // PRD-244 D5's mount default reads window.innerWidth, which jsdom pins at
  // 1024 — state the width both ways so the two cases are honest.
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1440 })
  // The rail's 1280 threshold is read with matchMedia, so the query answers
  // per test rather than through the suite-wide no-match shim.
  window.matchMedia = ((query: string) => ({
    matches: width.railInSheet && query.includes('1279'),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  })) as unknown as typeof window.matchMedia
})
afterEach(cleanup)

describe('the threads list has one home per width', () => {
  it('desktop: the threads column is in the grid', () => {
    const { container } = render(shell())
    expect(container.querySelector('.sh-chat-grid .sh-chat-threads')).not.toBeNull()
  })

  it('compact: nothing but the conversation is in the grid', () => {
    width.compact = true
    const { container } = render(shell())
    expect(container.querySelector('.sh-chat-grid .sh-chat-threads')).toBeNull()
    expect(container.querySelector('.sh-chat-grid .sh-chat-main')).not.toBeNull()
    expect(screen.getByTestId('conversation')).toBeInTheDocument()
  })

  it('compact: the bar’s toggle opens the same list in a Sheet', () => {
    width.compact = true
    render(shell())
    expect(document.querySelector('.sh-chat-threads')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show threads' }))
    // The sheet is a portal, so it is outside the render container.
    const panel = document.querySelector('[role="dialog"] .sh-chat-threads')
    expect(panel).not.toBeNull()
    expect(panel!.querySelector('.sh-chat-threads-head')).not.toBeNull()
    expect(panel!.closest('[role="dialog"]')!.className).toContain('safe-bottom')
  })
})

describe('the Auto now pill opens the rail below 1280', () => {
  it('wide: the rail is the grid’s third column', () => {
    const { container } = render(shell())
    expect(container.querySelector('.sh-chat-grid aside.sh-chat-rail')).not.toBeNull()
    expect(screen.getByTestId('auto-now-rail')).toBeInTheDocument()
  })

  it('below 1280: the pill opens the rail’s content in a Sheet, and there is still one rail', () => {
    width.railInSheet = true
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 900 })
    render(shell())
    expect(document.querySelector('.sh-chat-rail')).toBeNull()
    expect(screen.queryByTestId('auto-now-rail')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show Auto now rail' }))
    const rail = document.querySelector('[role="dialog"] .sh-chat-rail')
    expect(rail).not.toBeNull()
    expect(screen.getAllByTestId('auto-now-rail')).toHaveLength(1)
    expect(rail!.closest('[role="dialog"]')!.className).toContain('safe-bottom')
  })

  it('a collapse preference stored on a desktop never opens a sheet on a phone', () => {
    window.localStorage.setItem('studioChatRailCollapsed', '0')
    window.localStorage.setItem('studioChatThreadsCollapsed', '0')
    width.compact = true
    width.railInSheet = true
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 390 })
    render(shell())
    expect(document.querySelector('[role="dialog"]')).toBeNull()
    window.localStorage.clear()
  })
})

describe('no second primitive', () => {
  it('adds no drawer component — the rail and the threads use the one Sheet', () => {
    const src = readFileSync(
      path.join(path.resolve(__dirname, '..'), 'studio-chat-shell.tsx'),
      'utf8',
    )
    expect(src).toContain("from '@/components/ui/sheet'")
    const drawers = readdirSync(path.resolve(__dirname, '..', '..')).filter((f) => f.endsWith('-drawer.tsx'))
    expect(drawers).toEqual([])
    // One rail component and one threads panel, each rendered in either home.
    expect(src.match(/<AutoNowRail \/>/g) ?? []).toHaveLength(1)
    expect(src.match(/\{threadsPanel\}/g) ?? []).toHaveLength(2)
    expect(src.match(/\{railPanel\}/g) ?? []).toHaveLength(2)
  })

  it('the composer stays in flow, so the conversation never hides under it', () => {
    const chat = readFileSync(
      path.join(path.resolve(__dirname, '..'), 'chat.tsx'),
      'utf8',
    )
    // `sticky` keeps the composer in the flex column: the scroll container is
    // its sibling and ends above it. The home-bar inset is reserved ONCE, by
    // `.sh-shell.safe-bottom` (US-001) — the chat's ancestor in the Studio
    // shell — so the composer must not add a second one.
    expect(chat).toContain('className="sticky bottom-0 z-10')
    expect(chat).not.toContain('safe-bottom')
    expect(
      readFileSync(path.resolve(__dirname, '..', '..', 'layout', 'main-layout.tsx'), 'utf8'),
    ).toContain('className="sh-shell safe-bottom"')
  })
})
