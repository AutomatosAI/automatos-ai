/**
 * PRD-252 R2 — Discuss on a ticket opens /chat?ticket=<id>: a new conversation
 * with the ticket in its context, a bar that says so, and "Update ticket and
 * re-queue". A mission's ticket points at the mission instead (D4).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup, act } from '@testing-library/react'
import type { BoardTask } from '@/types/board'

const nav = vi.hoisted(() => ({ query: 'ticket=612', replace: vi.fn() }))
const board = vi.hoisted(() => ({ task: undefined as unknown, isError: false }))
const session = vi.hoisted(() => ({ newDraft: vi.fn() }))
// PRD-260 M67: which chat /chat shows (Play's at >= 1024 px under Play, else the core's).
const look = vi.hoisted(() => ({ play: false, narrow: false, ownsChat: true }))

vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(nav.query),
  useRouter: () => ({ replace: nav.replace }),
}))
vi.mock('next/link', () => ({ default: ({ href, children, ...rest }: any) => <a href={String(href)} {...rest}>{children}</a> }))
vi.mock('@/hooks/use-board-tasks', () => ({ useBoardTask: () => ({ data: board.task, isError: board.isError }) }))
// The theme's width source, the one ThemeSlot reads (P260-FX-25): Play's chat only under Play at >= 1024 px.
// PRD-261 A-02b: the bar asks whether the active theme's entry owns /chat; the fixture theme's does (A-19b).
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return {
    useIsStudio: () => true,
    useActiveTheme: () => (look.play ? { ...FIXTURE_THEME_ENTRY, ownsPages: look.ownsChat ? FIXTURE_THEME_ENTRY.ownsPages : [] } : null),
    useThemeLayout: () => (look.play && !look.narrow ? 'theme' : 'fallback'),
  }
})
// A small stand-in for the chat-session store: an open tab, and a draft Discuss opens.
const { OPEN_TAB } = vi.hoisted(() => ({
  OPEN_TAB: { activeChatId: 'chat-old', openChatIds: ['chat-old'], titles: { 'chat-old': 'Rota questions' }, draftOpen: false },
}))
vi.mock('@/stores/chat-session-store', async () => {
  const { create } = await import('zustand')
  const store: any = create(() => ({ hydrated: true, session: OPEN_TAB, newDraft: () => {} }))
  store.setState({
    newDraft: () => {
      session.newDraft()
      store.setState((s: any) => ({ session: { ...s.session, activeChatId: null, draftOpen: true } }))
    },
  })
  return { useChatSessionStore: store }
})
vi.mock('../rebrief-dialog', () => ({ RebriefDialog: () => <div data-testid="rebrief-dialog" /> }))

import { DiscussionBar } from '../discussion-bar'
import { useDiscussionStore } from '@/stores/discussion-store'
import { useChatSessionStore } from '@/stores/chat-session-store'

function activate(chatId: string, alsoOpen: string[] = []) {
  act(() => {
    (useChatSessionStore as any).setState((s: any) => ({
      session: { ...s.session, activeChatId: chatId, draftOpen: false, openChatIds: [...s.session.openChatIds, ...alsoOpen] },
    }))
  })
}

function ticket(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '612', type: 'task', name: 'Welcome email', status: 'review', priority: 'medium', tags: [],
    review_mode: 'human', source_id: '612', number: '#0042', assignee: { agent_id: 3, agent_name: 'Words' }, ...over }
}

beforeEach(() => {
  nav.query = 'ticket=612'
  nav.replace.mockReset()
  session.newDraft.mockReset()
  board.task = ticket()
  board.isError = false
  look.play = false
  look.narrow = false
  look.ownsChat = true
  useDiscussionStore.getState().end()
  ;(useChatSessionStore as any).setState({ session: OPEN_TAB })
})
afterEach(cleanup)

describe('DiscussionBar', () => {
  it('starts the discussion in a new conversation and says which ticket it is about', () => {
    render(<DiscussionBar />)
    expect(screen.getByTestId('discussion-bar')).toHaveTextContent('Discussing ticket #0042 · Welcome email')
    expect(screen.getByText('ticket #0042').closest('a')).toHaveAttribute('href', '/command-center?tab=board&task_id=612')
    expect(session.newDraft).toHaveBeenCalledTimes(1)
    expect(useDiscussionStore.getState().discussion).toMatchObject({ ticketId: '612', missionId: null })
    expect(screen.getByText(/To ask Words instead/)).toBeInTheDocument()
  })

  it('opens "Update ticket and re-queue"', () => {
    render(<DiscussionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'Update ticket and re-queue' }))
    expect(screen.getByTestId('rebrief-dialog')).toBeInTheDocument()
  })

  it("points a mission's ticket at the mission, with nothing to re-queue (D4)", () => {
    board.task = ticket({ type: 'mission', mission_id: 'run-9' })
    render(<DiscussionBar />)
    expect(screen.getByText('Decide on the mission →').closest('a')).toHaveAttribute('href', '/missions/run-9')
    expect(screen.queryByRole('button', { name: 'Update ticket and re-queue' })).toBeNull()
  })

  it('ends: the ticket leaves the chat and its link', () => {
    render(<DiscussionBar />)
    fireEvent.click(screen.getByRole('button', { name: 'End the discussion' }))
    expect(useDiscussionStore.getState().discussion).toBeNull()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
  })

  it("leaves a session ticket's canvas link alone, and says when the ticket cannot be opened", () => {
    nav.query = 'repo=projects%2Fsite&ticket=612&runtime=1'
    const { container } = render(<DiscussionBar />)
    expect(container).toBeEmptyDOMElement()
    expect(session.newDraft).not.toHaveBeenCalled()
    cleanup()
    nav.query = 'ticket=999'
    board.task = undefined
    board.isError = true
    render(<DiscussionBar />)
    expect(screen.getByTestId('discussion-bar')).toHaveTextContent('Ticket 999 could not be opened here')
  })

  it('stays with the conversation its draft becomes (review of #861)', () => {
    render(<DiscussionBar />)
    activate('chat-new', ['chat-new'])                       // the first message named the draft
    expect(useDiscussionStore.getState()).toMatchObject({ chatId: 'chat-new', discussion: { ticketId: '612' } })
    expect(nav.replace).not.toHaveBeenCalled()
  })

  it('ends when another conversation is opened, so the ticket never reaches it', () => {
    render(<DiscussionBar />)
    activate('chat-old')                                     // back to the tab that was open
    expect(useDiscussionStore.getState().discussion).toBeNull()
    expect(nav.replace).toHaveBeenCalledWith('/chat')
  })
})

const STUDIO_HINT = 'You are talking to Auto. To ask Words instead, pick them in the message box.'
const PLAY_HINT = "You are talking to Auto in Play's chat. To ask Words, switch to Studio from the theme menu."

describe("DiscussionBar's hint (PRD-260 M67)", () => {
  it("under Play, never sends the owner to an agent picker Play's message box doesn't have", () => {
    look.play = true
    render(<DiscussionBar />)
    expect(screen.queryByText(/pick them in the message box/)).toBeNull()
    expect(screen.getByText(PLAY_HINT)).toBeInTheDocument()
  })

  it("keeps Studio's hint under Studio, and under Play below 1024 px where /chat is the core's chat (D7)", () => {
    render(<DiscussionBar />)
    expect(screen.getByText(STUDIO_HINT)).toBeInTheDocument()
    cleanup()
    look.play = true
    look.narrow = true
    render(<DiscussionBar />)
    expect(screen.getByText(STUDIO_HINT)).toBeInTheDocument()
    expect(screen.queryByText(PLAY_HINT)).toBeNull()
  })

  it("keeps Studio's hint under a theme at its width whose entry doesn't own /chat (PRD-261 A-02b)", () => {
    look.play = true
    look.ownsChat = false
    render(<DiscussionBar />)
    expect(screen.getByText(STUDIO_HINT)).toBeInTheDocument()
    expect(screen.queryByText(PLAY_HINT)).toBeNull()
  })

  it("shows no hint on a mission's ticket, under Play as under Studio (D4)", () => {
    look.play = true
    board.task = ticket({ type: 'mission', mission_id: 'run-9' })
    render(<DiscussionBar />)
    expect(screen.queryByText(/You are talking to Auto/)).toBeNull()
  })
})
