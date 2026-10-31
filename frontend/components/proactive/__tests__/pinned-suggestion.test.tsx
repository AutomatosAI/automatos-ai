/**
 * PRD-268 §6.5.1, §6.12.3, §6.12.4 (US-W01b) — the newest open suggestion pinned above the composer: in the Auto
 * thread only (the chat's kind is 'auto'), only one addressed to the thread's owner, on one line that opens the card,
 * with Start and Hide on the line; hidden while a turn's Mission suggestion is unanswered (the presence store, which
 * the chat's MissionSuggestionCard wrapper feeds until it is launched, dismissed or unmounted); nothing in other
 * chats or with no suggestion open. chat.tsx changes only its two import lines.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type Mutation = { mutate: ReturnType<typeof vi.fn>; reset: ReturnType<typeof vi.fn>; isLoading: boolean; isError: boolean; error: unknown }
const state = vi.hoisted(() => ({
  list: undefined as unknown,
  start: null as unknown as Mutation,
  hide: null as unknown as Mutation,
  getChat: vi.fn(),
  composerProps: [] as unknown[],
  createMission: { mutate: vi.fn(), isLoading: false },
}))
vi.mock('@/hooks/use-suggestions', () => ({
  useSuggestions: () => ({ data: state.list }),
  useStartSuggestion: () => state.start,
  useHideSuggestion: () => state.hide,
}))
vi.mock('@/lib/chat/api', () => ({ getChat: (id: string) => state.getChat(id) }))
vi.mock('@/components/chatbot/multimodal-input', () => ({
  MultimodalInput: (props: { chatId: string }) => {
    state.composerProps.push(props)
    return <div data-testid="composer" data-chat={props.chatId} />
  },
}))
// The chat's Mission suggestion, as a stand-in whose "No thanks" answers it (the real card is tested below).
vi.mock('@/components/chatbot/mission-suggestion-card', () => ({
  MissionSuggestionCard: ({ goal, onAnswered }: { goal: string; onAnswered?: () => void }) => (
    <div data-testid="mission-card">
      {goal}
      <button type="button" onClick={onAnswered}>
        No thanks
      </button>
    </div>
  ),
}))
vi.mock('next/navigation', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/hooks/use-missions-api', () => ({ useCreateMission: () => state.createMission }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
// The consent line above the composer (US-W06) as a stand-in: the server offers no calibration here.
vi.mock('@/components/proactive/calibration-offer', () => ({ CalibrationOffer: () => null, useCalibrationOffered: () => false }))

import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'
import { MissionSuggestionCard } from '@/components/proactive/mission-suggestion-with-presence'
import {
  markMissionSuggestionOpen,
  missionSuggestionOpen,
  usePinnedSuggestionIds,
} from '@/components/proactive/mission-suggestion-presence'
import { newestFor } from '@/components/proactive/pinned-suggestion'
import { autoThreadQueryKey, retriesAfter } from '@/components/proactive/use-auto-thread'
import { STARTED_LINE, STARTED_MS } from '@/components/proactive/use-suggestion-card'
import type { Suggestion } from '@/lib/proactive/types'
import { HEX_COLOUR, MAX_COMPONENT_LINES, ORANGE_CLASS, ROUNDED_3XL, componentSpans, proactiveSource } from './source-rules'

const AUTO_CHAT = '3f2a9c10-7b4e-4d21-9a6f-1c2b3d4e5f60'
const USER_CHAT = '9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d'
const NEW_CHAT = 'c0ffee00-1234-4abc-8def-0123456789ab'
const FIRST = '7d1c0a52-3e0b-4c55-9a1e-2f6f0c1d9b10'
const SECOND = '0b6f3e21-94c7-4d0a-8e15-6a2d9c7f4b33'
const OWNER = 1

function suggestion(overrides: Partial<Suggestion> = {}): Suggestion {
  return {
    id: FIRST, workspace_id: 'ws-1', user_id: OWNER, source_type: 'check_in', source_agent_id: 4, run_id: null,
    title: 'Look at the Lisbon flights', detail: 'Three runs failed since Monday.', evidence: [],
    proposed_action: { kind: 'board_task', payload: {} }, subject_key: 'ticket:0419', severity: 'today',
    status: 'open', occurrences: 1, last_seen_at: null, started_ref: null, decided_via: null, hide_reason: null,
    decided_at: null, est_credits: null, created_at: null, delivered_at: null, expires_at: null,
    ...overrides,
  }
}

const second = (overrides: Partial<Suggestion> = {}) => suggestion({ id: SECOND, title: 'Renew the domain', ...overrides })
const mutation = (): Mutation => ({ mutate: vi.fn(), reset: vi.fn(), isLoading: false, isError: false, error: null })
const line = (title: string) => screen.findByRole('group', { name: `Auto suggests: ${title}` })

/** Shows which suggestions are pinned now, as the thread's cards read it. */
function PinnedProbe() {
  return <output data-testid="pinned-ids">{usePinnedSuggestionIds().join(',')}</output>
}

const CHATS: Readonly<Record<string, unknown>> = {
  [AUTO_CHAT]: { id: AUTO_CHAT, userId: OWNER, title: 'Auto', kind: 'auto' },
  [USER_CHAT]: { id: USER_CHAT, userId: OWNER, title: 'Plans', kind: 'user' },
}

function client(): QueryClient {
  const quiet = { log: () => undefined, warn: () => undefined, error: () => undefined }
  return new QueryClient({ defaultOptions: { queries: { retry: false } }, logger: quiet })
}

function drawComposer(chatId = AUTO_CHAT, beside: ReactNode = null) {
  const queryClient = client()
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={queryClient}>
      {children}
      <MultimodalInput chatId={chatId} status="ready" stop={vi.fn()} sendMessage={vi.fn()} selectedVisibilityType="private" />
    </QueryClientProvider>
  )
  const view = render(tree(beside))
  return { ...view, queryClient, chatId, redraw: (next: ReactNode) => view.rerender(tree(next)) }
}

/** Answer the last Start or Hide as the server would have. */
function succeed(target: Mutation, answer: unknown = {}) {
  const calls = target.mutate.mock.calls
  const options = calls[calls.length - 1][1] as { onSuccess: (answer: unknown) => void }
  act(() => options.onSuccess(answer))
}

/** Wait until the chat's kind has been read (or refused) and drawn, so a test can say the line is not there. */
async function settled({ queryClient, chatId }: { queryClient: QueryClient; chatId: string }) {
  await waitFor(() => expect(queryClient.getQueryState(autoThreadQueryKey(chatId))?.status).toMatch(/^(success|error)$/))
  await act(async () => undefined)
}

beforeEach(() => {
  state.list = [suggestion(), second()]
  state.start = mutation()
  state.hide = mutation()
  state.composerProps = []
  state.createMission = { mutate: vi.fn(), isLoading: false }
  state.getChat.mockReset()
  state.getChat.mockImplementation(async (id: string) => {
    if (CHATS[id]) return CHATS[id]
    throw Object.assign(new Error('HTTP 404: Chat not found'), { status: 404 })
  })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the pinned suggestion, where it shows', () => {
  it('pins the newest open suggestion on one line directly above the composer in the Auto thread', async () => {
    drawComposer()
    const pinned = await line('Look at the Lisbon flights')
    expect(pinned).toHaveAttribute('data-variant', 'pinned')
    expect(pinned).toHaveAttribute('data-suggestion-id', FIRST)
    expect(pinned.compareDocumentPosition(screen.getByTestId('composer')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.queryByRole('group', { name: 'Auto suggests: Renew the domain' })).toBeNull()
    expect(state.getChat).toHaveBeenCalledWith(AUTO_CHAT)
  })

  it('hands the composer exactly the props it was given', async () => {
    drawComposer()
    await line('Look at the Lisbon flights')
    const props = state.composerProps[state.composerProps.length - 1] as Record<string, unknown>
    expect(Object.keys(props).sort()).toEqual(['chatId', 'selectedVisibilityType', 'sendMessage', 'status', 'stop'])
    expect(props.chatId).toBe(AUTO_CHAT)
  })

  it('draws nothing in any other chat, the composer as before', async () => {
    await settled(drawComposer(USER_CHAT))
    expect(state.getChat).toHaveBeenCalledWith(USER_CHAT)
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    expect(screen.getByTestId('composer')).toHaveAttribute('data-chat', USER_CHAT)
  })

  it('draws nothing for a chat the server does not have yet, and never asks for an id that is not a chat id', async () => {
    await settled(drawComposer(NEW_CHAT))
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    state.getChat.mockClear()
    drawComposer('../../admin')
    await act(async () => undefined)
    expect(state.getChat).not.toHaveBeenCalled()
  })

  it('draws nothing when no suggestion is open, or while the list is unanswered', async () => {
    state.list = []
    const empty = drawComposer()
    await settled(empty)
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    empty.unmount()
    state.list = undefined
    await settled(drawComposer())
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
  })

  it('tells the thread which suggestion it pins, until it hides', async () => {
    const view = drawComposer(AUTO_CHAT, <PinnedProbe />)
    await line('Look at the Lisbon flights')
    expect(screen.getByTestId('pinned-ids')).toHaveTextContent(FIRST)
    view.redraw(
      <>
        <PinnedProbe />
        <MissionSuggestionCard goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} />
      </>,
    )
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    expect(screen.getByTestId('pinned-ids')).toBeEmptyDOMElement()
  })

  it('tries the chat once more after a transient failure, never after a refusal', () => {
    const refused = (status: number) => Object.assign(new Error(`HTTP ${status}`), { status })
    expect(retriesAfter(0, new Error('Failed to fetch'))).toBe(true)
    expect(retriesAfter(0, refused(502))).toBe(true)
    expect(retriesAfter(1, refused(502))).toBe(false)
    expect(retriesAfter(0, refused(404))).toBe(false)
    expect(retriesAfter(0, refused(403))).toBe(false)
    expect(retriesAfter(0, new Error('HTTP 401: Unauthorized (missing/invalid Clerk token).'))).toBe(false)
  })

  it("pins only a suggestion addressed to the thread's owner, the newest of those", async () => {
    state.list = [second({ user_id: 7 }), suggestion()]
    drawComposer()
    expect(await line('Look at the Lisbon flights')).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Auto suggests: Renew the domain' })).toBeNull()
    expect(newestFor([second({ user_id: 7 }), suggestion(), second({ id: 'x' })], '1')?.id).toBe(FIRST)
    expect(newestFor([second({ user_id: null })], '1')).toBeNull()
  })
})

describe('the pinned suggestion, one suggestion surface', () => {
  it("hides while a turn's Mission suggestion is unanswered, and comes back once it is answered", async () => {
    const mission = <MissionSuggestionCard goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} />
    const view = drawComposer(AUTO_CHAT, mission)
    await settled(view)
    expect(screen.getByTestId('mission-card')).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'No thanks' }))
    expect(await line('Look at the Lisbon flights')).toBeInTheDocument()
    view.redraw(mission)
    expect(screen.getByRole('group', { name: /Auto suggests/ })).toBeInTheDocument()
  })

  it('comes back once the Mission suggestion is unmounted', async () => {
    const view = drawComposer(AUTO_CHAT, <MissionSuggestionCard goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} />)
    await settled(view)
    expect(screen.queryByRole('group', { name: /Auto suggests/ })).toBeNull()
    view.redraw(null)
    expect(await line('Look at the Lisbon flights')).toBeInTheDocument()
    expect(missionSuggestionOpen()).toBe(false)
  })

  it("keeps each registration its own: one card's release never frees another's, and a release twice is harmless", () => {
    const one = markMissionSuggestionOpen()
    const two = markMissionSuggestionOpen()
    one()
    one()
    expect(missionSuggestionOpen()).toBe(true)
    two()
    expect(missionSuggestionOpen()).toBe(false)
  })
})

describe('the pinned suggestion, Start and Hide from the line', () => {
  it('Start posts with via web, says "Started. Auto is on it." in place, then lets the next open one take the line', async () => {
    const { redraw } = drawComposer()
    await line('Look at the Lisbon flights')
    vi.useFakeTimers()
    fireEvent.click(screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' }))
    expect(state.start.mutate).toHaveBeenCalledWith({ id: FIRST, via: 'web' }, expect.anything())
    succeed(state.start, { suggestion: suggestion({ status: 'started' }), opens: { type: 'board_task', board_task_id: 9 } })
    state.list = [second()]
    redraw(null)
    expect(screen.getByText(STARTED_LINE)).toBeInTheDocument()
    act(() => vi.advanceTimersByTime(STARTED_MS))
    expect(screen.queryByText(STARTED_LINE)).toBeNull()
    expect(screen.getByRole('group', { name: 'Auto suggests: Renew the domain' })).toBeInTheDocument()
  })

  it('Hide sends the reason picked from the menu, and the line goes at once', async () => {
    drawComposer()
    await line('Look at the Lisbon flights')
    fireEvent.click(screen.getByRole('button', { name: 'Hide: Look at the Lisbon flights' }))
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
    expect(state.hide.mutate).toHaveBeenCalledWith({ id: FIRST, via: 'web', reason: 'not_now' }, expect.anything())
    succeed(state.hide, { suggestion: suggestion({ status: 'hidden' }) })
    expect(screen.queryByRole('group', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeNull()
  })

  it("the line's title opens the full card in its place; Escape folds it back, after closing Hide's menu first", async () => {
    drawComposer()
    await line('Look at the Lisbon flights')
    fireEvent.click(screen.getByRole('button', { name: 'Open: Look at the Lisbon flights' }))
    const card = screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })
    expect(within(card).getByText('Three runs failed since Monday.')).toBeInTheDocument()
    expect(within(card).getByText('Start does: files a ticket on the Board for an agent to work.')).toBeInTheDocument()
    expect(screen.getByTestId('pinned-suggestion')).toHaveFocus()
    fireEvent.click(within(card).getByRole('button', { name: 'Hide: Look at the Lisbon flights' }))
    fireEvent.keyDown(screen.getByRole('group', { name: 'Why hide it?' }), { key: 'Escape' })
    expect(screen.queryByRole('group', { name: 'Why hide it?' })).toBeNull()
    expect(screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
    fireEvent.keyDown(screen.getByTestId('pinned-suggestion'), { key: 'Escape' })
    expect(screen.queryByRole('article', { name: /Auto suggests/ })).toBeNull()
    expect(screen.getByRole('group', { name: 'Auto suggests: Look at the Lisbon flights' })).toHaveAttribute('data-variant', 'pinned')
  })

  it('folds the card back to the line once focus leaves it, and keeps it open while focus moves inside it', async () => {
    drawComposer()
    await line('Look at the Lisbon flights')
    fireEvent.click(screen.getByRole('button', { name: 'Open: Look at the Lisbon flights' }))
    const start = screen.getByRole('button', { name: 'Start: Look at the Lisbon flights' })
    fireEvent.focusOut(screen.getByTestId('pinned-suggestion'), { relatedTarget: start })
    expect(screen.getByRole('article', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
    fireEvent.focusOut(screen.getByTestId('pinned-suggestion'), { relatedTarget: document.body })
    expect(screen.queryByRole('article', { name: /Auto suggests/ })).toBeNull()
    expect(screen.getByRole('group', { name: 'Auto suggests: Look at the Lisbon flights' })).toBeInTheDocument()
  })
})

describe("the chat's Mission suggestion card says when it is answered", () => {
  type Card = typeof import('@/components/chatbot/mission-suggestion-card')
  const real = () => vi.importActual<Card>('@/components/chatbot/mission-suggestion-card')

  it('on its X and on "No thanks"', async () => {
    const { MissionSuggestionCard: Real } = await real()
    const answered = vi.fn()
    const view = render(<Real goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} onAnswered={answered} />)
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss suggestion' }))
    expect(answered).toHaveBeenCalledTimes(1)
    view.unmount()

    render(<Real goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} onAnswered={answered} />)
    fireEvent.click(screen.getByRole('button', { name: 'No thanks' }))
    expect(answered).toHaveBeenCalledTimes(2)
  })

  it('once its Mission is launched', async () => {
    const { MissionSuggestionCard: Real } = await real()
    const answered = vi.fn()
    render(<Real goal="Plan the offsite" complexity="organ" chatId={AUTO_CHAT} onAnswered={answered} />)
    fireEvent.click(screen.getByRole('button', { name: /Launch Mission/ }))
    expect(answered).not.toHaveBeenCalled()
    const options = state.createMission.mutate.mock.calls[0][1] as { onSuccess: (mission: { id: string }) => void }
    act(() => options.onSuccess({ id: 'm-1' }))
    expect(answered).toHaveBeenCalledTimes(1)
  })
})

describe('the seams and the source rules', () => {
  const read = (...parts: string[]) => readFileSync(path.resolve(__dirname, '..', '..', '..', ...parts), 'utf8')

  it("chat.tsx draws the composer and the Mission suggestion through the wrappers, by their own names", () => {
    const chat = read('components', 'chatbot', 'chat.tsx')
    expect(chat).toContain("import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'")
    expect(chat).toContain("import { MissionSuggestionCard } from '@/components/proactive/mission-suggestion-with-presence'")
    expect(chat).not.toContain("from './multimodal-input'")
    expect(chat).not.toContain("from '@/components/chatbot/mission-suggestion-card'")
  })

  it('keeps the new files to tokens and short components', () => {
    const files = [
      'pinned-suggestion.tsx',
      'composer-with-suggestion.tsx',
      'mission-suggestion-with-presence.tsx',
      'mission-suggestion-presence.ts',
      'presence.ts',
      'use-auto-thread.ts',
    ]
    for (const file of files) {
      const source = proactiveSource(file)
      expect(source).not.toMatch(HEX_COLOUR)
      expect(source).not.toMatch(ORANGE_CLASS)
      expect(source).not.toMatch(ROUNDED_3XL)
      componentSpans(source).forEach((span) => expect(span.lines, span.name).toBeLessThanOrEqual(MAX_COMPONENT_LINES))
    }
  })
})
