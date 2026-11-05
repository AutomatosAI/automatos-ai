/**
 * SDK 1.5.0 — `/chat?agent=<id>` (components/chatbot/agent-link.tsx): the chat's composer
 * (components/proactive/composer-with-suggestion.tsx, the one chat.tsx draws) picks the link's agent through the
 * chat's own pick, `onAgentChange` (chat.tsx handleAgentChange: a session agent opens its Claude Code session, any
 * other answers the chat), once per link for the visit to /chat (AgentLinkProvider, in app/chat/layout.tsx), and only
 * once the conversation on screen has settled, the workspace is known and the roster has the agent.
 *
 * The composer's box, its pinned suggestion and its consent line are stand-ins; the session store is real. That
 * `agent` keeps the core chat page under a theme that owns /chat is components/theme-host/__tests__/theme-slot.test.tsx.
 */
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ search: 'agent=9' }))
const host = vi.hoisted(() => ({
  roster: [
    { id: 3, name: 'Words', configuration: { runtime: 'api' } },
    { id: 9, name: 'Claude Code', configuration: { runtime: 'cli' } },
  ] as unknown[] | undefined,
  workspace: { id: 'ws-1' } as { id: string } | null,
}))

vi.mock('next/navigation', () => ({ useSearchParams: () => new URLSearchParams(nav.search) }))
vi.mock('@/hooks/use-agent-api', () => ({ useAgents: () => ({ data: host.roster }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => (host.workspace ? { workspace: host.workspace } : null),
}))
vi.mock('@/components/chatbot/multimodal-input', () => ({
  MultimodalInput: ({ chatId }: { chatId: string }) => <div data-testid="the-composer" data-chat-id={chatId} />,
}))
vi.mock('@/components/proactive/pinned-suggestion', () => ({ PinnedSuggestion: () => null }))
vi.mock('@/components/proactive/calibration-offer', () => ({ CalibrationOffer: () => null, useCalibrationOffered: () => false }))

import { AgentLinkProvider, agentIdFromLink } from '@/components/chatbot/agent-link'
import { MultimodalInput } from '@/components/proactive/composer-with-suggestion'
import { EMPTY_CHAT_SESSION, type ChatSession } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

type Pointer = Pick<ChatSession, 'activeChatId' | 'draftOpen' | 'openChatIds'>
const ON_C1: Pointer = { activeChatId: 'c1', draftOpen: false, openChatIds: ['c1'] }
const COLD_START: Pointer = { activeChatId: null, draftOpen: false, openChatIds: [] }
const NEW_CHAT: Pointer = { activeChatId: null, draftOpen: true, openChatIds: [] }

function setSession(pointer: Pointer, hydrated = true) {
  act(() => {
    useChatSessionStore.setState({ key: 'k', hydrated, session: { ...EMPTY_CHAT_SESSION, ...pointer } })
  })
}

/** The chat's composer as chat.tsx draws it, in the chat layout's provider (or none: `bare`). */
function chatTree(onAgentChange: (id: number | null) => void, chatId = 'c1', { composer = true, bare = false } = {}) {
  const box = composer ? (
    <MultimodalInput chatId={chatId} status="ready" stop={() => {}} sendMessage={() => {}} selectedVisibilityType="private" onAgentChange={onAgentChange} />
  ) : null
  return bare ? box : <AgentLinkProvider>{box}</AgentLinkProvider>
}

beforeEach(() => {
  nav.search = 'agent=9'
  host.roster = [
    { id: 3, name: 'Words', configuration: { runtime: 'api' } },
    { id: 9, name: 'Claude Code', configuration: { runtime: 'cli' } },
  ]
  host.workspace = { id: 'ws-1' }
  setSession(ON_C1)
})

afterEach(cleanup)

describe('agentIdFromLink', () => {
  it('reads a positive whole number, and nothing else', () => {
    expect(agentIdFromLink(new URLSearchParams('agent=9'))).toBe(9)
    expect(agentIdFromLink(new URLSearchParams('agent=%209%20'))).toBe(9)
    for (const search of ['', 'agent=', 'agent=0', 'agent=-3', 'agent=3.5', 'agent=9x', 'agent=abc', 'agent=12345678901', 'repo=x']) {
      expect(agentIdFromLink(new URLSearchParams(search))).toBeNull()
    }
    expect(agentIdFromLink(null)).toBeNull()
  })
})

describe('/chat?agent=<id> picks through the chat’s own pick, once', () => {
  it("hands a session agent to the chat's pick (which opens its Claude Code session), and a chat agent alike", () => {
    const pick = vi.fn()
    render(chatTree(pick))
    expect(pick.mock.calls).toEqual([[9]])
    cleanup()
    nav.search = 'agent=3&ticket=612'
    const chatPick = vi.fn()
    render(chatTree(chatPick))
    expect(chatPick.mock.calls).toEqual([[3]])
  })

  it('picks once for the visit: the composer mounting again, or a new pick function, never picks it again', () => {
    const pick = vi.fn()
    const view = render(chatTree(pick))
    expect(pick).toHaveBeenCalledTimes(1)
    view.rerender(chatTree(pick, 'c1', { composer: false }))
    view.rerender(chatTree(pick))
    const another = vi.fn()
    view.rerender(chatTree(another))
    expect(pick).toHaveBeenCalledTimes(1)
    expect(another).not.toHaveBeenCalled()
  })

  it('picks again after the link left the URL and came back, and for another agent’s link', () => {
    const pick = vi.fn()
    const view = render(chatTree(pick))
    nav.search = ''
    view.rerender(chatTree(pick))
    nav.search = 'agent=9'
    view.rerender(chatTree(pick))
    nav.search = 'agent=3'
    view.rerender(chatTree(pick))
    expect(pick.mock.calls).toEqual([[9], [9], [3]])
  })
})

describe('/chat?agent=<id> waits until it can pick', () => {
  it('for the roster to have the agent: an id it lacks is never picked', () => {
    const pick = vi.fn()
    host.roster = undefined
    const view = render(chatTree(pick))
    expect(pick).not.toHaveBeenCalled()
    host.roster = [{ id: 3, name: 'Words', configuration: null }]
    view.rerender(chatTree(pick))
    expect(pick).not.toHaveBeenCalled()
    host.roster = [{ id: 9, name: 'Claude Code', configuration: { runtime: 'cli' } }]
    view.rerender(chatTree(pick))
    expect(pick.mock.calls).toEqual([[9]])
  })

  it('for the workspace (the session canvas needs it)', () => {
    const pick = vi.fn()
    host.workspace = null
    const view = render(chatTree(pick))
    expect(pick).not.toHaveBeenCalled()
    host.workspace = { id: 'ws-1' }
    view.rerender(chatTree(pick))
    expect(pick).toHaveBeenCalledTimes(1)
  })

  it("for the conversation on screen to settle: a cold start's blank chat waits, the chat it lands on is picked for", () => {
    const pick = vi.fn()
    setSession(COLD_START)
    const view = render(chatTree(pick, ''))
    expect(pick).not.toHaveBeenCalled()
    setSession(ON_C1)
    expect(pick).not.toHaveBeenCalled()
    view.rerender(chatTree(pick, 'c1'))
    expect(pick.mock.calls).toEqual([[9]])
  })

  it('for the session to hydrate, then picks for a new chat too', () => {
    const pick = vi.fn()
    setSession(NEW_CHAT, false)
    render(chatTree(pick, ''))
    expect(pick).not.toHaveBeenCalled()
    setSession(NEW_CHAT)
    expect(pick.mock.calls).toEqual([[9]])
  })
})

describe('without a link the composer is as before', () => {
  it.each(['', 'agent=abc', 'agent=0', 'repo=projects%2Fsite'])('picks nothing for ?%s', (search) => {
    nav.search = search
    const pick = vi.fn()
    render(chatTree(pick))
    expect(pick).not.toHaveBeenCalled()
  })

  it('picks nothing outside the chat layout’s provider', () => {
    const pick = vi.fn()
    render(chatTree(pick, 'c1', { bare: true }))
    expect(pick).not.toHaveBeenCalled()
  })
})
