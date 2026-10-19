/**
 * PRD-260 P260-FX-10 (M45, M44) — Play's Conversations list
 * (useConversationList) is the account's own and fresh: keyed by the chat scope
 * (the workspace and the signed-in user, useChatScope), asked again each time it
 * opens, and after a turn ends (useRefreshHistoryOnTurnEnd). A real QueryClient
 * with the root's staleTime (components/providers.tsx); the identity, the
 * workspace and getChatHistory are stand-ins.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat } from '@/types'

const identity = vi.hoisted(() => ({ user: 'u-a', loaded: true }))
/** The workspace provider's value (useWorkspaceOptional); null is no provider. */
const scope = vi.hoisted(() => ({ workspace: null as null | { workspace: { id: string } | null; isLoading: boolean } }))
vi.mock('@/lib/auth-hooks', () => ({
  useUser: () => ({ isLoaded: identity.loaded, isSignedIn: identity.loaded, user: identity.loaded ? { id: identity.user } : null }),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => scope.workspace }))

const chatApi = vi.hoisted(() => ({ getChatHistory: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

import { CONVERSATION_LIMIT, type ConversationListState, useConversationList } from '@/hooks/use-conversation-list'
import { useRefreshHistoryOnTurnEnd } from '@/hooks/use-refresh-history-on-turn-end'
import type { ChatStatus } from '@/lib/mascot/events'

/** The root providers' fresh window (components/providers.tsx): within it, nothing is asked again on its own. */
const ROOT_STALE_MS = 60_000

function chatRow(id: string, title: string): Chat {
  return { id, title, userId: identity.user, createdAt: '', updatedAt: '', visibility: 'private' }
}

const ACCOUNT_A_CHAT = chatRow('c-a', 'Laptop for uni')
const NEW_CHAT = chatRow('c-new', 'Bake sale')

function rootClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { staleTime: ROOT_STALE_MS, retry: false } } })
}

function wrapperOf(client: QueryClient) {
  return function Root({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
}

/** The titles of a loaded list, or its status. */
function shown(state: ConversationListState): string[] | string {
  return state.status === 'loaded' ? state.chats.map((c) => c.title) : state.status
}

/** getChatHistory's next answer never comes: the list is still being asked for. */
function holdNextAnswer() {
  chatApi.getChatHistory.mockReturnValueOnce(new Promise<Chat[]>(() => undefined))
}

beforeEach(() => {
  chatApi.getChatHistory.mockResolvedValue([ACCOUNT_A_CHAT])
})

afterEach(() => {
  vi.clearAllMocks()
  chatApi.getChatHistory.mockReset()
  Object.assign(identity, { user: 'u-a', loaded: true })
  scope.workspace = null
})

describe('useConversationList — one list per account (M45)', () => {
  it("never shows the last account's conversations to the next one on this browser", async () => {
    const hook = renderHook(() => useConversationList(), { wrapper: wrapperOf(rootClient()) })
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Laptop for uni']))
    identity.user = 'u-b'
    holdNextAnswer()
    hook.rerender()
    expect(hook.result.current).toEqual({ status: 'loading' })
    await waitFor(() => expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2))
    expect(hook.result.current).toEqual({ status: 'loading' })
  })

  it('a different workspace has its own list too', async () => {
    scope.workspace = { workspace: { id: 'ws-1' }, isLoading: false }
    const hook = renderHook(() => useConversationList(), { wrapper: wrapperOf(rootClient()) })
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Laptop for uni']))
    scope.workspace = { workspace: { id: 'ws-2' }, isLoading: false }
    holdNextAnswer()
    hook.rerender()
    expect(hook.result.current).toEqual({ status: 'loading' })
    await waitFor(() => expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2))
  })

  it('asks for nothing until the user and the workspace have settled', async () => {
    identity.loaded = false
    scope.workspace = { workspace: null, isLoading: true }
    const hook = renderHook(() => useConversationList(), { wrapper: wrapperOf(rootClient()) })
    expect(hook.result.current).toEqual({ status: 'loading' })
    identity.loaded = true
    hook.rerender()
    expect(chatApi.getChatHistory).not.toHaveBeenCalled()
    scope.workspace = { workspace: { id: 'ws-1' }, isLoading: false }
    hook.rerender()
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Laptop for uni']))
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
    expect(chatApi.getChatHistory).toHaveBeenCalledWith(CONVERSATION_LIMIT)
  })
})

describe('useConversationList — fresh each time it opens (M44)', () => {
  it('shows a conversation started since, when the page opens again within the fresh window', async () => {
    const client = rootClient()
    const first = renderHook(() => useConversationList(), { wrapper: wrapperOf(client) })
    await waitFor(() => expect(shown(first.result.current)).toEqual(['Laptop for uni']))
    first.unmount()
    chatApi.getChatHistory.mockResolvedValue([NEW_CHAT, ACCOUNT_A_CHAT])
    const again = renderHook(() => useConversationList(), { wrapper: wrapperOf(client) })
    // The list it had stays on screen while it is asked again.
    expect(shown(again.result.current)).toEqual(['Laptop for uni'])
    await waitFor(() => expect(shown(again.result.current)).toEqual(['Bake sale', 'Laptop for uni']))
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2)
  })
})

describe('useRefreshHistoryOnTurnEnd (M44)', () => {
  /** The list, beside a chat whose status the test moves. */
  function renderBesideChat() {
    return renderHook(
      ({ status }: { status: ChatStatus }) => {
        useRefreshHistoryOnTurnEnd(status)
        return useConversationList()
      },
      { initialProps: { status: 'idle' as ChatStatus }, wrapper: wrapperOf(rootClient()) },
    )
  }

  it('asks the history again when the status leaves streaming, however the turn ended', async () => {
    const hook = renderBesideChat()
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Laptop for uni']))
    chatApi.getChatHistory.mockResolvedValue([NEW_CHAT, ACCOUNT_A_CHAT])
    hook.rerender({ status: 'streaming' })
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
    hook.rerender({ status: 'idle' })
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Bake sale', 'Laptop for uni']))
    hook.rerender({ status: 'streaming' })
    hook.rerender({ status: 'error' })
    await waitFor(() => expect(chatApi.getChatHistory).toHaveBeenCalledTimes(3))
  })

  it('asks nothing while no turn ends', async () => {
    const hook = renderBesideChat()
    await waitFor(() => expect(shown(hook.result.current)).toEqual(['Laptop for uni']))
    hook.rerender({ status: 'idle' })
    hook.rerender({ status: 'error' })
    hook.rerender({ status: 'idle' })
    await act(async () => undefined)
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
  })
})
