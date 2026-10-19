/**
 * PRD-260 P260-FX-10 (M45, M44) — Play's Recent list (useRecentChats) is the
 * account's own and fresh: keyed by the chat scope (the workspace and the
 * signed-in user, useChatScope) and the limit, never by the chat on screen, and
 * asked again whenever the chat on screen changes. A real QueryClient with the
 * root's staleTime (components/providers.tsx) and the real session store; the
 * identity, the workspace and getChatHistory are stand-ins.
 */
import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat } from '@/types'

const identity = vi.hoisted(() => ({ user: 'u-a' }))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: identity.user } }) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => null }))

const chatApi = vi.hoisted(() => ({ getChatHistory: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

import { RECENT_CHATS_LIMIT, useRecentChats } from '@/hooks/use-recent-chats'
import { EMPTY_CHAT_SESSION } from '@/lib/chat/chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

/** The root providers' fresh window (components/providers.tsx): within it, nothing is asked again on its own. */
const ROOT_STALE_MS = 60_000
/** Fake time between one chat on screen and the next: three steps stay inside 30 s, well inside the fresh window. */
const STEP_MS = 10_000

function chatRow(id: string, title: string): Chat {
  return { id, title, userId: identity.user, createdAt: '', updatedAt: '', visibility: 'private' }
}

const FIRST = chatRow('c-a', 'Laptop for uni')
const STARTED_SINCE = chatRow('c-b', 'Bake sale')

function renderRecent() {
  const client = new QueryClient({ defaultOptions: { queries: { staleTime: ROOT_STALE_MS, retry: false } } })
  const Root = ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return renderHook(() => useRecentChats(), { wrapper: Root })
}

/** The chat on screen, as the session store has it. */
function showChat(activeChatId: string | null) {
  act(() => useChatSessionStore.setState({ session: { ...EMPTY_CHAT_SESSION, activeChatId } }))
}

const idsOf = (chats: readonly Chat[]) => chats.map((c) => c.id)

/** Let `ms` of fake time pass, with the answers and React updates it brings (react-query notifies on a setTimeout(0)). */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

beforeEach(() => {
  showChat(null)
  chatApi.getChatHistory.mockResolvedValue([FIRST])
})

afterEach(() => {
  vi.useRealTimers()
  chatApi.getChatHistory.mockReset()
  identity.user = 'u-a'
})

describe('useRecentChats — one list per account (M45)', () => {
  it("never shows the last account's Recent to the next one on this browser", async () => {
    const hook = renderRecent()
    await waitFor(() => expect(idsOf(hook.result.current)).toEqual(['c-a']))
    identity.user = 'u-b'
    chatApi.getChatHistory.mockReturnValueOnce(new Promise<Chat[]>(() => undefined))
    hook.rerender()
    expect(hook.result.current).toEqual([])
    await waitFor(() => expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2))
    expect(hook.result.current).toEqual([])
  })
})

describe('useRecentChats — fresh as the chat on screen changes (M44)', () => {
  it('keeps a conversation started since when the first one is shown again, inside 30 s', async () => {
    vi.useFakeTimers()
    showChat('c-a')
    const hook = renderRecent()
    await elapse(STEP_MS)
    expect(idsOf(hook.result.current)).toEqual(['c-a'])
    chatApi.getChatHistory.mockResolvedValue([STARTED_SINCE, FIRST])
    showChat('c-b')
    await elapse(STEP_MS)
    expect(idsOf(hook.result.current)).toEqual(['c-b', 'c-a'])
    showChat('c-a')
    await elapse(STEP_MS)
    expect(idsOf(hook.result.current)).toEqual(['c-b', 'c-a'])
  })

  it('asks once as it first shows, then once per change of the chat on screen, at its limit', async () => {
    vi.useFakeTimers()
    showChat('c-a')
    const hook = renderRecent()
    await elapse(STEP_MS)
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
    hook.rerender()
    await elapse(STEP_MS)
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(1)
    showChat(null)
    await elapse(STEP_MS)
    expect(chatApi.getChatHistory).toHaveBeenCalledTimes(2)
    expect(chatApi.getChatHistory).toHaveBeenLastCalledWith(RECENT_CHATS_LIMIT)
  })
})
