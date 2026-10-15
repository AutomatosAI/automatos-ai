/**
 * PRD-260 P260-FX-07 (M09, #71) — the session key while the workspace
 * refetches. Saving a key from the trial-exhausted banner refreshes the
 * workspace (refreshWorkspace sets isLoading again with the same workspace
 * still in hand): the key, and with it `hydrated`, must hold, or every surface
 * showing a conversation (Play's chat, Studio's, the widget) drops it for the
 * refetch and comes back from a stale snapshot. A different workspace still
 * rehydrates under its own key. The real store and the real hook; the
 * identity, the workspace and the server copy are stand-ins.
 */
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatSessionKey, EMPTY_CHAT_SESSION, type ServerChatSession } from '@/lib/chat/chat-session'

const ids = vi.hoisted(() => ({ user: 'u-1' }))
/** The workspace provider's value (useWorkspaceOptional); null is no provider. */
const scope = vi.hoisted(() => ({ workspace: null as null | { workspace: { id: string } | null; isLoading: boolean } }))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: ids.user } }) }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspaceOptional: () => scope.workspace }))

const chatApi = vi.hoisted(() => ({ getChatSession: vi.fn(), putChatSession: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

import { useChatScope, useChatSessionHydration } from '@/hooks/use-chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'

/** The server copy a fresh account has: never written. */
const UNWRITTEN: ServerChatSession = { activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null }
const FIRST = { id: 'ws-1' }
const SECOND = { id: 'ws-2' }

beforeEach(() => {
  useChatSessionStore.setState({ key: null, hydrated: false, session: EMPTY_CHAT_SESSION })
  chatApi.getChatSession.mockResolvedValue(UNWRITTEN)
  chatApi.putChatSession.mockImplementation(async (session: ServerChatSession) => session)
})

afterEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  scope.workspace = null
})

/** The hook, hydrated in the given workspace. */
async function hydratedIn(workspace: { id: string }) {
  scope.workspace = { workspace, isLoading: false }
  const hook = renderHook(() => useChatSessionHydration())
  await waitFor(() => expect(hook.result.current.hydrated).toBe(true))
  return hook
}

describe('useChatSessionHydration while the workspace loads', () => {
  it("waits for the workspace's first load, then binds its key", async () => {
    scope.workspace = { workspace: null, isLoading: true }
    const hook = renderHook(() => useChatSessionHydration())
    expect(hook.result.current).toEqual({ key: null, hydrated: false })
    scope.workspace = { workspace: FIRST, isLoading: false }
    hook.rerender()
    await waitFor(() => expect(hook.result.current).toEqual({ key: chatSessionKey(FIRST.id, ids.user), hydrated: true }))
  })

  it('a refresh of the same workspace keeps the key and stays hydrated throughout', async () => {
    const hook = await hydratedIn(FIRST)
    const key = chatSessionKey(FIRST.id, ids.user)
    scope.workspace = { workspace: FIRST, isLoading: true }
    hook.rerender()
    expect(hook.result.current).toEqual({ key, hydrated: true })
    scope.workspace = { workspace: FIRST, isLoading: false }
    hook.rerender()
    expect(hook.result.current).toEqual({ key, hydrated: true })
    expect(useChatSessionStore.getState().key).toBe(key)
  })

  it('a different workspace still rehydrates under its own key once it arrives', async () => {
    const hook = await hydratedIn(FIRST)
    scope.workspace = { workspace: FIRST, isLoading: true }
    hook.rerender()
    expect(hook.result.current.key).toBe(chatSessionKey(FIRST.id, ids.user))
    scope.workspace = { workspace: SECOND, isLoading: false }
    hook.rerender()
    const key = chatSessionKey(SECOND.id, ids.user)
    await waitFor(() => expect(hook.result.current).toEqual({ key, hydrated: true }))
    expect(useChatSessionStore.getState().key).toBe(key)
  })
})

// P260-FX-10 (M45): Play's history caches are keyed by the same scope the session binds under.
describe('useChatScope', () => {
  it("is null until the workspace's first load, then the session key", () => {
    scope.workspace = { workspace: null, isLoading: true }
    const hook = renderHook(() => useChatScope())
    expect(hook.result.current).toBeNull()
    scope.workspace = { workspace: FIRST, isLoading: false }
    hook.rerender()
    expect(hook.result.current).toBe(chatSessionKey(FIRST.id, ids.user))
  })

  it('names the account: another user, or another workspace, is another scope', () => {
    scope.workspace = { workspace: FIRST, isLoading: false }
    const hook = renderHook(() => useChatScope())
    const first = hook.result.current
    ids.user = 'u-2'
    try {
      hook.rerender()
      expect(hook.result.current).toBe(chatSessionKey(FIRST.id, 'u-2'))
      expect(hook.result.current).not.toBe(first)
    } finally {
      ids.user = 'u-1'
    }
    scope.workspace = { workspace: SECOND, isLoading: false }
    hook.rerender()
    expect(hook.result.current).toBe(chatSessionKey(SECOND.id, ids.user))
  })

  it('is the key the session hydrates under', async () => {
    const hook = await hydratedIn(FIRST)
    const scoped = renderHook(() => useChatScope())
    expect(scoped.result.current).toBe(hook.result.current.key)
  })
})
