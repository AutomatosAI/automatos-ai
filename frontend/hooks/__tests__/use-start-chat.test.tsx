/**
 * PRD-260 P260-FX-03 (M04) — New chat on a page that isn't the chat.
 * A theme's New chat (the SDK's useStartChat, lib/theme-sdk/chat.ts, hands a
 * theme this hook's startChat and ready) runs on every page, where nothing else
 * binds the chat session store: useStartChat hydrates it and does nothing until
 * it is, so the draft is never overwritten by the stored session and, in SaaS,
 * an empty tab list never reaches the server. The real hook, store and hydration
 * hook, under a stand-in New chat button; the identity, the workspace and the
 * server copy are stand-ins.
 */
import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { StartChat } from '@/hooks/use-start-chat'
import { chatSessionKey, EMPTY_CHAT_SESSION, type ServerChatSession } from '@/lib/chat/chat-session'
import { SERVER_SAVE_DEBOUNCE_MS } from '@/stores/chat-session-store'

/** Long enough for the store's debounced server save to have gone out. */
const SAVE_WAIT_MS = SERVER_SAVE_DEBOUNCE_MS * 3

const ids = vi.hoisted(() => ({ workspace: 'ws-1', user: 'u-1' }))
const nav = vi.hoisted(() => ({ push: vi.fn(), replace: vi.fn() }))
vi.mock('next/navigation', () => ({ usePathname: () => '/deliverables', useRouter: () => nav }))
vi.mock('@/lib/auth-hooks', () => ({ useUser: () => ({ isLoaded: true, user: { id: ids.user } }) }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: { id: ids.workspace }, isLoading: false }),
}))

const chatApi = vi.hoisted(() => ({ getChatHistory: vi.fn(), getChatSession: vi.fn(), putChatSession: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

/** The server copy a fresh account has: never written. */
const UNWRITTEN: ServerChatSession = { activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null }

let answerServerCopy: (copy: ServerChatSession) => void = () => undefined

beforeEach(() => {
  const stored = { ...EMPTY_CHAT_SESSION, activeChatId: 'abc', openChatIds: ['abc'], lastActiveAt: Date.now() }
  window.localStorage.setItem(chatSessionKey(ids.workspace, ids.user), JSON.stringify(stored))
  chatApi.getChatHistory.mockResolvedValue([])
  chatApi.putChatSession.mockImplementation(async (session: ServerChatSession) => session)
  chatApi.getChatSession.mockReturnValue(
    new Promise<ServerChatSession>((resolve) => {
      answerServerCopy = resolve
    }),
  )
})

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.resetModules()
  vi.clearAllMocks()
  window.localStorage.clear()
})

/** A theme's New chat button, as a theme draws one: pressable once the hook is ready. */
function NewChatButton({ useStartChat }: { readonly useStartChat: () => StartChat }) {
  const { startChat, ready } = useStartChat()
  return (
    <button type="button" onClick={startChat} disabled={!ready}>
      New chat
    </button>
  )
}

/** New chat on /deliverables, in an edition read afresh (it is read once at import). */
async function renderNewChat(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { useStartChat } = await import('@/hooks/use-start-chat')
  const { useChatSessionStore } = await import('@/stores/chat-session-store')
  render(<NewChatButton useStartChat={useStartChat} />)
  return useChatSessionStore
}

const newChat = () => screen.getByRole('button', { name: 'New chat' })

async function hydrate() {
  await act(async () => answerServerCopy(UNWRITTEN))
  await waitFor(() => expect(newChat()).toBeEnabled())
}

const sentTabLists = () => chatApi.putChatSession.mock.calls.map(([session]) => (session as ServerChatSession).openChatIds)

describe('New chat away from the chat (saas)', () => {
  it('is disabled until the chat session store is hydrated', async () => {
    await renderNewChat('saas')
    expect(newChat()).toBeDisabled()
    await hydrate()
  })

  it('then opens a fresh draft beside the stored tabs, and the chat', async () => {
    const store = await renderNewChat('saas')
    await hydrate()
    fireEvent.click(newChat())
    expect(store.getState().session).toMatchObject({ activeChatId: null, draftOpen: true, openChatIds: ['abc'] })
    expect(nav.push).toHaveBeenCalledWith('/chat')
    // PRD-261 A-19a: the press is counted in the host's store, which the SDK's chat follows.
    const { useFreshDraftStore } = await import('@/stores/fresh-draft-store')
    expect(useFreshDraftStore.getState().freshDrafts).toBe(1)
    // SaaS keeps the tabs on the server. Waiting for that debounced save also keeps its timer
    // from firing into the next test (the local edition's, which asserts nothing is sent).
    await waitFor(() => expect(chatApi.putChatSession).toHaveBeenCalled(), { timeout: SAVE_WAIT_MS })
    expect(sentTabLists()).toContainEqual(['abc'])
  })

  it('never saves an empty tab list to the server, even when pressed before hydration', async () => {
    await renderNewChat('saas')
    fireEvent.click(newChat())
    await hydrate()
    fireEvent.click(newChat())
    await waitFor(() => expect(chatApi.putChatSession).toHaveBeenCalled(), { timeout: SAVE_WAIT_MS })
    expect(sentTabLists()).not.toContainEqual([])
    expect(sentTabLists()).toContainEqual(['abc'])
  })
})

describe('New chat away from the chat (local)', () => {
  it('opens a fresh draft beside the stored tabs, and nothing goes to the server', async () => {
    const store = await renderNewChat('local')
    await waitFor(() => expect(store.getState().hydrated).toBe(true))
    fireEvent.click(newChat())
    expect(store.getState().session).toMatchObject({ activeChatId: null, draftOpen: true, openChatIds: ['abc'] })
    await act(() => new Promise<void>((resolve) => setTimeout(resolve, SAVE_WAIT_MS)))
    expect(chatApi.putChatSession).not.toHaveBeenCalled()
    expect(chatApi.getChatSession).not.toHaveBeenCalled()
  })
})
