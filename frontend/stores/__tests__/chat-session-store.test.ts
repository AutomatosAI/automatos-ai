/**
 * PRD-237: the conversation session store — persistence per edition.
 *
 * hosted (default env): browser copy + a debounced server PUT; the server copy
 * is merged on hydrate when it is newer. local: browser only, never the API.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

const api = vi.hoisted(() => ({
  getChatSession: vi.fn(),
  putChatSession: vi.fn(),
}))
vi.mock('@/lib/chat/api', () => api)

import { EMPTY_CHAT_SESSION, chatSessionKey, loadChatSession, saveChatSession, withChatOpened } from '@/lib/chat/chat-session'
import { SERVER_SAVE_DEBOUNCE_MS, useChatSessionStore } from '@/stores/chat-session-store'

const KEY = chatSessionKey('ws-1', 'u-1')
const NOW = 1_800_000_000_000

const flush = () => new Promise((r) => setTimeout(r, 0))

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  api.getChatSession.mockResolvedValue({ activeChatId: null, draftOpen: false, openChatIds: [], lastReadAt: {}, updatedAt: null })
  api.putChatSession.mockResolvedValue(undefined)
  useChatSessionStore.setState({ key: null, hydrated: false, session: EMPTY_CHAT_SESSION })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('hydrate (hosted)', () => {
  it('loads the browser copy, then merges a newer server copy', async () => {
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'local', NOW))
    api.getChatSession.mockResolvedValue({
      activeChatId: 'srv',
      draftOpen: false,
      openChatIds: ['srv'],
      lastReadAt: {},
      updatedAt: new Date(NOW + 60_000).toISOString(),
    })
    await useChatSessionStore.getState().hydrate(KEY)
    const { session, hydrated, key } = useChatSessionStore.getState()
    expect(key).toBe(KEY)
    expect(hydrated).toBe(true)
    expect(session.activeChatId).toBe('srv')
    expect(loadChatSession(KEY).activeChatId).toBe('srv') // the merge is saved back
  })

  it('keeps the browser copy when the server copy is older or empty, and survives an API failure', async () => {
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'local', NOW))
    await useChatSessionStore.getState().hydrate(KEY)
    expect(useChatSessionStore.getState().session.activeChatId).toBe('local')

    useChatSessionStore.setState({ key: null, hydrated: false, session: EMPTY_CHAT_SESSION })
    api.getChatSession.mockRejectedValue(new Error('offline'))
    await useChatSessionStore.getState().hydrate(KEY)
    expect(useChatSessionStore.getState().hydrated).toBe(true)
    expect(useChatSessionStore.getState().session.activeChatId).toBe('local')
  })

  it('a workspace switch mid-flight is not applied to the new key', async () => {
    let resolve: (v: unknown) => void = () => {}
    api.getChatSession.mockReturnValue(new Promise((r) => (resolve = r)))
    const first = useChatSessionStore.getState().hydrate(KEY)
    useChatSessionStore.setState({ key: chatSessionKey('ws-2', 'u-1') })
    resolve({ activeChatId: 'srv', draftOpen: false, openChatIds: ['srv'], lastReadAt: {}, updatedAt: new Date(NOW).toISOString() })
    await first
    expect(useChatSessionStore.getState().session.activeChatId).toBeNull()
  })
})

describe('mutations (hosted)', () => {
  it('saves every change to the browser and coalesces server writes', async () => {
    vi.useFakeTimers()
    await useChatSessionStore.getState().hydrate(KEY)
    const store = useChatSessionStore.getState()
    store.openChat('a')
    store.openChat('b')
    store.newDraft()
    expect(loadChatSession(KEY).openChatIds).toEqual(['a', 'b'])
    expect(loadChatSession(KEY).draftOpen).toBe(true)
    expect(api.putChatSession).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(SERVER_SAVE_DEBOUNCE_MS + 5)
    expect(api.putChatSession).toHaveBeenCalledTimes(1)
    expect(api.putChatSession).toHaveBeenCalledWith({
      activeChatId: null,
      draftOpen: true,
      openChatIds: ['a', 'b'],
      lastReadAt: expect.any(Object),
    })
  })

  it('titles and unread flags stay on the device — no server write, no-ops save nothing', async () => {
    vi.useFakeTimers()
    await useChatSessionStore.getState().hydrate(KEY)
    const store = useChatSessionStore.getState()
    store.openChat('a')
    store.openChat('b')
    await vi.advanceTimersByTimeAsync(SERVER_SAVE_DEBOUNCE_MS + 5)
    expect(api.putChatSession).toHaveBeenCalledTimes(1)

    const setItem = vi.spyOn(Storage.prototype, 'setItem')
    store.setTitles({ a: 'Alpha' })
    store.touchChat('a')
    expect(setItem).toHaveBeenCalledTimes(2)
    store.chatIdAssigned('b') // already the active open tab — nothing to do
    expect(setItem).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(SERVER_SAVE_DEBOUNCE_MS + 5)
    expect(api.putChatSession).toHaveBeenCalledTimes(1)
    expect(useChatSessionStore.getState().session.titles.a).toBe('Alpha')
    expect(useChatSessionStore.getState().session.unreadChatIds).toEqual(['a'])
  })

  it('reload follows a change made by another browser tab', async () => {
    await useChatSessionStore.getState().hydrate(KEY)
    saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'elsewhere', NOW))
    useChatSessionStore.getState().reload()
    expect(useChatSessionStore.getState().session.activeChatId).toBe('elsewhere')
  })
})

describe('local edition', () => {
  it('is hydrated from the browser alone and never calls the API', async () => {
    vi.resetModules()
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
    try {
      const localApi = { getChatSession: vi.fn(), putChatSession: vi.fn() }
      vi.doMock('@/lib/chat/api', () => localApi)
      const { useChatSessionStore: localStore } = await import('@/stores/chat-session-store')
      saveChatSession(KEY, withChatOpened(EMPTY_CHAT_SESSION, 'mine', NOW))
      await localStore.getState().hydrate(KEY)
      expect(localStore.getState().hydrated).toBe(true)
      expect(localStore.getState().session.activeChatId).toBe('mine')
      localStore.getState().openChat('other')
      await flush()
      await new Promise((r) => setTimeout(r, SERVER_SAVE_DEBOUNCE_MS + 20))
      expect(localApi.getChatSession).not.toHaveBeenCalled()
      expect(localApi.putChatSession).not.toHaveBeenCalled()
      expect(loadChatSession(KEY).activeChatId).toBe('other')
    } finally {
      vi.unstubAllEnvs()
      vi.doUnmock('@/lib/chat/api')
      vi.resetModules()
    }
  })
})
