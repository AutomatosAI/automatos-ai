/**
 * PRD-261 A-19e2 — the core behaviour a theme's Conversations page stands on, on the fixture theme
 * (lib/theme-host/__tests__/fixture-theme-entry.ts, A-19b). Its page is the fixture's own route
 * (`fixture-route`, registerRoute), written here against '@automatos/theme-sdk' only and drawn by
 * ThemeSlotRoute as /x/<route> draws it. The SDK's useConversationList is the real one over the
 * host's (hooks/use-conversation-list.ts, lib/chat/api.ts getChatHistory), keyed by the real chat
 * scope; the backend client, the signed-in user and the workspace are stood in. The query client
 * has the root's fresh window (components/providers.tsx), so nothing here is asked again by age.
 *
 * Paging: the list is one page of the history, its size on the wire (`limit`: the host's
 * CONVERSATION_LIMIT, or the theme's own), newest first as the backend sends it. Rename: a
 * conversation the backend renames (a new chat gets its title after its first turn) shows its new
 * title, as the same row, when the turn ends (useRefreshHistoryOnTurnEnd, which the host's chat
 * controller runs) and when the page opens again.
 *
 * Replaces, in the core test ledger (plan §2.2.8), the core assertions of
 * components/play/conversations/__tests__/play-conversations.test.tsx (the history asked once
 * through getChatHistory at its page size, a row's /chat?chatId=<id> link, a failed load with no
 * internals and Try again loading it). Its Play-rendering assertions move to the kit (B-05d), which
 * lists these cases in `rehomed`. New chat's fresh draft is A-19e1's (chat-controller-cases).
 */
import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Chat } from '@/types'

const api = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/lib/auth-hooks', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/auth-hooks')>()),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'u-1' } }),
}))
vi.mock('@/components/workspace-provider', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/components/workspace-provider')>()),
  useWorkspaceOptional: () => ({ workspace: { id: 'ws-1' }, isLoading: false }),
}))
vi.mock('@/hooks/use-studio-theme', async () => {
  const { FIXTURE_THEME_ENTRY } = await import('@/lib/theme-host/__tests__/fixture-theme-entry')
  return { useIsStudio: () => true, useActiveTheme: () => FIXTURE_THEME_ENTRY, useThemeLayout: () => 'theme' }
})

import { CHAT_HREF, chatHref, HostLink, useConversationList, type HostConversationListState } from '@automatos/theme-sdk'
import { ThemeHostContext } from '@/components/theme-host/theme-host-context'
import { ThemeSlotRoute } from '@/components/theme-host/theme-slot-route'
import { CONVERSATION_LIMIT } from '@/hooks/use-conversation-list'
import { CHAT_HISTORY_QUERY_PREFIX, useRefreshHistoryOnTurnEnd } from '@/hooks/use-refresh-history-on-turn-end'
import type { ChatStatus } from '@/lib/chat/chat-status'
import { FIXTURE_THEME_ROUTE, FIXTURE_THEME_TEST_IDS, fixtureThemeHost } from '@/lib/theme-host/__tests__/fixture-theme-entry'

const HISTORY_PATH = '/api/chat/history?'
/** The root providers' fresh window (components/providers.tsx). */
const ROOT_STALE_MS = 60_000
/** A page size of the theme's own. */
const THEME_LIMIT = 20
const LIST_LABEL = 'Conversations'
const LOADING = 'Loading your conversations'
const LIST_FAILED = "The fixture theme's words: the conversations didn't load."
const TRY_AGAIN = 'Try again'
const INTERNALS = /502|10\.0\.0\.7|upstream/

/** Every state the fixture's page was handed, render by render. */
const seen = vi.hoisted(() => ({ states: [] as unknown[] }))

function chat(id: string, title: string, updatedAt: string, preview: string | null = null): Chat {
  return { id, userId: 'u-1', title, createdAt: updatedAt, updatedAt, visibility: 'private', lastMessagePreview: preview, kind: 'user' }
}

const LAPTOP = chat('c-laptop', 'Laptop for uni', '2026-10-14T15:28:00Z', 'Done! Three laptops make the cut.')
const CHEM = chat('c-chem', 'Revision plan for chemistry', '2026-10-13T18:00:00Z', 'Week 1 covers bonding and moles.')
const CAFE = chat('c-cafe', 'Cover letter for the café job', '2026-10-01T10:00:00Z')
const NEW_CHAT = chat('c-new', 'New chat', '2026-10-14T15:29:00Z')
const RENAMED = { ...NEW_CHAT, title: 'Bake sale captions', updatedAt: '2026-10-14T15:30:00Z' }

/** A fresh copy of a backend answer, as the network hands it over. */
const answer = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T

/** The backend's history: `chats` for every ask, until the test changes it. */
function serveHistory(chats: readonly Chat[]) {
  api.request.mockImplementation(async (path: string) => {
    if (!path.startsWith(HISTORY_PATH)) throw new Error(`unexpected call to ${path}`)
    return answer(chats)
  })
}

const historyAsks = () => api.request.mock.calls.map(([path]) => String(path)).filter((path) => path.startsWith(HISTORY_PATH))
const limitOf = (path: string | undefined) => new URLSearchParams(String(path).split('?')[1]).get('limit')
const lastState = () => seen.states[seen.states.length - 1] as HostConversationListState

/** The fixture theme's Conversations page: the list, its loading line, its own words for a failure. */
function FixtureConversations({ limit }: { readonly limit?: number }) {
  const list = useConversationList(limit)
  seen.states.push(list)
  if (list.status === 'loading') return <p role="status">{LOADING}</p>
  if (list.status === 'failed') {
    return (
      <div role="alert">
        {LIST_FAILED}
        <button type="button" onClick={list.retry}>{TRY_AGAIN}</button>
      </div>
    )
  }
  return (
    <ul aria-label={LIST_LABEL} data-testid={FIXTURE_THEME_TEST_IDS.route}>
      {list.chats.map((summary) => (
        <li key={summary.id}>
          <HostLink href={chatHref(summary.id)}>{summary.title}</HostLink>
        </li>
      ))}
    </ul>
  )
}

function FixtureConversationsAtThemeLimit() {
  return <FixtureConversations limit={THEME_LIMIT} />
}

/** The host's turn-end refresh, as the chat controller runs it beside the page. */
function TurnEnds({ status }: { readonly status: ChatStatus }) {
  useRefreshHistoryOnTurnEnd(status)
  return null
}

function rootClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { staleTime: ROOT_STALE_MS, retry: false } } })
}

/** The fixture's route under the theme host, beside a chat whose status the test moves. */
function routeTree(client: QueryClient, status: ChatStatus = 'idle', page = FixtureConversations) {
  const host = fixtureThemeHost('active', { routes: new Map([[FIXTURE_THEME_ROUTE, page]]) })
  return (
    <QueryClientProvider client={client}>
      <ThemeHostContext.Provider value={host}>
        <ThemeSlotRoute route={FIXTURE_THEME_ROUTE} />
        <TurnEnds status={status} />
      </ThemeHostContext.Provider>
    </QueryClientProvider>
  )
}

const titles = () => within(screen.getByRole('list', { name: LIST_LABEL })).getAllByRole('link').map((link) => link.textContent)

beforeEach(() => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  api.request.mockReset()
  seen.states.length = 0
})

describe('A-19e2 — paging: one page of the history, its size on the wire', () => {
  it("asks once for the host's page size and lists it newest first, as the backend sends it", async () => {
    serveHistory([LAPTOP, CHEM, CAFE])
    render(routeTree(rootClient()))
    expect(screen.getByRole('status')).toHaveTextContent(LOADING)
    await screen.findByRole('list', { name: LIST_LABEL })
    expect(titles()).toEqual([LAPTOP.title, CHEM.title, CAFE.title])
    expect(historyAsks()).toHaveLength(1)
    expect(limitOf(historyAsks()[0])).toBe(String(CONVERSATION_LIMIT))
  })

  it("puts the theme's own page size on the wire", async () => {
    serveHistory([LAPTOP])
    render(routeTree(rootClient(), 'idle', FixtureConversationsAtThemeLimit))
    await screen.findByRole('list', { name: LIST_LABEL })
    expect(historyAsks()).toHaveLength(1)
    expect(limitOf(historyAsks()[0])).toBe(String(THEME_LIMIT))
  })

  it("hands each conversation over as its id, title, last activity and newest line, never the owner's id", async () => {
    serveHistory([LAPTOP, CAFE])
    render(routeTree(rootClient()))
    await screen.findByRole('list', { name: LIST_LABEL })
    expect(lastState()).toEqual({
      status: 'loaded',
      chats: [
        { id: LAPTOP.id, title: LAPTOP.title, updatedAt: LAPTOP.updatedAt, preview: LAPTOP.lastMessagePreview },
        { id: CAFE.id, title: CAFE.title, updatedAt: CAFE.updatedAt, preview: null },
      ],
    })
    expect(JSON.stringify(lastState())).not.toContain('u-1')
  })

  it("links each row to its conversation in the chat, the id encoded (chatHref)", async () => {
    serveHistory([LAPTOP, chat('c 1/x', 'Odd id', LAPTOP.updatedAt)])
    render(routeTree(rootClient()))
    expect(await screen.findByRole('link', { name: LAPTOP.title })).toHaveAttribute('href', `${CHAT_HREF}?chatId=c-laptop`)
    expect(screen.getByRole('link', { name: 'Odd id' })).toHaveAttribute('href', `${CHAT_HREF}?chatId=c%201%2Fx`)
  })
})

describe('A-19e2 — a failed load', () => {
  it('hands the theme a failure with Try again and no internals; Try again loads the list', async () => {
    api.request.mockRejectedValueOnce(new Error('HTTP 502: upstream connect error at http://10.0.0.7:8000'))
    render(routeTree(rootClient()))
    const alert = await screen.findByRole('alert')
    expect(lastState().status).toBe('failed')
    expect(JSON.stringify(lastState())).not.toMatch(INTERNALS)
    expect(alert).not.toHaveTextContent(INTERNALS)
    expect(console.error).toHaveBeenCalled()

    serveHistory([LAPTOP])
    fireEvent.click(within(alert).getByRole('button', { name: TRY_AGAIN }))
    expect(await screen.findByRole('link', { name: LAPTOP.title })).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(historyAsks()).toHaveLength(2)
  })

  it('keeps a loaded list on screen when asking again fails', async () => {
    serveHistory([LAPTOP, CHEM])
    const client = rootClient()
    render(routeTree(client))
    await screen.findByRole('list', { name: LIST_LABEL })
    api.request.mockRejectedValueOnce(new Error('HTTP 502'))
    await act(async () => {
      await client.invalidateQueries({ queryKey: CHAT_HISTORY_QUERY_PREFIX })
    })
    await waitFor(() => expect(historyAsks()).toHaveLength(2))
    expect(titles()).toEqual([LAPTOP.title, CHEM.title])
    expect(screen.queryByRole('alert')).toBeNull()
  })
})

describe('A-19e2 — rename: the new title reaches the list as the same row', () => {
  it('when the turn that renamed it ends, inside the fresh window', async () => {
    serveHistory([NEW_CHAT, LAPTOP])
    const client = rootClient()
    const view = render(routeTree(client, 'idle'))
    await screen.findByRole('link', { name: NEW_CHAT.title })

    serveHistory([RENAMED, LAPTOP])
    view.rerender(routeTree(client, 'streaming'))
    expect(historyAsks()).toHaveLength(1)
    view.rerender(routeTree(client, 'idle'))
    expect(await screen.findByRole('link', { name: RENAMED.title })).toHaveAttribute('href', chatHref(NEW_CHAT.id))
    expect(titles()).toEqual([RENAMED.title, LAPTOP.title])
    expect(historyAsks()).toHaveLength(2)
  })

  it('when the page opens again inside the fresh window: the old title until the answer, then the new', async () => {
    serveHistory([NEW_CHAT])
    const client = rootClient()
    const first = render(routeTree(client))
    await screen.findByRole('link', { name: NEW_CHAT.title })
    first.unmount()

    serveHistory([RENAMED])
    render(routeTree(client))
    expect(screen.getByRole('link', { name: NEW_CHAT.title })).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: RENAMED.title })).toHaveAttribute('href', chatHref(NEW_CHAT.id))
    expect(titles()).toEqual([RENAMED.title])
    expect(historyAsks()).toHaveLength(2)
  })
})
