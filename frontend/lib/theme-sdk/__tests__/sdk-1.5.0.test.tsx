/**
 * SDK 1.5.0 — themes talk to individual agents and use Plan. What 1.5.0 adds reaches a theme through
 * '@automatos/theme-sdk', over the host's REAL chat controller and stream (lib/chat-controller, useChatWithReceipts and
 * useChat posting to /api/chat, the wire stubbed as lib/chat-controller/__tests__/support/chat-wire.ts stubs it):
 *
 * - A theme's chat page, written against the SDK alone, picks an agent of the roster with `chooseAgent`: the next
 *   message goes with `agentId`, as Studio's chat sends it, and the host's AI label names that agent while it answers.
 *   null is Auto again: no `agentId` goes.
 * - A session agent (`configuration.runtime` 'cli') is refused (false) as Studio's selector refuses it, and so is an
 *   id the roster lacks: the message still goes to Auto. A value that is neither a whole number nor null throws.
 * - `setPlanMode(true)` is Studio's own Plan switch (stores/mission-store.ts): the next message goes with `planMode`,
 *   and Studio's store has it; anything but true or false throws.
 * - `useAgents()`' roster agents carry `configuration`, so a theme can tell a session agent apart and link to it
 *   (`/chat?agent=<id>`, which HostLink and useHostRouter take).
 */
import { act, cleanup, render, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  getChat: vi.fn(),
  getChatMessages: vi.fn(),
  getChatHistory: vi.fn(),
  putChatSession: vi.fn(),
  cancelChatTurn: vi.fn(),
}))
const roster = vi.hoisted(() => [
  { id: 3, name: 'Words', status: 'active', configuration: { runtime: 'api' } },
  { id: 9, name: 'Claude Code', status: 'active', configuration: { runtime: 'cli' } },
])

vi.mock('next/navigation', () => ({
  usePathname: () => '/chat',
  useSearchParams: () => new URLSearchParams(''),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), prefetch: vi.fn() }),
}))
vi.mock('next/link', () => ({ default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a> }))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn(), info: vi.fn() } }))
vi.mock('@/lib/chat/api', () => api)
vi.mock('@/lib/page-context', () => ({ usePageContext: () => ({ page: 'chat', route: '/chat' }) }))
// The local edition's identity shapes (lib/auth-hooks.ts), with no token.
vi.mock('@/lib/auth-hooks', () => ({
  useAuth: () => ({ isLoaded: true, isSignedIn: true, getToken: async () => null }),
  useUser: () => ({ isLoaded: true, isSignedIn: true, user: { id: 'u1' } }),
  useSession: () => ({ isLoaded: true, isSignedIn: true, session: null }),
  useOrganization: () => ({ isLoaded: true, organization: null, membership: null }),
  useClerk: () => ({ signOut: async () => undefined }),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ workspace: null }), useWorkspaceOptional: () => null }))
// The host's roster, as react-query hands it to Studio's selector and to the SDK.
vi.mock('@/hooks/use-agent-api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/hooks/use-agent-api')>()),
  useAgents: () => ({ data: roster, error: null, isInitialLoading: false, isFetching: false, refetch: vi.fn() }),
}))

import * as sdk from '@automatos/theme-sdk'
import { PAGE_AI_LABEL, useAiLabelStore } from '@/lib/ai-label/store'
import { useMissionStore } from '@/stores/mission-store'

import { chatEndpoint, endStream, posts, turns } from '../../chat-controller/__tests__/support/chat-wire'
import * as chatArea from '../chat'
import { NEW_CHAT, setSession } from './support/hello-chat-page'

/** What the theme's page last drew: its conversation's controller. */
const seen: { chat: sdk.HostChatController | null } = { chat: null }

/** A theme's chat, from the SDK alone: the conversation on screen, then its chat in a child keyed by it. */
function ThemeConversation({ conversation }: { readonly conversation: sdk.HostChatConversation }) {
  seen.chat = sdk.useChatController(conversation)
  return null
}

function ThemeChatPage() {
  const { conversation } = sdk.useChatController()
  return conversation ? <ThemeConversation key={conversation.key} conversation={conversation} /> : null
}

function renderThemeChat() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ThemeChatPage />
    </QueryClientProvider>,
  )
}

async function opened(): Promise<sdk.HostChatController> {
  await waitFor(() => expect(seen.chat).not.toBeNull())
  return seen.chat as sdk.HostChatController
}

/** The theme sends; the host posts it to /api/chat. */
async function send(text: string) {
  const before = posts.length
  act(() => {
    expect(seen.chat?.send(text)).toBe(true)
  })
  await waitFor(() => expect(posts).toHaveLength(before + 1))
  return posts[posts.length - 1]
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset())
  api.getChatHistory.mockResolvedValue([])
  api.cancelChatTurn.mockResolvedValue(undefined)
  setSession(NEW_CHAT)
  useAiLabelStore.setState({ hiddenLabels: [], shownLabels: [PAGE_AI_LABEL] })
  vi.stubGlobal('fetch', chatEndpoint)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  chatEndpoint.mockClear()
  turns.length = 0
  posts.length = 0
  seen.chat = null
  useAiLabelStore.setState({ hiddenLabels: [], shownLabels: [], agentId: null })
  useMissionStore.setState({ isPlanMode: false, isMissionMode: false })
  window.localStorage.clear()
})

describe('SDK 1.5.0 — the chat area', () => {
  it('is the chat area’s own controller, at SDK 1.5.0', () => {
    expect(sdk.useChatController).toBe(chatArea.useChatController)
    expect(sdk.SDK_VERSION).toBe('1.5.0')
  })
})

describe('SDK 1.5.0 — chooseAgent: who answers', () => {
  it("sends the theme's pick as agentId and the AI label names it; null is Auto again", async () => {
    renderThemeChat()
    const chat = await opened()
    expect(chat.agentId).toBeNull()
    act(() => {
      expect(chat.chooseAgent(3)).toBe(true)
    })
    await waitFor(() => expect(seen.chat?.agentId).toBe(3))
    await waitFor(() => expect(useAiLabelStore.getState().agentId).toBe(3))
    expect(await send('Draft the launch note')).toMatchObject({ agentId: 3 })
  })

  it('goes to Auto, with no agentId, until the theme picks; after null it goes to Auto again', async () => {
    renderThemeChat()
    const chat = await opened()
    act(() => {
      chat.chooseAgent(3)
    })
    await waitFor(() => expect(seen.chat?.agentId).toBe(3))
    act(() => {
      expect(seen.chat?.chooseAgent(null)).toBe(true)
    })
    await waitFor(() => expect(seen.chat?.agentId).toBeNull())
    await waitFor(() => expect(useAiLabelStore.getState().agentId).toBeNull())
    expect(await send('Hello')).not.toHaveProperty('agentId')
  })

  it("refuses a session agent (runtime 'cli') and an id the roster lacks: the message goes to Auto", async () => {
    renderThemeChat()
    const chat = await opened()
    act(() => {
      expect(chat.chooseAgent(9)).toBe(false)
      expect(chat.chooseAgent(404)).toBe(false)
    })
    expect(seen.chat?.agentId).toBeNull()
    expect(await send('Open my session')).not.toHaveProperty('agentId')
  })

  it('throws for anything but a whole number or null, naming what it takes', async () => {
    renderThemeChat()
    const chat = await opened()
    for (const bad of ['3', 3.5, 0, -1, undefined]) {
      expect(() => chat.chooseAgent(bad as unknown as number)).toThrow(/chooseAgent takes an agent id/)
    }
  })
})

describe('SDK 1.5.0 — planMode: Studio’s own Plan switch', () => {
  it('sends planMode once the theme turns Plan on, and Studio’s store has it', async () => {
    renderThemeChat()
    const chat = await opened()
    expect(chat.planMode).toBe(false)
    expect(await send('First')).not.toHaveProperty('planMode')
    await endStream()
    act(() => chat.setPlanMode(true))
    await waitFor(() => expect(seen.chat?.planMode).toBe(true))
    expect(useMissionStore.getState().isPlanMode).toBe(true)
    expect(await send('Now plan it')).toMatchObject({ planMode: true })
  })

  it('sends planMode when Studio’s Plan is already on', async () => {
    useMissionStore.getState().setPlanMode(true)
    renderThemeChat()
    expect((await opened()).planMode).toBe(true)
    expect(await send('Plan my week')).toMatchObject({ planMode: true })
  })

  it('throws for anything but true or false', async () => {
    renderThemeChat()
    const chat = await opened()
    for (const bad of ['on', 1, null, undefined]) {
      expect(() => chat.setPlanMode(bad as unknown as boolean)).toThrow(/setPlanMode takes true or false/)
    }
    expect(useMissionStore.getState().isPlanMode).toBe(false)
  })
})

describe('SDK 1.5.0 — the roster tells a session agent apart, and the link to one', () => {
  it("hands each agent's configuration on, frozen", () => {
    const { result } = renderHook(() => sdk.useAgents())
    const agents = result.current.data ?? []
    expect(agents.map((agent) => [agent.id, agent.configuration?.runtime])).toEqual([[3, 'api'], [9, 'cli']])
    expect(Object.isFrozen(agents[1].configuration)).toBe(true)
  })

  it('links to a session agent through HostLink and useHostRouter: /chat?agent=<id> is a path inside the app', () => {
    const { result } = renderHook(() => sdk.useHostRouter())
    expect(() => result.current.push('/chat?agent=9')).not.toThrow()
    const link = render(<sdk.HostLink href="/chat?agent=9">Open Claude Code</sdk.HostLink>)
    expect(link.getByText('Open Claude Code').getAttribute('href')).toBe('/chat?agent=9')
  })
})
