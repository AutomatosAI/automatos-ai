/**
 * PRD-260 T-004 — Play's sidebar: the nav built from the Studio menu with the
 * filters StudioSidebar applies (studio-sidebar-edition.test.tsx expects the same
 * of the Studio rail), Play's Conversations and Look added; the wordmark; New
 * chat; Recent from getChatHistory; the connection pill from GET /health, "Not
 * running" only once every retry has failed (P260-RVW-1).
 * Roles, attributes and text only (CSS modules are proxies under css: false).
 */
import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  BACKEND_HEALTH_POLL_MS,
  BACKEND_HEALTH_RETRIES,
  BACKEND_HEALTH_RETRY_DELAY_MS,
} from '@/hooks/use-backend-health'

// The edition is read once at import, so each case imports the sidebar afresh.
vi.setConfig({ testTimeout: 30_000 })

const nav = vi.hoisted(() => ({ pathname: '/command-center', push: vi.fn() }))
vi.mock('next/navigation', () => ({
  usePathname: () => nav.pathname,
  useRouter: () => ({ push: nav.push, replace: vi.fn() }),
}))

const workspaceRef = vi.hoisted(() => ({ current: null as unknown }))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspaceOptional: () => ({ workspace: workspaceRef.current }),
}))

const roleRef = vi.hoisted(() => ({ current: { isAdmin: true } as { isAdmin: boolean } | null }))
vi.mock('@/contexts/role-context', () => ({ useSystemRoleOptional: () => roleRef.current }))

const store = vi.hoisted(() => ({ newDraft: vi.fn(), session: { activeChatId: null as string | null } }))
vi.mock('@/stores/chat-session-store', () => ({
  useChatSessionStore: (select: (state: typeof store) => unknown) => select(store),
}))
// useStartChat binds the chat session store itself (P260-FX-03): hydrated here.
vi.mock('@/hooks/use-chat-session', () => ({ useChatSessionHydration: () => ({ key: 'k', hydrated: true }) }))

const chatApi = vi.hoisted(() => ({ getChatHistory: vi.fn() }))
vi.mock('@/lib/chat/api', () => chatApi)

const api = vi.hoisted(() => ({ request: vi.fn(), getBaseUrl: () => 'http://localhost:8000' }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

const LOCAL_BASIC_EXPOSURE = { plan: 'basic', families: {}, marketplace_depth: 1, nav: { analytics: false, team: false } }
const BUSINESS_EXPOSURE = { plan: 'business', families: {}, marketplace_depth: 3, nav: { analytics: true, team: true } }

/** Every entry Play's nav shows an admin on a business plan, in order. */
const SAAS_NAV = [
  '/chat', '/conversations', '/command-center', '/assignments', '/deliverables',
  '/agents', '/tools', '/documents', '/marketplace',
  '/team', '/analytics', '/admin/workspaces',
  '/looks',
]
const ELEVEN = [
  'Chat', 'Conversations', 'Command Centre', 'Assignments', 'Deliverables', 'Agent Management',
  'Tools & Integrations', 'Knowledge Base', 'Marketplace', 'Analytics', 'Look',
]

function chat(id: string, title: string) {
  return { id, userId: 'u1', title, createdAt: '2026-10-14T09:00:00Z', updatedAt: '2026-10-14T09:00:00Z', visibility: 'private' }
}

async function renderSidebar(edition: 'local' | 'saas' = 'saas', { fakeTimers = false } = {}) {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
  const { PlaySidebar } = await import('../play-sidebar')
  // Faked after the imports, before the health query sets its timers.
  if (fakeTimers) vi.useFakeTimers()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <PlaySidebar />
    </QueryClientProvider>,
  )
}

function navHrefs(): string[] {
  const main = screen.getByRole('navigation', { name: 'Main' })
  return within(main).getAllByRole('link').map((a) => a.getAttribute('href') ?? '')
}

function primeBackend({ up = true, chats = [] as ReturnType<typeof chat>[] } = {}) {
  api.request.mockImplementation(() => (up ? Promise.resolve({ status: 'healthy' }) : Promise.reject(new Error('Failed to fetch'))))
  chatApi.getChatHistory.mockResolvedValue(chats)
}

const healthCalls = () => api.request.mock.calls.filter(([endpoint]) => endpoint === '/health').length

/** Let `ms` of fake time pass, with the promises and React updates it starts. */
const elapse = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

/**
 * Let react-query hand a settled ask to React. Its notify is a setTimeout(0), and a timer set
 * during a fake tick fires 1 ms later, so the last retry's answer lands one fake ms after it.
 */
const settle = () => elapse(1)

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllEnvs()
  vi.resetModules()
  vi.clearAllMocks()
  workspaceRef.current = null
  roleRef.current = { isAdmin: true }
  nav.pathname = '/command-center'
  store.session.activeChatId = null
})

describe('Play nav — the Studio menu, Conversations and Look (saas)', () => {
  it('lists every core page for an admin on a business plan, Conversations after Chat, Look last', async () => {
    primeBackend()
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    await renderSidebar('saas')
    expect(navHrefs()).toEqual(SAAS_NAV)
    const main = screen.getByRole('navigation', { name: 'Main' })
    for (const label of ELEVEN) expect(within(main).getByRole('link', { name: label })).toBeInTheDocument()
    expect(within(main).getByRole('link', { name: 'Team Management' })).toBeInTheDocument()
    expect(within(main).getByRole('link', { name: 'Workspace Admin' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Settings' })).toHaveAttribute('href', '/settings')
    expect(screen.getByRole('link', { name: 'Docs' })).toHaveAttribute('href', 'https://docs.automatos.app')
  })

  it('groups the menu as Studio does, with Look standing alone', async () => {
    primeBackend()
    await renderSidebar('saas')
    expect(screen.getAllByRole('group')).toHaveLength(3)
    expect(screen.getByRole('group', { name: 'Operations' })).toContainElement(screen.getByRole('link', { name: 'Conversations' }))
    expect(screen.getByRole('group', { name: 'Workspace' })).not.toContainElement(screen.getByRole('link', { name: 'Look' }))
    expect(screen.getByText('Operations')).toBeInTheDocument()
    expect(screen.getByText('Workforce')).toBeInTheDocument()
    expect(screen.getByText('Workspace')).toBeInTheDocument()
  })

  it('a member sees neither Workspace Admin nor Settings', async () => {
    primeBackend()
    roleRef.current = { isAdmin: false }
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    await renderSidebar('saas')
    expect(navHrefs()).toEqual(SAAS_NAV.filter((h) => h !== '/admin/workspaces'))
    expect(screen.queryByRole('link', { name: 'Settings' })).toBeNull()
    expect(screen.getByRole('link', { name: 'Docs' })).toBeInTheDocument()
  })

  it('an unknown role (no provider) is not an admin', async () => {
    primeBackend()
    roleRef.current = null
    workspaceRef.current = { exposure: BUSINESS_EXPOSURE }
    await renderSidebar('saas')
    expect(screen.queryByText('Workspace Admin')).toBeNull()
    expect(screen.queryByRole('link', { name: 'Settings' })).toBeNull()
  })

  it('the basic plan hides Team and Analytics by exposure', async () => {
    primeBackend()
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    await renderSidebar('saas')
    expect(navHrefs()).toEqual(SAAS_NAV.filter((h) => h !== '/team' && h !== '/analytics'))
  })

  it('no workspace provider fails open, as the Studio rail does', async () => {
    primeBackend()
    await renderSidebar('saas')
    expect(navHrefs()).toEqual(SAAS_NAV)
  })
})

describe('Play nav — local edition', () => {
  it('drops Team and Workspace Admin, keeps Analytics under the plan-less basic exposure', async () => {
    primeBackend()
    workspaceRef.current = { exposure: LOCAL_BASIC_EXPOSURE }
    await renderSidebar('local')
    expect(navHrefs()).toEqual(SAAS_NAV.filter((h) => h !== '/team' && h !== '/admin/workspaces'))
    expect(screen.getByRole('link', { name: 'Settings' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Docs' })).toBeInTheDocument()
  })
})

describe('Play sidebar — the page you are on', () => {
  it.each([
    ['/command-center', 'Command Centre'],
    ['/chat', 'Chat'],
    ['/conversations', 'Conversations'],
    ['/looks/style', 'Look'],
    ['/deliverables/explorer', 'Deliverables'],
    ['/settings', 'Settings'],
  ])('%s marks %s as the current page, and only it', async (pathname, label) => {
    primeBackend()
    nav.pathname = pathname
    await renderSidebar('saas')
    const current = screen.getAllByRole('link').filter((a) => a.getAttribute('aria-current') === 'page')
    expect(current.map((a) => a.textContent)).toEqual([label])
  })
})

describe('Play sidebar — wordmark, New chat, Recent', () => {
  it('the wordmark reads Automatos Play and opens the chat', async () => {
    primeBackend()
    await renderSidebar()
    expect(screen.getByRole('link', { name: 'Automatos Play' })).toHaveAttribute('href', '/chat')
  })

  it('New chat starts a draft in the chat session store, then opens the chat', async () => {
    primeBackend()
    await renderSidebar()
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(store.newDraft).toHaveBeenCalledTimes(1)
    expect(nav.push).toHaveBeenCalledWith('/chat')
  })

  it('lists at most 8 recent conversations from getChatHistory, each opening in the chat', async () => {
    const chats = Array.from({ length: 10 }, (_, i) => chat(`c${i}`, `Chat number ${i}`))
    primeBackend({ chats: [...chats.slice(0, 9), chat('c9', '')] })
    await renderSidebar()
    const recent = await screen.findByRole('region', { name: 'Recent' })
    const links = within(recent).getAllByRole('link')
    expect(chatApi.getChatHistory).toHaveBeenCalledWith(8)
    expect(links).toHaveLength(8)
    expect(links[0]).toHaveTextContent('Chat number 0')
    expect(links[0]).toHaveAttribute('href', '/chat?chatId=c0')
  })

  it('names an untitled conversation and marks the open one on the chat', async () => {
    nav.pathname = '/chat'
    store.session.activeChatId = 'c2'
    primeBackend({ chats: [chat('c1', ''), chat('c2', 'Laptop for uni')] })
    await renderSidebar()
    const recent = await screen.findByRole('region', { name: 'Recent' })
    expect(within(recent).getByRole('link', { name: 'Conversation' })).toHaveAttribute('href', '/chat?chatId=c1')
    expect(within(recent).getByRole('link', { name: 'Laptop for uni' })).toHaveAttribute('aria-current', 'true')
  })

  it('shows no Recent section while there are no conversations', async () => {
    primeBackend({ chats: [] })
    await renderSidebar()
    await waitFor(() => expect(chatApi.getChatHistory).toHaveBeenCalled())
    expect(screen.queryByRole('region', { name: 'Recent' })).toBeNull()
  })
})

describe('Play sidebar — the connection pill (GET /health)', () => {
  it('reads Connected with the API host when /health answers', async () => {
    primeBackend({ up: true })
    await renderSidebar()
    const pill = screen.getByRole('status')
    expect(pill).toHaveTextContent('Checking…')
    await waitFor(() => expect(pill).toHaveTextContent('Connected · localhost'))
    expect(api.request).toHaveBeenCalledWith('/health')
  })

  it('reads Not running only once /health has failed on every retry', async () => {
    primeBackend({ up: false })
    await renderSidebar('saas', { fakeTimers: true })
    await elapse(0)
    const pill = screen.getByRole('status')
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      // A failed ask is not the answer yet: /health is asked again first.
      expect(pill).toHaveTextContent('Checking…')
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    await settle()
    expect(healthCalls()).toBe(1 + BACKEND_HEALTH_RETRIES)
    expect(pill).toHaveTextContent('Not running')
  })

  it('stays Connected while a failed poll is asked again, and reads Not running once every retry fails', async () => {
    primeBackend({ up: true })
    await renderSidebar('saas', { fakeTimers: true })
    await elapse(0)
    const pill = screen.getByRole('status')
    expect(pill).toHaveTextContent('Connected · localhost')
    primeBackend({ up: false })
    await elapse(BACKEND_HEALTH_POLL_MS)
    for (let retry = 1; retry <= BACKEND_HEALTH_RETRIES; retry += 1) {
      expect(pill).toHaveTextContent('Connected · localhost')
      await elapse(BACKEND_HEALTH_RETRY_DELAY_MS)
    }
    await settle()
    expect(healthCalls()).toBe(2 + BACKEND_HEALTH_RETRIES)
    expect(pill).toHaveTextContent('Not running')
  })
})
