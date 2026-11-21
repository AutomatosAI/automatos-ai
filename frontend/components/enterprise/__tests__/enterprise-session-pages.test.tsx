/**
 * PRD-277 US-501, US-502 — the enterprise session provider on public pages, when the session
 * ends, and the role it hands the app.
 *
 *   - A public page (lib/public-routes.ts, the middleware's own list) is never sent to sign in: a
 *     401 from who-am-I there leaves it rendered; a page that needs a person still goes to sign in.
 *   - Any 401 from the app's own API opens "Your session ended", with the reason's words when the
 *     API's `session_ended` gave one.
 *   - The dialog's button keeps the chat box's text, for the person signed in, and goes to the
 *     API's start address, bringing them back to this page.
 *   - RoleProvider in enterprise reads the install role from `/me`.
 *   - The provider puts the session fetch over `window.fetch` and takes it off when it unmounts.
 *   - Sign-out removes the kept draft, and goes only to a plain `https:` end-session address.
 */
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), setClerkTokenGetter: vi.fn() }))
const nav = vi.hoisted(() => ({ pathname: '/chat' }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@clerk/nextjs', () => ({
  useAuth: vi.fn(), useUser: vi.fn(), useSession: vi.fn(), useOrganization: vi.fn(), useClerk: vi.fn(),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: { ...api, getBaseUrl: () => '' } }))
vi.mock('next/navigation', () => ({ usePathname: () => nav.pathname }))

import { EnterpriseSessionProvider } from '@/components/enterprise/enterprise-session-provider'
import { SessionEndedDialog } from '@/components/enterprise/session-ended-dialog'
import { RoleProvider, useSystemRole } from '@/contexts/role-context'
import { reportUnauthorized } from '@/lib/api-unauthorized'
import { SIGN_IN_START_PATH, safeEndSessionAddress, type WhoAmI } from '@/lib/enterprise/auth-api'
import { DRAFT_STORAGE_KEY, keepChatDraft, setDraftOwner, useKeptChatDraft } from '@/lib/enterprise/kept-chat-draft'
import { SESSION_ENDED_BODY, SESSION_ENDED_REASONS, SESSION_ENDED_TITLE, SIGN_IN_AGAIN } from '@/lib/enterprise/messages'
import { signOutOfEnterprise } from '@/lib/enterprise/session-context'
import { pageGateOf, signInPageFor, useSessionState } from '@/lib/enterprise/use-session-state'

const WS_A = '00000000-0000-0000-0000-00000000000a'

function who(role: WhoAmI['person']['system_role'] = 'user'): WhoAmI {
  return {
    edition: 'enterprise',
    person: { id: 7, email: 'ada@example.test', name: 'Ada Example', system_role: role },
    workspaces: [{ id: WS_A, name: 'Research', role: 'editor' }],
    in_no_workspace: false,
    default_workspace_id: null,
  }
}

const unauthorized = () => Object.assign(new Error('HTTP 401: Unauthorized'), { body: { detail: 'Not authenticated' } })

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()
  setDraftOwner(null)
  nav.pathname = '/chat'
  vi.clearAllMocks()
})

describe('public pages stay public', () => {
  it("reads the middleware's own list", () => {
    expect(['/marketplace', '/verify/abc', '/go/pair', '/sign-in/x', '/agents', '/verify-email'].map(pageGateOf)).toEqual([
      'public', 'public', 'public', 'sign-in', 'protected', 'protected',
    ])
  })

  it('a 401 from who-am-I on a public page sends nobody to sign in; a protected page still does', async () => {
    window.history.replaceState(null, '', '/agents')
    api.get.mockRejectedValue(unauthorized())
    const navigate = vi.fn()
    const { result, rerender } = renderHook(({ path }) => useSessionState(path, navigate), { initialProps: { path: '/marketplace' } })
    await waitFor(() => expect(result.current.load).toBe('signed-out'))
    expect(navigate).not.toHaveBeenCalled()

    rerender({ path: '/agents' })
    await waitFor(() => expect(navigate).toHaveBeenCalledWith(signInPageFor('/agents')))
  })

  it('renders a public page signed out, with no notice over it', async () => {
    nav.pathname = '/marketplace'
    api.get.mockRejectedValue(unauthorized())
    render(<EnterpriseSessionProvider><p>the shop window</p></EnterpriseSessionProvider>)
    expect(screen.getByText('the shop window')).toBeTruthy()
    await waitFor(() => expect(api.get).toHaveBeenCalledWith('/api/v1/auth/me'))
    expect(screen.getByText('the shop window')).toBeTruthy()
  })
})

describe('"Your session ended"', () => {
  it('opens on any 401 from the API, with the general words when the API says nothing more', async () => {
    api.get.mockResolvedValue(who())
    render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(await screen.findByText('the app')).toBeTruthy()

    act(() => reportUnauthorized(undefined))
    expect(await screen.findByText(SESSION_ENDED_TITLE)).toBeTruthy()
    expect(screen.getByText(SESSION_ENDED_BODY)).toBeTruthy()
  })

  it("says why when the API's session_ended gave a reason", async () => {
    api.get.mockResolvedValue(who())
    render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(await screen.findByText('the app')).toBeTruthy()

    act(() => reportUnauthorized({ detail: { code: 'session_ended', reason: 'expired', message: 'Your session has ended.' } }))
    expect(await screen.findByText(`${SESSION_ENDED_REASONS.expired} ${SESSION_ENDED_BODY}`)).toBeTruthy()
  })

  it("its button keeps the chat box's text for the person and goes to the API's start address", () => {
    window.history.replaceState(null, '', '/chat/abc?tab=files')
    setDraftOwner('7')
    renderHook(() => useKeptChatDraft('half a question', vi.fn()))
    const navigate = vi.fn()
    render(<SessionEndedDialog navigate={navigate} />)

    fireEvent.click(screen.getByText(SIGN_IN_AGAIN))

    const kept = JSON.parse(window.sessionStorage.getItem(DRAFT_STORAGE_KEY) ?? '{}')
    expect([kept.text, kept.owner]).toEqual(['half a question', '7'])
    const target = new URL(navigate.mock.calls[0][0], 'https://automatos.example.test')
    expect(target.pathname).toBe(SIGN_IN_START_PATH)
    expect(target.searchParams.get('return_to')).toBe('/chat/abc?tab=files')
  })
})

describe('the session and the app', () => {
  it("RoleProvider reads the install role from /me's system_role", async () => {
    api.get.mockResolvedValue(who('admin'))
    const seen: Array<{ role: string; admin: boolean; loading: boolean }> = []
    function Probe() {
      const { systemRole, isAdmin, isLoading } = useSystemRole()
      seen.push({ role: systemRole, admin: isAdmin, loading: isLoading })
      return null
    }
    render(<EnterpriseSessionProvider><RoleProvider><Probe /></RoleProvider></EnterpriseSessionProvider>)
    await waitFor(() => expect(seen[seen.length - 1]).toEqual({ role: 'admin', admin: true, loading: false }))
    expect(api.get).toHaveBeenCalledWith('/api/v1/auth/me')
  })

  it('names the signed-in person as the owner of any kept draft, removing one another person typed', async () => {
    keepChatDraft('not yours', window.sessionStorage, Date.now(), '8')
    api.get.mockResolvedValue(who())
    render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(await screen.findByText('the app')).toBeTruthy()
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })

  it('puts the session fetch over window.fetch, and takes it off when it unmounts', async () => {
    const before = window.fetch
    const original = vi.fn(async () => new Response('{}')) as unknown as typeof fetch
    window.fetch = original
    api.get.mockResolvedValue(who())
    const view = render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(window.fetch).not.toBe(original)
    await screen.findByText('the app')

    view.unmount()
    expect(window.fetch).toBe(original)
    window.fetch = before
  })
})

describe('signing out', () => {
  it('removes the kept draft', async () => {
    keepChatDraft('half a question', window.sessionStorage, Date.now(), '7')
    api.post.mockResolvedValue(null)
    const navigate = vi.fn()
    void signOutOfEnterprise(navigate)
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/sign-in'))
    expect(window.sessionStorage.getItem(DRAFT_STORAGE_KEY)).toBeNull()
  })

  it.each([
    'javascript:alert(1)',
    'http://idp.example.test/logout',
    'https:idp.example.test/logout',
    'https://' + 'someone' + ':' + 'x' + '@idp.example.test/logout',
    'https://@idp.example.test/logout',
    'https://idp.example.test/log out',
    'https://idp.example.test/\nlogout',
    ' https://idp.example.test/logout',
    '//idp.example.test/logout',
    'https://',
    'https://idp.example.test\\@elsewhere.example.test/',
  ])('goes to the sign-in page, not to %j', async (address) => {
    expect(safeEndSessionAddress(address)).toBeNull()
    api.post.mockResolvedValueOnce({ redirect_to: address })
    const navigate = vi.fn()
    void signOutOfEnterprise(navigate)
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('/sign-in'))
  })

  it("goes to a plain https end-session address as the URL reads it", () => {
    expect(safeEndSessionAddress('https://idp.example.test/logout?state=x')).toBe('https://idp.example.test/logout?state=x')
    expect(safeEndSessionAddress('HTTPS://IdP.Example.Test/logout')).toBe('https://idp.example.test/logout')
    expect(safeEndSessionAddress(42)).toBeNull()
  })
})
