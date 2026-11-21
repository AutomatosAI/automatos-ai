/**
 * PRD-277 US-501, US-502 — the enterprise session in the browser.
 *
 *   - Who the person is and their install role come from GET /api/v1/auth/me; the app renders
 *     once the API has answered and the workspace is stored.
 *   - The workspace is the stored one when the person holds it, else their first; an install
 *     admin in none opens the install's default; anyone else in none waits for an administrator.
 *   - A 401 from who-am-I sends the browser to /sign-in, bringing the person back.
 *   - A later 401 `session_ended` opens "Your session ended" over the page, which stays mounted.
 *   - The five auth hooks answer from the API (never Clerk): signed in, the person's name, no token.
 *   - Sign-out calls the API, then goes to the provider's end-session address when it has one.
 */
import { act, render, renderHook, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), setClerkTokenGetter: vi.fn() }))
const nav = vi.hoisted(() => ({ pathname: '/chat' }))

vi.mock('@/lib/api-client', () => ({ apiClient: { ...api, getBaseUrl: () => '' } }))
vi.mock('next/navigation', () => ({ usePathname: () => nav.pathname }))

import { EnterpriseSessionProvider } from '@/components/enterprise/enterprise-session-provider'
import { reportUnauthorized } from '@/lib/api-unauthorized'
import type { WhoAmI } from '@/lib/enterprise/auth-api'
import { useEnterpriseAuth, useEnterpriseUser } from '@/lib/enterprise/auth-hooks'
import { SESSION_ENDED_TITLE, WAITING_BODY } from '@/lib/enterprise/messages'
import { signOutOfEnterprise } from '@/lib/enterprise/session-context'
import { chooseWorkspace, signInPageFor, useSessionState } from '@/lib/enterprise/use-session-state'

const WS_A = '00000000-0000-0000-0000-00000000000a'
const WS_B = '00000000-0000-0000-0000-00000000000b'
const DEFAULT_WS = '00000000-0000-0000-0000-0000000000de'

function who(overrides: Partial<WhoAmI> = {}, role: WhoAmI['person']['system_role'] = 'user'): WhoAmI {
  return {
    edition: 'enterprise',
    person: { id: 7, email: 'ada@example.test', name: 'Ada Example', system_role: role },
    workspaces: [{ id: WS_A, name: 'Research', role: 'editor' }, { id: WS_B, name: 'Ops', role: 'viewer' }],
    in_no_workspace: false,
    default_workspace_id: DEFAULT_WS,
    ...overrides,
  }
}

const unauthorized = () => Object.assign(new Error('HTTP 401: Unauthorized'), { body: { detail: { code: 'session_ended' } } })

afterEach(() => {
  window.localStorage.clear()
  nav.pathname = '/chat'
  vi.clearAllMocks()
})

describe('choosing the workspace', () => {
  it('keeps the stored one the person holds, else takes their first', () => {
    expect(chooseWorkspace(who(), WS_B)).toBe(WS_B)
    expect(chooseWorkspace(who(), 'someone-elses')).toBe(WS_A)
    expect(chooseWorkspace(who(), null)).toBe(WS_A)
  })

  it('opens the default for an install admin in none, and nothing for anyone else', () => {
    const none = { workspaces: [], in_no_workspace: true }
    expect(chooseWorkspace(who(none, 'admin'), null)).toBe(DEFAULT_WS)
    expect(chooseWorkspace(who(none, 'user'), null)).toBeNull()
  })
})

describe('useSessionState', () => {
  it('stores the workspace and is signed in once the API answers', async () => {
    api.get.mockResolvedValue(who())
    const { result } = renderHook(() => useSessionState('/chat', vi.fn()))
    await waitFor(() => expect(result.current.load).toBe('signed-in'))
    expect(api.get).toHaveBeenCalledWith('/api/v1/auth/me')
    expect(result.current.workspaceId).toBe(WS_A)
    expect(window.localStorage.getItem('last_active_workspace')).toBe(WS_A)
  })

  it('sends the browser to sign in on a 401, bringing the person back', async () => {
    window.history.replaceState(null, '', '/agents?view=grid')
    api.get.mockRejectedValue(unauthorized())
    const navigate = vi.fn()
    const { result } = renderHook(() => useSessionState('/agents', navigate))
    await waitFor(() => expect(result.current.load).toBe('signed-out'))
    expect(navigate).toHaveBeenCalledWith(signInPageFor('/agents?view=grid'))
  })

  it('asks nothing on the sign-in page', () => {
    renderHook(() => useSessionState('/sign-in', vi.fn()))
    expect(api.get).not.toHaveBeenCalled()
  })
})

describe('EnterpriseSessionProvider', () => {
  it('renders the app once signed in, and "Your session ended" over it on session_ended', async () => {
    api.get.mockResolvedValue(who())
    render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(await screen.findByText('the app')).toBeTruthy()
    expect(api.setClerkTokenGetter).toHaveBeenCalledTimes(1)
    expect(await api.setClerkTokenGetter.mock.calls[0][0]()).toBeNull()

    act(() => reportUnauthorized({ detail: { code: 'session_ended', reason: 'revoked' } }))
    expect(await screen.findByText(SESSION_ENDED_TITLE)).toBeTruthy()
    expect(screen.getByText('the app')).toBeTruthy()
  })

  it('shows a person in no workspace that an administrator must add them', async () => {
    api.get.mockResolvedValue(who({ workspaces: [], in_no_workspace: true }))
    render(<EnterpriseSessionProvider><p>the app</p></EnterpriseSessionProvider>)
    expect(await screen.findByText(WAITING_BODY)).toBeTruthy()
    expect(screen.queryByText('the app')).toBeNull()
  })

  it('renders the sign-in page with no session at all', () => {
    nav.pathname = '/sign-in'
    render(<EnterpriseSessionProvider><p>sign-in page</p></EnterpriseSessionProvider>)
    expect(screen.getByText('sign-in page')).toBeTruthy()
    expect(api.get).not.toHaveBeenCalled()
  })

  it("answers the auth hooks from the API's who-am-I, with no token", async () => {
    api.get.mockResolvedValue(who({}, 'admin'))
    const seen: Array<{ signedIn: boolean; name: string | null }> = []
    function Probe() {
      const auth = useEnterpriseAuth()
      const { user } = useEnterpriseUser()
      seen.push({ signedIn: auth.isSignedIn, name: user?.fullName ?? null })
      return null
    }
    render(<EnterpriseSessionProvider><Probe /></EnterpriseSessionProvider>)
    await waitFor(() => expect(seen[seen.length - 1]).toEqual({ signedIn: true, name: 'Ada Example' }))
    const { result } = renderHook(() => useEnterpriseAuth())
    expect(await result.current.getToken()).toBeNull()
  })
})

describe('signing out', () => {
  it("calls the API, forgets the workspace and goes to the provider's end-session address", async () => {
    window.localStorage.setItem('last_active_workspace', WS_A)
    api.post.mockResolvedValue({ redirect_to: 'https://idp.example.test/logout?state=x' })
    const navigate = vi.fn()
    void signOutOfEnterprise(navigate)
    await waitFor(() => expect(navigate).toHaveBeenCalledWith('https://idp.example.test/logout?state=x'))
    expect(api.post).toHaveBeenCalledWith('/api/v1/auth/sign-out')
    expect(window.localStorage.getItem('last_active_workspace')).toBeNull()
  })

  it('goes to the sign-in page when the API gives no address, or the session had already ended', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    for (const answer of [() => Promise.resolve(null), () => Promise.reject(unauthorized())]) {
      api.post.mockImplementationOnce(answer)
      const navigate = vi.fn()
      void signOutOfEnterprise(navigate)
      await waitFor(() => expect(navigate).toHaveBeenCalledWith('/sign-in'))
    }
  })
})
