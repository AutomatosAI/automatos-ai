/**
 * PRD-233 S7 — the SaaS-only route pages in each edition.
 *
 * `local`: Team, Workspace Admin and Invitations render the shared notice
 * (the route still resolves — hidden ≠ deleted); the Clerk sign-in sub-flows
 * (reset-password, sso-callback) send visitors home like sign-in/up do.
 * `saas`: every page renders its hosted UI exactly as today.
 *
 * The real seam is exercised (env stubbed, module graph reset per test). Heavy
 * children, the layout shell, Clerk and the API client are stubbed.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, waitFor, act } from '@testing-library/react'
import React from 'react'

// Each case dynamically imports a page (and its tree) after stubbing the
// edition; the first transform can exceed vitest's 5s default under load.
vi.setConfig({ testTimeout: 30_000 })

const redirectMock = vi.hoisted(() =>
  vi.fn((to: string) => {
    throw new Error(`NEXT_REDIRECT:${to}`)
  }),
)

vi.mock('next/navigation', () => ({
  redirect: redirectMock,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(''),
  usePathname: () => '/',
}))
vi.mock('next/image', () => ({
  default: (props: { alt?: string; src?: unknown }) =>
    React.createElement('img', { alt: props.alt ?? '', src: typeof props.src === 'string' ? props.src : '' }),
}))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="main-layout">{children}</div>,
}))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/team/team-management', () => ({
  TeamManagement: () => <div data-testid="team-management" />,
}))
vi.mock('@/lib/api-client', () => ({
  apiClient: { request: vi.fn(async () => ({ items: [], total: 0, page: 1, limit: 20 })) },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: null, refreshWorkspace: vi.fn(), isLoading: false }),
}))
vi.mock('@clerk/nextjs', () => ({
  useAuth: () => ({ isLoaded: true, isSignedIn: false }),
  useUser: () => ({ user: null }),
  useSignIn: () => ({ isLoaded: false, signIn: null, setActive: null }),
  SignIn: () => <div data-testid="clerk-sign-in" />,
  SignUp: () => <div data-testid="clerk-sign-up" />,
  AuthenticateWithRedirectCallback: () => <div data-testid="clerk-sso-callback" />,
}))

const NOTICE = 'This area is part of the hosted edition; the local edition has no accounts, teams or plans.'

const pages = {
  team: () => import('@/app/team/page'),
  admin: () => import('@/app/admin/workspaces/page'),
  invitation: () => import('@/app/accept-invitation/page'),
  reset: () => import('@/app/reset-password/page'),
  sso: () => import('@/app/sso-callback/page'),
  plugins: () => import('@/app/admin/plugins/page'),
  devReset: () => import('@/app/dev/reset-onboarding/page'),
}

async function load(edition: 'local' | 'saas', key: keyof typeof pages) {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const mod = await pages[key]()
  return mod.default
}

beforeEach(() => {
  // accept-invitation looks its token up on mount; keep it off the network.
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, json: async () => ({}) })))
})

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.resetModules()
  redirectMock.mockClear()
})

describe('/team', () => {
  it('local: renders the notice inside the shell, not TeamManagement', async () => {
    const TeamPage = await load('local', 'team')
    render(<TeamPage />)

    expect(screen.getByTestId('main-layout')).toBeInTheDocument()
    expect(screen.getByTestId('saas-only-notice')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Team' })).toBeInTheDocument()
    expect(screen.getByText(NOTICE)).toBeInTheDocument()
    expect(screen.queryByTestId('team-management')).toBeNull()
  })

  it('saas: renders TeamManagement exactly as today', async () => {
    const TeamPage = await load('saas', 'team')
    render(<TeamPage />)

    expect(screen.getByTestId('team-management')).toBeInTheDocument()
    expect(screen.queryByTestId('saas-only-notice')).toBeNull()
  })
})

describe('/admin/workspaces', () => {
  it('local: renders the notice inside the shell; the admin console never mounts', async () => {
    const AdminPage = await load('local', 'admin')
    render(<AdminPage />)

    expect(screen.getByTestId('main-layout')).toBeInTheDocument()
    expect(screen.getByTestId('saas-only-notice')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Workspace Admin' })).toBeInTheDocument()
    expect(screen.queryByText(/Manage all workspaces/)).toBeNull()
    const { apiClient } = await import('@/lib/api-client')
    expect(apiClient.request).not.toHaveBeenCalled()
  })

  it('saas: renders the admin console exactly as today', async () => {
    const AdminPage = await load('saas', 'admin')
    const { apiClient } = await import('@/lib/api-client')
    render(<AdminPage />)

    expect(screen.queryByTestId('saas-only-notice')).toBeNull()
    expect(screen.getByRole('heading', { name: /Workspace Admin/ })).toBeInTheDocument()
    expect(screen.getByText(/Manage all workspaces/)).toBeInTheDocument()
    // The console's list fetch runs on mount (the hosted path is live); let its
    // resolved state update land inside the test instead of after it.
    await waitFor(() => expect(apiClient.request).toHaveBeenCalled())
    await act(async () => {
      await Promise.resolve()
    })
  })
})

describe('/accept-invitation', () => {
  it('local: renders the notice; no Clerk hook runs', async () => {
    const AcceptInvitationPage = await load('local', 'invitation')
    render(<AcceptInvitationPage />)

    expect(screen.getByTestId('saas-only-notice')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Invitations' })).toBeInTheDocument()
    expect(globalThis.fetch).not.toHaveBeenCalled()
  })

  it('saas: renders the invitation flow exactly as today', async () => {
    const AcceptInvitationPage = await load('saas', 'invitation')
    render(<AcceptInvitationPage />)

    expect(screen.queryByTestId('saas-only-notice')).toBeNull()
    // No token in the URL ⇒ the flow's own error state.
    expect(await screen.findByText('Invalid link')).toBeInTheDocument()
  })
})

describe('Clerk sign-in sub-flows', () => {
  let consoleError: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
  })
  afterEach(() => {
    consoleError.mockRestore()
  })

  it('/reset-password local: redirects home before any Clerk hook runs', async () => {
    const ResetPasswordPage = await load('local', 'reset')
    expect(() => render(<ResetPasswordPage />)).toThrow('NEXT_REDIRECT:/')
    expect(redirectMock).toHaveBeenCalledWith('/')
  })

  it('/reset-password saas: renders the reset form, no redirect', async () => {
    const ResetPasswordPage = await load('saas', 'reset')
    const { container } = render(<ResetPasswordPage />)
    expect(redirectMock).not.toHaveBeenCalled()
    expect(container.querySelector('form')).not.toBeNull()
  })

  it('/sso-callback local: redirects home', async () => {
    const SSOCallbackPage = await load('local', 'sso')
    expect(() => render(<SSOCallbackPage />)).toThrow('NEXT_REDIRECT:/')
    expect(redirectMock).toHaveBeenCalledWith('/')
  })

  it('/sso-callback saas: mounts Clerk\'s callback exactly as today', async () => {
    const SSOCallbackPage = await load('saas', 'sso')
    render(<SSOCallbackPage />)
    expect(redirectMock).not.toHaveBeenCalled()
    expect(screen.getByTestId('clerk-sso-callback')).toBeInTheDocument()
  })
})

describe('/admin/plugins (hosted hub moderation queue)', () => {
  it('local: renders the notice inside the shell; the console never mounts', async () => {
    const AdminPluginsPage = await load('local', 'plugins')
    // Mocked-module spies survive vi.resetModules(); start this count clean.
    const { apiClient } = await import('@/lib/api-client')
    vi.mocked(apiClient.request).mockClear()
    render(<AdminPluginsPage />)

    expect(screen.getByTestId('main-layout')).toBeInTheDocument()
    expect(screen.getByTestId('saas-only-notice')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Plugin moderation' })).toBeInTheDocument()
    expect(apiClient.request).not.toHaveBeenCalled()
  })

  it('saas: renders the moderation console exactly as today', async () => {
    const AdminPluginsPage = await load('saas', 'plugins')
    const { apiClient } = await import('@/lib/api-client')
    render(<AdminPluginsPage />)

    expect(screen.queryByTestId('saas-only-notice')).toBeNull()
    await waitFor(() => expect(apiClient.request).toHaveBeenCalled())
    await act(async () => {
      await Promise.resolve()
    })
  })
})

describe('/dev/reset-onboarding (Clerk-only dev page)', () => {
  let consoleError: ReturnType<typeof vi.spyOn>
  beforeEach(() => {
    consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
  })
  afterEach(() => {
    consoleError.mockRestore()
  })

  it('local: redirects home before any Clerk hook runs', async () => {
    const ResetOnboardingPage = await load('local', 'devReset')
    expect(() => render(<ResetOnboardingPage />)).toThrow('NEXT_REDIRECT:/')
    expect(redirectMock).toHaveBeenCalledWith('/')
  })

  it('saas: renders the dev page, no redirect', async () => {
    const ResetOnboardingPage = await load('saas', 'devReset')
    const { container } = render(<ResetOnboardingPage />)
    expect(redirectMock).not.toHaveBeenCalled()
    expect(container).not.toBeEmptyDOMElement()
  })
})
