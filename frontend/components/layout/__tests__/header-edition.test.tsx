/**
 * PRD-233 S7 — the header's account slot is a SaaS surface.
 *
 * ProfileMenu's signed-out branch is a "Sign In" button (router.push('/sign-in'))
 * and its signed-in branch is Clerk's account menu (Profile / Logout). The local
 * edition has no accounts: the LOCAL_USER seam (lib/auth-hooks) carries a null
 * user, so ProfileMenu rendered a sign-in affordance on every local page. Both
 * headers (classic + studio) now mount ProfileMenu only in `saas`; the rest of the
 * utilities cluster (help, theme, notifications) is edition-neutral and stays.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

// Dynamic imports transform the whole rail (framer-motion, lucide) on first use;
// under CI/parallel load that can exceed vitest's 5s default and a timed-out
// render would leak DOM into the next case. Generous budget + explicit cleanup.
vi.setConfig({ testTimeout: 30_000 })

vi.mock('@/components/auth/profile-menu', () => ({
  ProfileMenu: () => <div data-testid="profile-menu">Sign In</div>,
}))
vi.mock('@/components/auth/user-profile-button', () => ({
  UserProfileButton: () => <button data-testid="user-profile-button">operator</button>,
}))
vi.mock('@/components/notifications/notification-bell', () => ({
  NotificationBell: () => <div data-testid="notification-bell" />,
}))
vi.mock('@/components/ui/theme-toggle', () => ({
  ThemeToggle: () => <div data-testid="theme-toggle" />,
}))

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.resetModules()
})

async function load(edition: 'local' | 'saas') {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  const [{ Header }, { StudioHeader }] = await Promise.all([
    import('../header'),
    import('../studio-header'),
  ])
  return { Header, StudioHeader }
}

describe('Header (classic) — account slot per edition', () => {
  it('local: no ProfileMenu (no sign-in affordance); help, theme, notifications remain', async () => {
    const { Header } = await load('local')
    render(<Header onMenuClick={() => {}} />)

    expect(screen.queryByTestId('profile-menu')).toBeNull()
    expect(screen.queryByText('Sign In')).toBeNull()
    expect(screen.getByTestId('theme-toggle')).toBeInTheDocument()
    expect(screen.getByTestId('notification-bell')).toBeInTheDocument()
    expect(screen.getByLabelText('Help & documentation')).toBeInTheDocument()
  })

  it('saas: ProfileMenu mounts exactly as today', async () => {
    const { Header } = await load('saas')
    render(<Header onMenuClick={() => {}} />)

    expect(screen.getByTestId('profile-menu')).toBeInTheDocument()
    expect(screen.getByTestId('theme-toggle')).toBeInTheDocument()
    expect(screen.getByTestId('notification-bell')).toBeInTheDocument()
  })
})

describe('StudioHeader — account slot per edition', () => {
  it('local: no ProfileMenu; the utilities cluster remains', async () => {
    const { StudioHeader } = await load('local')
    render(<StudioHeader />)

    expect(screen.queryByTestId('profile-menu')).toBeNull()
    expect(screen.getByTestId('theme-toggle')).toBeInTheDocument()
    expect(screen.getByTestId('notification-bell')).toBeInTheDocument()
    expect(screen.getByLabelText('Open search (⌘K)')).toBeInTheDocument()
  })

  it('saas: ProfileMenu mounts exactly as today', async () => {
    const { StudioHeader } = await load('saas')
    render(<StudioHeader />)
    expect(screen.getByTestId('profile-menu')).toBeInTheDocument()
  })
})
