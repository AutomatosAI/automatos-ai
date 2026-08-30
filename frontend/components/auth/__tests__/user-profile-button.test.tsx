/**
 * PRD-233 S6 — the profile slot per edition.
 *
 * local → the operator's initials (or avatar) + name from GET /api/profile,
 *         linking to /settings/profile; no Clerk component is rendered.
 * saas  → Clerk's UserButton exactly as before; the profile API is never called.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

const { getProfileMock, clerkUseUserMock } = vi.hoisted(() => ({
  getProfileMock: vi.fn(),
  clerkUseUserMock: vi.fn(),
}))

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: any) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

vi.mock('@/lib/api-client', () => ({
  apiClient: { getProfile: getProfileMock },
}))

vi.mock('@clerk/nextjs', () => ({
  useUser: clerkUseUserMock,
  useAuth: vi.fn(),
  useSession: vi.fn(),
  useOrganization: vi.fn(),
  useClerk: vi.fn(),
  UserButton: () => <div data-testid="clerk-user-button" />,
}))

async function loadButton(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  return import('../user-profile-button')
}

beforeEach(() => {
  getProfileMock.mockReset()
  clerkUseUserMock.mockReset()
})

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('initialsFor', () => {
  it('derives initials from the name, then the email, then a placeholder', async () => {
    const { initialsFor } = await loadButton('local')
    expect(initialsFor('Ada Lovelace', null)).toBe('AL')
    expect(initialsFor('Ada Byron Lovelace', null)).toBe('AL')
    expect(initialsFor('ada', null)).toBe('AD')
    expect(initialsFor('  ', 'operator@example.test')).toBe('O')
    expect(initialsFor(null, null)).toBe('?')
  })
})

describe('UserProfileButton (PRD-233 S6)', () => {
  it('local: shows the operator initials + name and links to /settings/profile', async () => {
    getProfileMock.mockResolvedValue({
      edition: 'local',
      editable: true,
      id: 1,
      email: 'local@automatos.local',
      name: 'Local Operator',
      username: 'local',
      avatar_url: null,
      system_role: 'super_admin',
      email_note: '',
    })
    const { UserProfileButton } = await loadButton('local')
    render(<UserProfileButton />)

    const link = await screen.findByRole('link', { name: /your profile/i })
    expect(link).toHaveAttribute('href', '/settings/profile')
    expect(await screen.findByText('Local Operator')).toBeInTheDocument()
    expect(link).toHaveTextContent('LO')
    expect(screen.queryByTestId('clerk-user-button')).not.toBeInTheDocument()
    expect(clerkUseUserMock).not.toHaveBeenCalled()
  })

  it('local: still links to the profile when the API fails (no name yet)', async () => {
    getProfileMock.mockRejectedValue(new Error('offline'))
    const { UserProfileButton } = await loadButton('local')
    render(<UserProfileButton />)

    const link = await screen.findByRole('link', { name: /your profile/i })
    expect(link).toHaveAttribute('href', '/settings/profile')
    expect(link).toHaveTextContent('Operator')
  })

  it('saas: renders the Clerk UserButton and never calls the profile API', async () => {
    clerkUseUserMock.mockReturnValue({ isSignedIn: true, isLoaded: true, user: {} })
    const { UserProfileButton } = await loadButton('saas')
    render(<UserProfileButton />)

    expect(screen.getByTestId('clerk-user-button')).toBeInTheDocument()
    expect(getProfileMock).not.toHaveBeenCalled()
  })
})
