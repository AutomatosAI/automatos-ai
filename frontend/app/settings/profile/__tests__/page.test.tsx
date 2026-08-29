/**
 * PRD-233 S6 — Settings → Profile, one route, two editions.
 *
 *   local → a plain form over GET/PUT /api/profile (name / username / avatar;
 *           email read-only with the LOCAL_OPERATOR_EMAIL hint). No Clerk hook
 *           is ever called.
 *   saas  → the Clerk-managed page renders exactly as before and never touches
 *           the profile API.
 *
 * The edition seam (`@/lib/auth-edition`) is mocked per test; everything else
 * that reaches the network (api-client, workspace hook, Clerk) is stubbed.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'

const { getProfileMock, updateProfileMock, clerkUseUserMock } = vi.hoisted(() => ({
  getProfileMock: vi.fn(),
  updateProfileMock: vi.fn(),
  clerkUseUserMock: vi.fn(),
}))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ back: vi.fn(), push: vi.fn() }),
}))

vi.mock('next/link', () => ({
  default: ({ href, children, ...rest }: any) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}))

vi.mock('@/hooks/use-workspace', () => ({
  useWorkspace: () => ({ workspaceId: '00000000-0000-0000-0000-0000000000aa', loading: false, error: null }),
}))

vi.mock('@/lib/api-client', () => ({
  apiClient: { getProfile: getProfileMock, updateProfile: updateProfileMock },
}))

vi.mock('@/components/shared', () => ({
  PageHeader: ({ title, titleAccent }: any) => (
    <h1>
      {title} {titleAccent}
    </h1>
  ),
}))

// auth-hooks.ts reads every Clerk hook at module evaluation — all must exist.
vi.mock('@clerk/nextjs', () => ({
  useUser: clerkUseUserMock,
  useAuth: vi.fn(),
  useSession: vi.fn(),
  useOrganization: vi.fn(),
  useClerk: vi.fn(),
  UserButton: () => <div data-testid="clerk-user-button" />,
}))

const LOCAL_PROFILE = {
  edition: 'local',
  editable: true,
  id: 1,
  email: 'local@automatos.local',
  name: 'Local Operator',
  username: 'local',
  avatar_url: null,
  system_role: 'super_admin',
  email_note:
    'Email is the operator lookup key — it is set by LOCAL_OPERATOR_EMAIL in .env and cannot be changed here.',
}

async function loadPage(edition: 'local' | 'saas') {
  vi.doMock('@/lib/auth-edition', () => ({
    authEdition: edition,
    isLocal: edition === 'local',
    isSaaS: edition === 'saas',
  }))
  const mod = await import('../page')
  return mod.default
}

beforeEach(() => {
  getProfileMock.mockReset()
  updateProfileMock.mockReset()
  clerkUseUserMock.mockReset()
})

afterEach(() => {
  vi.doUnmock('@/lib/auth-edition')
  vi.resetModules()
})

describe('Settings → Profile (PRD-233 S6) — local edition', () => {
  it('renders the operator form from GET /api/profile with a read-only email and never calls Clerk', async () => {
    getProfileMock.mockResolvedValue(LOCAL_PROFILE)
    const Page = await loadPage('local')
    render(<Page />)

    const nameInput = (await screen.findByLabelText('Display name')) as HTMLInputElement
    expect(nameInput.value).toBe('Local Operator')
    expect((screen.getByLabelText('Username') as HTMLInputElement).value).toBe('local')

    const emailInput = screen.getByLabelText('Email') as HTMLInputElement
    expect(emailInput.value).toBe('local@automatos.local')
    expect(emailInput).toHaveAttribute('readonly')
    expect(screen.getByText(/LOCAL_OPERATOR_EMAIL/)).toBeInTheDocument()

    // Initials avatar from the seeded name.
    expect(screen.getByLabelText('Avatar initials')).toHaveTextContent('LO')

    expect(getProfileMock).toHaveBeenCalledTimes(1)
    expect(clerkUseUserMock).not.toHaveBeenCalled()
    expect(screen.queryByTestId('clerk-user-button')).not.toBeInTheDocument()
  })

  it('saves only the changed fields through PUT /api/profile and confirms', async () => {
    getProfileMock.mockResolvedValue(LOCAL_PROFILE)
    updateProfileMock.mockResolvedValue({ ...LOCAL_PROFILE, name: 'Test Operator' })
    const Page = await loadPage('local')
    render(<Page />)

    const nameInput = (await screen.findByLabelText('Display name')) as HTMLInputElement
    const save = screen.getByRole('button', { name: /save/i })
    expect(save).toBeDisabled()

    fireEvent.change(nameInput, { target: { value: 'Test Operator' } })
    expect(save).toBeEnabled()
    fireEvent.click(save)

    await waitFor(() => expect(updateProfileMock).toHaveBeenCalledWith({ name: 'Test Operator' }))
    expect(await screen.findByText('Profile updated.')).toBeInTheDocument()
    expect((screen.getByLabelText('Display name') as HTMLInputElement).value).toBe('Test Operator')
  })

  it('shows the backend detail when a save is rejected', async () => {
    getProfileMock.mockResolvedValue(LOCAL_PROFILE)
    updateProfileMock.mockRejectedValue(new Error('username is already taken'))
    const Page = await loadPage('local')
    render(<Page />)

    fireEvent.change(await screen.findByLabelText('Username'), { target: { value: 'taken' } })
    fireEvent.click(screen.getByRole('button', { name: /save/i }))

    expect(await screen.findByText('username is already taken')).toBeInTheDocument()
  })

  it('makes a FastAPI 422 detail readable', async () => {
    const { readableError } = await import('../local-profile-form')
    const detail = JSON.stringify([{ loc: ['body', 'avatar_url'], msg: 'Value error, avatar_url must be an http(s) URL' }])
    expect(readableError(new Error(detail))).toBe('avatar_url: Value error, avatar_url must be an http(s) URL')
    expect(readableError(new Error('plain message'))).toBe('plain message')
  })
})

describe('Settings → Profile (PRD-233 S6) — saas edition', () => {
  it('renders the Clerk-managed page untouched and never calls the profile API', async () => {
    clerkUseUserMock.mockReturnValue({
      isLoaded: true,
      user: {
        fullName: 'Sample User',
        firstName: 'Sample',
        lastName: 'User',
        username: 'sample',
        imageUrl: '',
        emailAddresses: [{ id: 'e1', emailAddress: 'sample@example.test' }],
        primaryEmailAddressId: 'e1',
        primaryEmailAddress: { emailAddress: 'sample@example.test' },
      },
    })
    const Page = await loadPage('saas')
    render(<Page />)

    expect(await screen.findByText('Edit Profile')).toBeInTheDocument()
    expect(screen.getByText('Sample User')).toBeInTheDocument()
    expect(clerkUseUserMock).toHaveBeenCalled()
    expect(getProfileMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText('Display name')).not.toBeInTheDocument()
  })
})
