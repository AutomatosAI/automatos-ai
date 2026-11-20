/**
 * PRD-277 US-504 — the Team page and the profile page in enterprise.
 *
 *   - No invitations: no "Invite Member", no pending invitations, no invite dialog.
 *   - An install admin (the API's role) sees the people waiting for a workspace and adds one with
 *     a role below owner; the add goes through step-up (useStepUp).
 *   - An install admin can end a member's sessions and deactivate them, each after asking.
 *   - Anyone else sees neither.
 *   - The profile page is read-only and says the account is managed by the organisation.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const role = vi.hoisted(() => ({ systemRole: 'admin' as 'admin' | 'user' }))
const team = vi.hoisted(() => ({ fetchMembers: vi.fn(async () => {}) }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => ({ apiClient: { ...api, getBaseUrl: () => '' } }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ orgId: null }) }))
vi.mock('@/contexts/role-context', () => ({ useSystemRoleOptional: () => ({ systemRole: role.systemRole }) }))
vi.mock('@/lib/theme-host/api', () => ({
  useStepUp: <T,>(action: T) => action,
  themeErrorMessage: (error: unknown) => String(error),
}))
vi.mock('@/components/shared', () => ({
  PageHeader: ({ title, actions }: { title: string; actions?: React.ReactNode }) => (
    <header><h1>{title}</h1>{actions}</header>
  ),
  SearchInput: () => null,
}))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children?: React.ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))
vi.mock('@/components/team/use-team-members', () => ({
  useTeamMembers: () => ({
    workspaceId: 'ws-1',
    loading: false,
    members: [{ id: 3, user_id: '42', email: 'grace@example.test', name: 'Grace Example', role: 'editor', joined_at: '2026-11-01T00:00:00Z', last_active_at: null, tool_runs_30d: null, chats_30d: null, invited_by_email: null }],
    pendingInvites: [{ id: 1, email: 'old@example.test', role: 'viewer', status: 'pending', expires_at: '', created_at: '' }],
    fetchMembers: team.fetchMembers,
    handleRevokeInvite: vi.fn(),
    handleRoleChange: vi.fn(),
    handleRemoveMember: vi.fn(),
  }),
}))

import { TeamManagement } from '@/components/team/team-management'
import { EnterpriseProfile } from '@/app/settings/profile/enterprise-profile'
import { EnterpriseSessionContext, type EnterpriseSession } from '@/lib/enterprise/session-context'
import { PROFILE_MANAGED_NOTE } from '@/lib/enterprise/messages'

vi.mock('next/navigation', () => ({ useRouter: () => ({ back: vi.fn() }) }))

const WAITING = { people: [{ user_id: 9, email: 'new@example.test', name: 'New Person', signed_in_first_at: '2026-11-19T10:00:00Z' }] }

afterEach(() => {
  vi.clearAllMocks()
  role.systemRole = 'admin'
})

describe('the Team page in enterprise', () => {
  it('has no invitations', () => {
    api.get.mockResolvedValue({ people: [] })
    render(<TeamManagement />)
    expect(screen.queryByText('Invite Member')).toBeNull()
    expect(screen.queryByText('old@example.test')).toBeNull()
  })

  it('lets an install admin add a person waiting, with a role below owner', async () => {
    api.get.mockResolvedValue(WAITING)
    api.post.mockResolvedValue({})
    render(<TeamManagement />)
    expect(await screen.findByText('New Person')).toBeTruthy()
    const select = screen.getByLabelText('Role for New Person') as HTMLSelectElement
    expect(Array.from(select.options).map((option) => option.value)).toEqual(['viewer', 'editor', 'admin'])

    fireEvent.change(select, { target: { value: 'editor' } })
    fireEvent.click(screen.getByText('Add to workspace'))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/workspaces/ws-1/team/waiting/9', { role: 'editor' }))
    await waitFor(() => expect(team.fetchMembers).toHaveBeenCalled())
  })

  it("lets an install admin end a member's sessions and deactivate them, after asking", async () => {
    api.get.mockResolvedValue({ people: [] })
    api.post.mockResolvedValue({})
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    render(<TeamManagement />)

    fireEvent.click(screen.getByLabelText("End Grace Example's sessions"))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/v1/auth/people/42/sessions/revoke'))
    fireEvent.click(screen.getByLabelText('Deactivate Grace Example'))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/v1/auth/people/42/deactivate'))
    expect(window.confirm).toHaveBeenCalledTimes(2)
  })

  it('shows neither to someone who is no install admin', () => {
    role.systemRole = 'user'
    render(<TeamManagement />)
    expect(api.get).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("End Grace Example's sessions")).toBeNull()
  })
})

describe('the profile page in enterprise', () => {
  it('is read-only and says the organisation manages the account', () => {
    const session: EnterpriseSession = {
      status: 'signed-in',
      workspaceId: 'ws-1',
      signOut: async () => {},
      who: {
        edition: 'enterprise',
        person: { id: 7, email: 'ada@example.test', name: 'Ada Example', system_role: 'user' },
        workspaces: [{ id: 'ws-1', name: 'Research', role: 'editor' }],
        in_no_workspace: false,
        default_workspace_id: null,
      },
    }
    render(<EnterpriseSessionContext.Provider value={session}><EnterpriseProfile /></EnterpriseSessionContext.Provider>)

    expect(screen.getByText(PROFILE_MANAGED_NOTE)).toBeTruthy()
    const inputs = Array.from(document.querySelectorAll('input'))
    expect(inputs.length).toBeGreaterThan(0)
    expect(inputs.every((input) => input.readOnly)).toBe(true)
    expect((screen.getByLabelText('Email') as HTMLInputElement).value).toBe('ada@example.test')
    expect(screen.queryByRole('button', { name: /save/i })).toBeNull()
  })
})
