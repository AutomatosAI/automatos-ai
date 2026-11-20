/**
 * PRD-277 US-504 — the Team page and the profile page in enterprise.
 *
 *   - No invitations: no "Invite Member", no pending invitations, no invite dialog.
 *   - An install admin (the API's role, from the enterprise session) sees the people waiting for a
 *     workspace, oldest first and a page at a time, and adds one with a role below owner; the add
 *     goes through the enterprise step-up, never the Clerk one (lib/theme-host/api.ts is a double
 *     that fails if it is used).
 *   - An install admin can end a member's sessions and deactivate them, each after asking.
 *   - Anyone else sees neither, and no request is sent.
 *   - The profile page is read-only and says the account is managed by the organisation.
 */
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }))
const team = vi.hoisted(() => ({ fetchMembers: vi.fn(async () => {}) }))

vi.mock('@/lib/edition-name', () => ({ isEnterprise: true, resolveEdition: () => 'enterprise' }))
vi.mock('@/lib/api-client', () => ({ apiClient: { ...api, getBaseUrl: () => '' } }))
vi.mock('@/lib/auth-hooks', () => ({ useAuth: () => ({ orgId: null }) }))
vi.mock('@/lib/theme-host/api', () => ({
  useStepUp: () => {
    throw new Error('the Clerk-side step-up ran in enterprise')
  },
  themeErrorMessage: () => {
    throw new Error('the Clerk-side message ran in enterprise')
  },
}))
vi.mock('@/components/shared', () => ({
  PageHeader: ({ title, actions }: { title: string; actions?: React.ReactNode }) => (
    <header><h1>{title}</h1>{actions}</header>
  ),
  SearchInput: () => null,
}))
vi.mock('@/components/shared/page-header', () => ({
  PageHeader: ({ title }: { title: string }) => <header><h1>{title}</h1></header>,
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
vi.mock('next/navigation', () => ({ useRouter: () => ({ back: vi.fn() }) }))

import { TeamManagement } from '@/components/team/team-management'
import { EnterpriseProfile } from '@/app/settings/profile/enterprise-profile'
import type { SystemRole } from '@/lib/enterprise/auth-api'
import { EnterpriseSessionContext, type EnterpriseSession } from '@/lib/enterprise/session-context'
import { PROFILE_MANAGED_NOTE } from '@/lib/enterprise/messages'

const WAITING_ONE = { user_id: 9, email: 'new@example.test', name: 'New Person', signed_in_first_at: '2026-11-19T10:00:00Z' }
const page = (people: unknown[], total = people.length, offset = 0) => ({ people, total, limit: 50, offset })

function session(role: SystemRole, workspaceRole: string | null = 'editor'): EnterpriseSession {
  return {
    status: 'signed-in',
    workspaceId: 'ws-1',
    signOut: async () => {},
    who: {
      edition: 'enterprise',
      person: { id: 7, email: 'ada@example.test', name: 'Ada Example', system_role: role },
      workspaces: [{ id: 'ws-1', name: 'Research', role: workspaceRole }],
      in_no_workspace: false,
      default_workspace_id: null,
    },
  }
}

function drawTeam(role: SystemRole = 'admin') {
  return render(<EnterpriseSessionContext.Provider value={session(role)}><TeamManagement /></EnterpriseSessionContext.Provider>)
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('the Team page in enterprise', () => {
  it('has no invitations', () => {
    api.get.mockResolvedValue(page([]))
    drawTeam()
    expect(screen.queryByText('Invite Member')).toBeNull()
    expect(screen.queryByText('old@example.test')).toBeNull()
  })

  it('lets an install admin add a person waiting, with a role below owner', async () => {
    api.get.mockResolvedValue(page([WAITING_ONE]))
    api.post.mockResolvedValue({})
    drawTeam()
    expect(await screen.findByText('New Person')).toBeTruthy()
    expect(api.get).toHaveBeenCalledWith('/api/workspaces/ws-1/team/waiting?limit=50&offset=0')
    const select = screen.getByLabelText('Role for New Person') as HTMLSelectElement
    expect(Array.from(select.options).map((option) => option.value)).toEqual(['viewer', 'editor', 'admin'])

    fireEvent.change(select, { target: { value: 'editor' } })
    fireEvent.click(screen.getByText('Add to workspace'))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/workspaces/ws-1/team/waiting/9', { role: 'editor' }))
    await waitFor(() => expect(team.fetchMembers).toHaveBeenCalled())
  })

  it('pages through a long waiting list, oldest first, with the total', async () => {
    const first = Array.from({ length: 50 }, (_, index) => ({ ...WAITING_ONE, user_id: 100 + index, name: `Early ${index}` }))
    api.get.mockResolvedValueOnce(page(first, 51, 0)).mockResolvedValueOnce(page([{ ...WAITING_ONE, name: 'The Latest' }], 51, 50))
    drawTeam()
    expect(await screen.findByText('1–50 of 51, oldest first')).toBeTruthy()
    expect((screen.getByText('Previous') as HTMLButtonElement).disabled).toBe(true)

    fireEvent.click(screen.getByText('Next'))
    expect(await screen.findByText('The Latest')).toBeTruthy()
    expect(api.get).toHaveBeenLastCalledWith('/api/workspaces/ws-1/team/waiting?limit=50&offset=50')
    expect(screen.getByText('51–51 of 51, oldest first')).toBeTruthy()
    expect((screen.getByText('Next') as HTMLButtonElement).disabled).toBe(true)
  })

  it("lets an install admin end a member's sessions and deactivate them, after asking", async () => {
    api.get.mockResolvedValue(page([]))
    api.post.mockResolvedValue({})
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    drawTeam()

    fireEvent.click(screen.getByLabelText("End Grace Example's sessions"))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/v1/auth/people/42/sessions/revoke'))
    fireEvent.click(screen.getByLabelText('Deactivate Grace Example'))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/api/v1/auth/people/42/deactivate'))
    expect(window.confirm).toHaveBeenCalledTimes(2)
  })

  it('shows neither to someone who is no install admin, and asks the API nothing', () => {
    drawTeam('user')
    expect(api.get).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("End Grace Example's sessions")).toBeNull()
  })
})

describe('the profile page in enterprise', () => {
  it('is read-only and says the organisation manages the account', () => {
    render(<EnterpriseSessionContext.Provider value={session('user')}><EnterpriseProfile /></EnterpriseSessionContext.Provider>)

    expect(screen.getByText(PROFILE_MANAGED_NOTE)).toBeTruthy()
    const inputs = Array.from(document.querySelectorAll('input'))
    expect(inputs.length).toBeGreaterThan(0)
    expect(inputs.every((input) => input.readOnly)).toBe(true)
    expect((screen.getByLabelText('Email') as HTMLInputElement).value).toBe('ada@example.test')
    expect((screen.getByLabelText('Your role') as HTMLInputElement).value).toBe('editor')
    expect(screen.queryByRole('button', { name: /save/i })).toBeNull()
  })

  it('shows an empty role, not a made-up one, when the member row holds none', () => {
    render(<EnterpriseSessionContext.Provider value={session('user', null)}><EnterpriseProfile /></EnterpriseSessionContext.Provider>)
    expect((screen.getByLabelText('Your role') as HTMLInputElement).value).toBe('')
  })
})
