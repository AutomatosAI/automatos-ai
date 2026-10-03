/**
 * PRD-251B US-B106 — the workspace's own Socials switch in Settings.
 *
 * An owner or admin flips it: the card calls setWorkspaceSocialsEnabled with the new
 * state and refreshes the workspace, so every gate reads it without a reload. While the
 * platform master is off the card is read-only and says who turns it on; a member
 * without workspace:manage sees the state but no live switch.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const setWorkspaceSocialsEnabled = vi.hoisted(() => vi.fn())
const refreshWorkspace = vi.hoisted(() => vi.fn())
const workspaceState = vi.hoisted(() => ({ current: null as any }))

vi.mock('@/lib/api-client', () => ({
  apiClient: { setWorkspaceSocialsEnabled: (on: boolean) => setWorkspaceSocialsEnabled(on) },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: workspaceState.current, refreshWorkspace }),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { ASK_AN_ADMIN_TEXT, MASTER_OFF_TEXT, WorkspaceSocialsCard } from '../WorkspaceSocialsCard'

function workspace(role: string, socials: { available: boolean; enabled: boolean }) {
  return { id: 'ws-1', name: 'Acme', slug: 'acme', plan: 'basic', role, planLimits: {}, socials }
}

function renderCard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WorkspaceSocialsCard />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  setWorkspaceSocialsEnabled.mockReset()
  setWorkspaceSocialsEnabled.mockResolvedValue({ status: 'saved', socials: { available: true, enabled: false } })
  refreshWorkspace.mockReset()
  refreshWorkspace.mockResolvedValue(undefined)
})

describe('WorkspaceSocialsCard (PRD-251B US-B106)', () => {
  it('lets an admin turn Socials off and refreshes the workspace', async () => {
    workspaceState.current = workspace('admin', { available: true, enabled: true })
    renderCard()

    expect(screen.getByText('ON')).toBeInTheDocument()
    const toggle = screen.getByRole('switch')
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(toggle).not.toBeDisabled()

    fireEvent.click(toggle)
    await waitFor(() => expect(setWorkspaceSocialsEnabled).toHaveBeenCalledWith(false))
    await waitFor(() => expect(refreshWorkspace).toHaveBeenCalledTimes(1))
  })

  it('lets an owner turn Socials on', async () => {
    workspaceState.current = workspace('owner', { available: true, enabled: false })
    renderCard()

    expect(screen.getByText('OFF')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(setWorkspaceSocialsEnabled).toHaveBeenCalledWith(true))
    await waitFor(() => expect(refreshWorkspace).toHaveBeenCalledTimes(1))
  })

  it('is read-only while the platform master is off, and says who turns it on', () => {
    workspaceState.current = workspace('owner', { available: false, enabled: true })
    renderCard()

    expect(screen.getByText('OFF')).toBeInTheDocument()  // the master off means off, whatever the workspace stored
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(screen.getByTestId('socials-master-off')).toHaveTextContent(MASTER_OFF_TEXT)
    expect(setWorkspaceSocialsEnabled).not.toHaveBeenCalled()
  })

  it('shows a member the state but no live switch', () => {
    workspaceState.current = workspace('editor', { available: true, enabled: true })
    renderCard()

    expect(screen.getByRole('switch')).toBeDisabled()
    expect(screen.getByText(ASK_AN_ADMIN_TEXT)).toBeInTheDocument()
  })
})
