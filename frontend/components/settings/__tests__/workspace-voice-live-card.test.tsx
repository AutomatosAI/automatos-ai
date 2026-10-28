/**
 * Settings → Voice → Auto Live — the workspace's own switch.
 *
 * An owner or admin flips it: the card calls setWorkspaceVoiceLiveEnabled with the new
 * state and refreshes the workspace, so every reader sees it with no reload. The switch
 * stays visible but disabled, with a plain reason, when the workspace's plan does not
 * include Auto Live, or when the viewer cannot manage the workspace (never hidden: a
 * member still sees the state).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

const setWorkspaceVoiceLiveEnabled = vi.hoisted(() => vi.fn())
const refreshWorkspace = vi.hoisted(() => vi.fn())
const workspaceState = vi.hoisted(() => ({ current: null as any }))

vi.mock('@/lib/api-client', () => ({
  apiClient: { setWorkspaceVoiceLiveEnabled: (on: boolean) => setWorkspaceVoiceLiveEnabled(on) },
}))
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: workspaceState.current, refreshWorkspace }),
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'
import { ASK_AN_ADMIN_TEXT, WorkspaceVoiceLiveCard } from '../WorkspaceVoiceLiveCard'

function workspace(role: string, voiceLive: { available: boolean; enabled: boolean }) {
  return { id: 'ws-1', name: 'Acme', slug: 'acme', plan: 'basic', role, planLimits: {}, voiceLive }
}

function renderCard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <WorkspaceVoiceLiveCard />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  setWorkspaceVoiceLiveEnabled.mockReset()
  setWorkspaceVoiceLiveEnabled.mockResolvedValue({ status: 'saved', voice_live: { enabled: false } })
  refreshWorkspace.mockReset()
  refreshWorkspace.mockResolvedValue(undefined)
})

describe('WorkspaceVoiceLiveCard (Settings → Voice → Auto Live)', () => {
  it('lets an admin turn Auto Live off and refreshes the workspace', async () => {
    workspaceState.current = workspace('admin', { available: true, enabled: true })
    renderCard()

    expect(screen.getByText('ON')).toBeInTheDocument()
    const toggle = screen.getByRole('switch')
    expect(toggle).toHaveAttribute('aria-checked', 'true')
    expect(toggle).not.toBeDisabled()

    fireEvent.click(toggle)
    await waitFor(() => expect(setWorkspaceVoiceLiveEnabled).toHaveBeenCalledWith(false))
    await waitFor(() => expect(refreshWorkspace).toHaveBeenCalledTimes(1))
  })

  it('lets an owner turn Auto Live on', async () => {
    workspaceState.current = workspace('owner', { available: true, enabled: false })
    renderCard()

    expect(screen.getByText('OFF')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('switch'))
    await waitFor(() => expect(setWorkspaceVoiceLiveEnabled).toHaveBeenCalledWith(true))
    await waitFor(() => expect(refreshWorkspace).toHaveBeenCalledTimes(1))
  })

  it("is disabled with the plan's reason when the workspace's plan does not include Auto Live", () => {
    workspaceState.current = workspace('owner', { available: false, enabled: false })
    renderCard()

    const toggle = screen.getByRole('switch')
    expect(toggle).toBeDisabled()
    expect(screen.getByTestId('voice-live-plan-blocked')).toHaveTextContent(
      LIVE_REASON_SENTENCES.voice_live_plan_required,
    )
    expect(setWorkspaceVoiceLiveEnabled).not.toHaveBeenCalled()
  })

  it('shows a member the state but a disabled switch, and says who turns it on', () => {
    workspaceState.current = workspace('editor', { available: true, enabled: true })
    renderCard()

    expect(screen.getByRole('switch')).toBeDisabled()
    expect(screen.getByText(ASK_AN_ADMIN_TEXT)).toBeInTheDocument()
  })

  it('shows OFF and only the plan reason for a downgrade that left `enabled: true` stored', () => {
    // A workspace that was on and then lost the plan (or an old PUT from before the backend
    // gate existed): the master's own gate wins, like WorkspaceSocialsCard's `available && enabled`.
    workspaceState.current = workspace('owner', { available: false, enabled: true })
    renderCard()

    expect(screen.getByText('OFF')).toBeInTheDocument()
    const toggle = screen.getByRole('switch')
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    expect(toggle).toBeDisabled()
    expect(screen.getByTestId('voice-live-plan-blocked')).toHaveTextContent(
      LIVE_REASON_SENTENCES.voice_live_plan_required,
    )
    expect(screen.queryByText(ASK_AN_ADMIN_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText('This workspace can start a live call with Auto.')).not.toBeInTheDocument()
  })
})
