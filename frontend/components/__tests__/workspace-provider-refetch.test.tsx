/**
 * F226 — a board and a count from two workspaces never share a page.
 *
 * Every request names the workspace stored in localStorage; the provider then
 * stores the one /api/workspaces/current answers with. Reads sent before that
 * answer used the old workspace, and their query keys carry none, so the step-2
 * screenshots caught the board on one workspace and Needs you on another. When
 * the workspace changes, every query refetches.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, waitFor, cleanup } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

vi.mock('@/lib/auth-hooks', () => ({
  useAuth: () => ({ isSignedIn: true, getToken: async () => null }),
  useOrganization: () => ({ organization: null }),
}))
vi.mock('next/navigation', () => ({ usePathname: () => '/command-center' }))

import { WorkspaceProvider } from '@/components/workspace-provider'

function open(stored: string | null, served: string) {
  if (stored) localStorage.setItem('last_active_workspace', stored)
  else localStorage.removeItem('last_active_workspace')
  const answered = vi.fn(async () => ({ id: served, name: 'Harbourline', settings: {} }))
  globalThis.fetch = vi.fn(async () => ({ ok: true, status: 200, json: answered })) as unknown as typeof fetch
  const client = new QueryClient()
  const refetch = vi.spyOn(client, 'invalidateQueries')
  render(<QueryClientProvider client={client}><WorkspaceProvider><div /></WorkspaceProvider></QueryClientProvider>)
  return { refetch, answered }
}

/** The provider has read /current's answer and stored it. */
async function settled(answered: ReturnType<typeof vi.fn>) {
  await waitFor(() => expect(answered).toHaveBeenCalled())
  await new Promise((resolve) => setTimeout(resolve, 0))
}

afterEach(cleanup)

describe('WorkspaceProvider', () => {
  it('refetches every query when the workspace in use changes', async () => {
    const { refetch, answered } = open('ws-sim', 'ws-c1')
    await settled(answered)
    expect(localStorage.getItem('last_active_workspace')).toBe('ws-c1')
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it('refetches nothing when the workspace stays the same, or on a first visit', async () => {
    const same = open('ws-c1', 'ws-c1')
    await settled(same.answered)
    expect(same.refetch).not.toHaveBeenCalled()
    cleanup()
    const first = open(null, 'ws-c1')
    await settled(first.answered)
    expect(localStorage.getItem('last_active_workspace')).toBe('ws-c1')
    expect(first.refetch).not.toHaveBeenCalled()
  })
})
