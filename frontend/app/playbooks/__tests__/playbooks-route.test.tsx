/**
 * PRD-246 US-007 — the route forks on the STYLE axis alone: the Studio style
 * renders the Studio surface at every width (it has a compact form now), the
 * Classic style renders the classic one. Width informs layout inside a
 * component, never which component.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

const nav = vi.hoisted(() => ({ replace: vi.fn(), search: 'id=recipe-7' }))
const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => state.tabletOrBelow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/playbooks/PlaybooksPanel', () => ({ PlaybooksPanel: () => <div data-testid="panel" /> }))

import PlaybooksPage from '@/app/playbooks/page'

afterEach(() => { cleanup(); nav.replace.mockClear(); state.tabletOrBelow = false; state.studio = true })

describe('/playbooks', () => {
  it('Studio desktop forwards to the hub with the params kept', async () => {
    render(<PlaybooksPage />)
    await waitFor(() => expect(nav.replace).toHaveBeenCalled())
    expect(String(nav.replace.mock.calls[0][0])).toBe('/assignments?id=recipe-7&tab=playbooks')
    expect(screen.queryByTestId('panel')).toBeNull()
  })
  it('Classic desktop keeps the standalone panel and does not redirect', () => {
    state.studio = false
    render(<PlaybooksPage />)
    expect(screen.getByTestId('panel')).toBeInTheDocument()
    expect(nav.replace).not.toHaveBeenCalled()
  })
  it('the Studio style renders the Studio surface at phone width too (PRD-246 US-007)', () => {
    state.tabletOrBelow = true
    render(<PlaybooksPage />)
    expect(nav.replace).toHaveBeenCalled()
    expect(screen.queryByTestId('panel')).toBeNull()
  })
})
