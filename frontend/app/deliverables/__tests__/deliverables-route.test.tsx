/**
 * PRD-246 US-007 — the route forks on the STYLE axis alone: the Studio style
 * renders the Studio surface at every width (it has a compact form now), the
 * Classic style renders the classic one. Width informs layout inside a
 * component, never which component.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('next/navigation', () => ({ useRouter: () => ({ replace: vi.fn(), push: vi.fn() }), useSearchParams: () => new URLSearchParams('') }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => state.tabletOrBelow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ workspace: null, isLoading: true }) }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/deliverables/studio/deliverables-studio', () => ({ DeliverablesStudio: () => <div data-testid="studio" /> }))
vi.mock('@/components/deliverables/outputs-feed', () => ({ OutputsFeed: () => null }))
vi.mock('@/components/deliverables/deliverables-blogs', () => ({ DeliverablesBlog: () => null }))
vi.mock('@/components/documents/blocks/TemplateStudio', () => ({ TemplateStudio: () => null }))
vi.mock('@/components/workspace/gallery-view', () => ({ GalleryView: () => null }))

import DeliverablesPage from '@/app/deliverables/page'

afterEach(() => { cleanup(); state.tabletOrBelow = false; state.studio = true })

describe('Deliverables route', () => {
  it('Studio desktop → the Studio page', () => {
    render(<DeliverablesPage />)
    expect(screen.getByTestId('studio')).toBeInTheDocument()
  })
  it('Classic desktop → the classic page (its loading state here), never the Studio one', () => {
    state.studio = false
    render(<DeliverablesPage />)
    expect(screen.queryByTestId('studio')).toBeNull()
  })
  it('the Studio style renders the Studio surface at phone width too (PRD-246 US-007)', () => {
    state.tabletOrBelow = true
    render(<DeliverablesPage />)
    expect(screen.getByTestId('studio')).toBeInTheDocument()
  })
})
