/**
 * PRD-251 S0.5 (D1) — the Socials tab in BOTH Deliverables shells, through the
 * real /deliverables route: absent while `socials.available` is false (and
 * `?tab=socials` then renders Outputs), present with its body when true.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const state = vi.hoisted(() => ({ studio: true, search: '', workspace: null as any }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(state.search),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => false }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ({ workspace: state.workspace, isLoading: false }) }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/deliverables/outputs-feed', () => ({ OutputsFeed: () => <div data-testid="body-outputs" /> }))
vi.mock('@/components/deliverables/deliverables-blogs', () => ({ DeliverablesBlog: () => <div data-testid="body-blogs" /> }))
vi.mock('@/components/documents/blocks/TemplateStudio', () => ({ TemplateStudio: () => <div data-testid="body-templates" /> }))
vi.mock('@/components/workspace/gallery-view', () => ({ GalleryView: () => <div data-testid="gallery" /> }))
vi.mock('@/components/deliverables/socials/socials-tab', () => ({ SocialsTab: () => <div data-testid="body-socials" /> }))

import DeliverablesPage from '@/app/deliverables/page'
import { resolveDeliverableTab, visibleDeliverableTabs } from '@/lib/deliverables/tabs'

function workspaceWith(available: boolean, enabled = false) {
  return { id: 'w1', role: 'owner', socials: { available, enabled } }
}

function tabLabels(): string[] {
  // Studio: the cc-tabs buttons; Classic: the Radix tab triggers.
  const studio = Array.from(document.querySelectorAll('nav.cc-tabs button.cc-tab'))
  const classic = screen.queryAllByRole('tab')
  return (studio.length ? studio : classic).map((el) => (el.textContent || '').trim())
}

beforeEach(() => { state.search = ''; state.workspace = workspaceWith(false) })
afterEach(cleanup)

describe.each([
  ['Studio', true],
  ['Classic', false],
])('the %s shell', (_name, studio) => {
  beforeEach(() => { state.studio = studio })

  it('has no Socials tab while Socials is unavailable, and ?tab=socials renders Outputs', () => {
    state.workspace = workspaceWith(false)
    state.search = 'tab=socials'
    render(<DeliverablesPage />)
    expect(tabLabels()).toEqual(['Outputs', 'Blogs', 'Templates', 'Explorer'])
    expect(screen.getByTestId('body-outputs')).toBeInTheDocument()
    expect(screen.queryByTestId('body-socials')).toBeNull()
  })

  it('a workspace payload without a socials block is the same as unavailable', () => {
    state.workspace = { id: 'w1', role: 'owner' }
    state.search = 'tab=socials'
    render(<DeliverablesPage />)
    expect(tabLabels()).not.toContain('Socials')
    expect(screen.getByTestId('body-outputs')).toBeInTheDocument()
  })

  it('shows the Socials tab and its body when Socials is available', () => {
    state.workspace = workspaceWith(true)
    state.search = 'tab=socials'
    render(<DeliverablesPage />)
    expect(tabLabels()).toEqual(['Outputs', 'Blogs', 'Templates', 'Socials', 'Explorer'])
    expect(screen.getByTestId('body-socials')).toBeInTheDocument()
    expect(screen.queryByTestId('body-outputs')).toBeNull()
  })
})

describe('the tab rule both shells share', () => {
  it('lists socials only when available and falls back to outputs', () => {
    expect(visibleDeliverableTabs(false)).toEqual(['outputs', 'blogs', 'templates'])
    expect(visibleDeliverableTabs(true)).toEqual(['outputs', 'blogs', 'templates', 'socials'])
    expect(resolveDeliverableTab('socials', false)).toBe('outputs')
    expect(resolveDeliverableTab('socials', true)).toBe('socials')
    expect(resolveDeliverableTab('blogs', false)).toBe('blogs')
    expect(resolveDeliverableTab('nonsense', true)).toBe('outputs')
    expect(resolveDeliverableTab(null, true)).toBe('outputs')
  })
})
