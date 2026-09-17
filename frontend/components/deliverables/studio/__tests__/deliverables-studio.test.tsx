/** PRD-244 W5b — the Studio Deliverables page: in-page tabs from ?tab=, Explorer as a route, drill-down back to the feed, the shared bodies. */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

const nav = vi.hoisted(() => ({ replace: vi.fn(), push: vi.fn(), search: '' }))
const ws = vi.hoisted(() => ({ workspace: { id: 'w1' } as any, isLoading: false }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: nav.push }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => ws }))
vi.mock('@/components/deliverables/outputs-feed', () => ({ OutputsFeed: () => <div data-testid="body-outputs" /> }))
vi.mock('@/components/deliverables/deliverables-blogs', () => ({ DeliverablesBlog: ({ variant }: { variant?: string }) => <div data-testid="body-blogs" data-variant={variant} /> }))
vi.mock('@/components/documents/blocks/TemplateStudio', () => ({ TemplateStudio: () => <div data-testid="body-templates" /> }))
vi.mock('@/components/workspace/gallery-view', () => ({ GalleryView: () => <div data-testid="gallery" /> }))
vi.mock('@/components/icons/deliverable-icon', () => ({ isDeliverableType: (t: string) => t === 'report', deliverableLabel: () => 'Reports' }))
vi.mock('@/hooks/use-deliverables-api', () => ({ DEFAULT_FILTERS: {}, FEED_DEFAULT_FILTERS: {} }))

import { DeliverablesStudio } from '@/components/deliverables/studio/deliverables-studio'

beforeEach(() => { nav.replace.mockClear(); nav.push.mockClear(); nav.search = ''; ws.isLoading = false; ws.workspace = { id: 'w1' } })
afterEach(cleanup)

describe('DeliverablesStudio', () => {
  it('renders the head, four tabs (Explorer last) and the outputs feed by default', () => {
    const { container } = render(<DeliverablesStudio />)
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Deliverables')
    expect(container.querySelectorAll('nav.cc-tabs button.cc-tab')).toHaveLength(4)
    expect(screen.getByTestId('body-outputs')).toBeInTheDocument()
  })

  it('reads ?tab=, writes it back on click, and sends Explorer to its route', () => {
    nav.search = 'tab=blogs'
    render(<DeliverablesStudio />)
    expect(screen.getByTestId('body-blogs')).toHaveAttribute('data-variant', 'studio')
    fireEvent.click(screen.getByRole('button', { name: 'Templates' }))
    expect(nav.replace).toHaveBeenCalledWith('/deliverables?tab=templates')
    fireEvent.click(screen.getByRole('button', { name: 'Explorer' }))
    expect(nav.push).toHaveBeenCalledWith('/deliverables/explorer')
  })

  it('a type drill-down shows the gallery with a way back to the feed', () => {
    nav.search = 'tab=outputs&artifact_type=report'
    render(<DeliverablesStudio />)
    expect(screen.getByTestId('gallery')).toBeInTheDocument()
    expect(screen.getByText('Showing all Reports')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Back to feed/ }))
    expect(nav.replace).toHaveBeenCalledWith('/deliverables?tab=outputs')
  })

  it('shows the spinner while the workspace loads', () => {
    ws.isLoading = true
    render(<DeliverablesStudio />)
    expect(screen.getByLabelText('Loading')).toBeInTheDocument()
  })
})
