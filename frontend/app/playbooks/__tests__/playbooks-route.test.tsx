/** PRD-244 W2 — /playbooks forwards to the Assignments Playbooks tab on desktop (every theme); the standalone panel stays below 1024 px. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'

const nav = vi.hoisted(() => ({ replace: vi.fn(), search: 'id=recipe-7' }))
const width = vi.hoisted(() => ({ tabletOrBelow: false }))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ replace: nav.replace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(nav.search),
}))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => width.tabletOrBelow }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/playbooks/PlaybooksPanel', () => ({ PlaybooksPanel: () => <div data-testid="panel" /> }))

import PlaybooksPage from '@/app/playbooks/page'

afterEach(() => { cleanup(); nav.replace.mockClear(); width.tabletOrBelow = false })

describe('/playbooks', () => {
  it('forwards to the hub on desktop with the params kept', async () => {
    render(<PlaybooksPage />)
    await waitFor(() => expect(nav.replace).toHaveBeenCalled())
    expect(String(nav.replace.mock.calls[0][0])).toBe('/assignments?id=recipe-7&tab=playbooks')
    expect(screen.queryByTestId('panel')).toBeNull()
  })
  it('keeps the standalone panel below 1024 px', () => {
    width.tabletOrBelow = true
    render(<PlaybooksPage />)
    expect(screen.getByTestId('panel')).toBeInTheDocument()
    expect(nav.replace).not.toHaveBeenCalled()
  })
})
