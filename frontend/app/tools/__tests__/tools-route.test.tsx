/** PRD-244 W5c (two styles) — Studio desktop renders the Studio frame; Classic keeps the classic one; below 1024 px both use Classic. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => state.tabletOrBelow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children, fullBleed }: { children: React.ReactNode; fullBleed?: boolean }) => <div data-fullbleed={String(!!fullBleed)}>{children}</div> }))
vi.mock('@/components/tools/tools-dashboard', () => ({ ToolsDashboard: ({ variant }: { variant?: string }) => <div data-testid="dash" data-variant={variant ?? 'classic'} /> }))

import ToolsPage from '@/app/tools/page'

afterEach(() => { cleanup(); state.tabletOrBelow = false; state.studio = true })

describe('Tools route', () => {
  it('Studio desktop → the Studio frame, full bleed', () => {
    render(<ToolsPage />)
    expect(screen.getByTestId('dash')).toHaveAttribute('data-variant', 'studio')
    expect(screen.getByTestId('dash').parentElement).toHaveAttribute('data-fullbleed', 'true')
  })
  it('Classic desktop → the classic frame', () => {
    state.studio = false
    render(<ToolsPage />)
    expect(screen.getByTestId('dash')).toHaveAttribute('data-variant', 'classic')
  })
  it('below 1024 px → the classic frame in every style', () => {
    state.tabletOrBelow = true
    render(<ToolsPage />)
    expect(screen.getByTestId('dash')).toHaveAttribute('data-variant', 'classic')
  })
})
