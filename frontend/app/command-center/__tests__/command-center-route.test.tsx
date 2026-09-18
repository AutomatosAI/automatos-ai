/**
 * PRD-246 US-007 — the route forks on the STYLE axis alone: the Studio style
 * renders the Studio surface at every width (it has a compact form now), the
 * Classic style renders the classic one. Width informs layout inside a
 * component, never which component.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => false,
  useIsTabletOrBelow: () => state.tabletOrBelow,
}))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="layout">{children}</div>,
}))
vi.mock('@/components/command-center/command-center-shell', () => ({
  CommandCenterShell: () => <div data-testid="shell" />,
}))
vi.mock('@/components/activity/activity-page', () => ({
  ActivityPage: () => <div data-testid="classic" />,
}))

import CommandCenterPage from '@/app/command-center/page'

afterEach(() => {
  cleanup()
  state.tabletOrBelow = false
  state.studio = true
})

describe('Command Centre route', () => {
  it('Studio style on desktop renders the shell', () => {
    render(<CommandCenterPage />)
    expect(screen.getByTestId('shell')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })

  it('Classic style on desktop renders the classic page, never the shell', () => {
    state.studio = false
    render(<CommandCenterPage />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
    expect(screen.queryByTestId('shell')).toBeNull()
  })

  it('the Studio style renders the Studio surface at phone width too (PRD-246 US-007)', () => {
    state.tabletOrBelow = true
    render(<CommandCenterPage />)
    expect(screen.getByTestId('shell')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })
})
