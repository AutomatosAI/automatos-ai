/**
 * PRD-244 W1 (D2) — the Command Centre route renders the shell at every desktop
 * width whatever the theme; below 1024 px the classic page remains until the
 * mobile pass (PRD-245, D6).
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const width = vi.hoisted(() => ({ tabletOrBelow: false }))
vi.mock('@/hooks/use-mobile', () => ({
  useIsMobile: () => false,
  useIsTabletOrBelow: () => width.tabletOrBelow,
}))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({
  MainLayout: ({ children }: { children: React.ReactNode }) => <div data-testid="layout">{children}</div>,
}))
vi.mock('@/components/command-center/command-center-shell', () => ({
  CommandCenterShell: () => <div data-testid="shell" />,
}))
vi.mock('@/components/activity/activity-page', () => ({
  ActivityPage: () => <div data-testid="legacy" />,
}))

import CommandCenterPage from '@/app/command-center/page'

afterEach(() => {
  cleanup()
  width.tabletOrBelow = false
})

describe('Command Centre route', () => {
  it('renders the shell on desktop with no theme condition', () => {
    render(<CommandCenterPage />)
    expect(screen.getByTestId('shell')).toBeInTheDocument()
    expect(screen.queryByTestId('legacy')).toBeNull()
  })

  it('keeps the classic page below 1024 px until the mobile pass', () => {
    width.tabletOrBelow = true
    render(<CommandCenterPage />)
    expect(screen.getByTestId('legacy')).toBeInTheDocument()
    expect(screen.queryByTestId('shell')).toBeNull()
  })

  it('no longer forks on the theme', () => {
    const src = readFileSync(path.resolve(__dirname, '..', 'page.tsx'), 'utf8')
    expect(src).not.toContain('useIsStudio')
  })
})
