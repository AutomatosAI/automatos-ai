/** PRD-244 (two styles, two tones) — Studio desktop renders the hub; Classic keeps the AssignmentsPage; below 1024 px both use the AssignmentsPage. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const state = vi.hoisted(() => ({ tabletOrBelow: false, studio: true }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => state.tabletOrBelow }))
vi.mock('@/hooks/use-studio-theme', () => ({ useIsStudio: () => state.studio }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/assignments/studio/assignments-hub', () => ({ StudioAssignmentsHub: () => <div data-testid="hub" /> }))
vi.mock('@/components/assignments/assignments-page', () => ({ AssignmentsPage: () => <div data-testid="classic" /> }))

import AssignmentsRoute from '@/app/assignments/page'

afterEach(() => { cleanup(); state.tabletOrBelow = false; state.studio = true })

describe('Assignments route', () => {
  it('Studio style on desktop renders the hub', () => {
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('hub')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })
  it('Classic style on desktop renders the classic page, never the hub', () => {
    state.studio = false
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
    expect(screen.queryByTestId('hub')).toBeNull()
  })
  it('below 1024 px both styles render the classic page', () => {
    state.tabletOrBelow = true
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
  })
  it('the Studio-designed pages fork on the style, never on width alone (nothing crosses between styles)', () => {
    for (const rel of ['assignments', 'playbooks', 'chat', 'command-center']) {
      const src = readFileSync(path.resolve(__dirname, '..', '..', rel, 'page.tsx'), 'utf8')
      expect(src, rel).toContain('useIsStudio')
      expect(src, rel).toMatch(/isStudio && !is(TabletOrBelow|MobileLayout)/)
    }
  })
})
