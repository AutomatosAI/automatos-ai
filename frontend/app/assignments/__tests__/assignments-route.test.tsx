/**
 * PRD-246 US-007 — the route forks on the STYLE axis alone: the Studio style
 * renders the Studio surface at every width (it has a compact form now), the
 * Classic style renders the classic one. Width informs layout inside a
 * component, never which component.
 */
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
  it('the Studio style renders the Studio surface at phone width too (PRD-246 US-007)', () => {
    state.tabletOrBelow = true
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('hub')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })
  it('the Studio-designed pages fork on the style, never on width alone (nothing crosses between styles)', () => {
    for (const rel of ['assignments', 'playbooks', 'chat', 'command-center']) {
      const src = readFileSync(path.resolve(__dirname, '..', '..', rel, 'page.tsx'), 'utf8')
      expect(src, rel).toContain('useIsStudio')
      // PRD-246 US-007: the style alone chooses the component — the width term
      // is gone, because every Studio surface now has a compact form.
      expect(src, rel).not.toMatch(/isStudio && !is(TabletOrBelow|MobileLayout)/)
    }
  })
})
