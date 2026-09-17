/** PRD-244 W2 (D3) — the hub is the Assignments page at every desktop width, whatever the theme. */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'

const width = vi.hoisted(() => ({ tabletOrBelow: false }))
vi.mock('@/hooks/use-mobile', () => ({ useIsMobile: () => false, useIsTabletOrBelow: () => width.tabletOrBelow }))
vi.mock('@/hooks/use-page-api', () => ({ usePageAPI: () => {} }))
vi.mock('@/components/layout/main-layout', () => ({ MainLayout: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }))
vi.mock('@/components/assignments/studio/assignments-hub', () => ({ StudioAssignmentsHub: () => <div data-testid="hub" /> }))
vi.mock('@/components/assignments/assignments-page', () => ({ AssignmentsPage: () => <div data-testid="classic" /> }))

import AssignmentsRoute from '@/app/assignments/page'

afterEach(() => { cleanup(); width.tabletOrBelow = false })

describe('Assignments route', () => {
  it('renders the hub on desktop with no theme condition', () => {
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('hub')).toBeInTheDocument()
    expect(screen.queryByTestId('classic')).toBeNull()
  })
  it('keeps the classic page below 1024 px until the mobile pass', () => {
    width.tabletOrBelow = true
    render(<AssignmentsRoute />)
    expect(screen.getByTestId('classic')).toBeInTheDocument()
  })
  it('no route forks on the theme any more', () => {
    for (const rel of ['assignments', 'missions', 'playbooks', 'chat', 'command-center']) {
      expect(readFileSync(path.resolve(__dirname, '..', '..', rel, 'page.tsx'), 'utf8')).not.toContain('useIsStudio')
    }
  })
})
