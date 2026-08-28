/**
 * PRD-222 US-020 — the post-setup checklist card.
 *
 * Renders the server read-model on chat + Command Center at the powerup/completed
 * stages, dismissible, with dismissal persisted server-side (a PATCH, never
 * localStorage — D8). The manual Academy item checks itself on click. Pure/mocked
 * — the workspace provider + checklist hooks are stubbed.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

const { useWorkspaceMock, useChecklistMock, mutateMock } = vi.hoisted(() => ({
  useWorkspaceMock: vi.fn(),
  useChecklistMock: vi.fn(),
  mutateMock: vi.fn(),
}))

vi.mock('@/components/workspace-provider', () => ({ useWorkspace: () => useWorkspaceMock() }))
vi.mock('@/hooks/use-onboarding-checklist', () => ({
  useOnboardingChecklist: () => useChecklistMock(),
  useUpdateChecklist: () => ({ mutate: mutateMock }),
}))

import { SetupChecklistCard } from '../setup-checklist-card'

const CHECKLIST = {
  items: [
    { id: 'connect_second_app', label: 'Connect a second app', done: true },
    { id: 'run_first_mission', label: 'Run your first mission', done: false },
    { id: 'invite_teammate', label: 'Invite a teammate', done: false },
    {
      id: 'take_course',
      label: 'Take the matched Academy course',
      done: false,
      href: 'https://academy.automatos.app/abf',
      manual: true,
    },
  ],
  dismissed: false,
  completed_count: 1,
  total_count: 4,
}

function setup(stage: string, checklist: typeof CHECKLIST | null = CHECKLIST) {
  useWorkspaceMock.mockReturnValue({ workspace: { onboarding: { stage } } })
  useChecklistMock.mockReturnValue({ data: checklist })
}

describe('SetupChecklistCard (US-020)', () => {
  beforeEach(() => {
    useWorkspaceMock.mockReset()
    useChecklistMock.mockReset()
    mutateMock.mockReset()
    localStorage.clear()
  })

  it('renders the items on the completed stage', () => {
    setup('completed')
    render(<SetupChecklistCard />)
    expect(screen.getByTestId('setup-checklist-card')).toBeInTheDocument()
    expect(screen.getByTestId('setup-checklist-item-connect_second_app')).toHaveAttribute('data-done', 'true')
    expect(screen.getByTestId('setup-checklist-item-run_first_mission')).toHaveAttribute('data-done', 'false')
    expect(screen.getByTestId('setup-checklist-progress')).toHaveTextContent('1 of 4 done')
  })

  it('renders at the powerup stage too (Command Center)', () => {
    setup('powerup')
    render(<SetupChecklistCard />)
    expect(screen.getByTestId('setup-checklist-card')).toBeInTheDocument()
  })

  it('does not render before the powerup stage', () => {
    setup('building')
    const { container } = render(<SetupChecklistCard />)
    expect(container).toBeEmptyDOMElement()
  })

  it('does not render once dismissed (server flag)', () => {
    setup('completed', { ...CHECKLIST, dismissed: true })
    const { container } = render(<SetupChecklistCard />)
    expect(container).toBeEmptyDOMElement()
  })

  it('dismiss persists server-side (PATCH), not to localStorage', () => {
    setup('completed')
    render(<SetupChecklistCard />)
    fireEvent.click(screen.getByTestId('setup-checklist-dismiss'))
    expect(mutateMock).toHaveBeenCalledWith({ dismissed: true })
    expect(localStorage.length).toBe(0)
  })

  it('the Academy item links out and checks itself on click', () => {
    setup('completed')
    render(<SetupChecklistCard />)
    const link = screen.getByTestId('setup-checklist-link-take_course')
    expect(link).toHaveAttribute('href', 'https://academy.automatos.app/abf')
    expect(link).toHaveAttribute('target', '_blank')
    fireEvent.click(link)
    expect(mutateMock).toHaveBeenCalledWith({ academy_done: true })
  })

  it('renders the dual-surface positioning className', () => {
    setup('completed')
    render(<SetupChecklistCard className="my-3" />)
    expect(screen.getByTestId('setup-checklist-card').className).toContain('my-3')
  })
})
