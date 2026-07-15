/**
 * PRD-203 O·S1 — the intake entry point is live.
 *
 * The welcome modal offers the "Start Business Intake" CTA (previously a
 * commented-out block "hidden for pilot") and its click routes the user to the
 * onboarding wizard. Pure/mocked — router, image, and tour storage are stubbed;
 * no server.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const { pushMock } = vi.hoisted(() => ({ pushMock: vi.fn() }))

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: pushMock }),
}))

vi.mock('next/image', () => ({
  default: (props: Record<string, unknown>) => {
    // eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text
    return <img {...(props as Record<string, string>)} />
  },
}))

const markTourSkipped = vi.fn()
vi.mock('@/lib/shepherd/tour-storage', () => ({
  markTourSkipped: (...args: unknown[]) => markTourSkipped(...args),
}))

import { WelcomeModal } from '../welcome-modal'

describe('WelcomeModal — business intake CTA (O·S1)', () => {
  beforeEach(() => {
    pushMock.mockReset()
    markTourSkipped.mockReset()
  })

  it('renders the Start Business Intake CTA in the welcome modal', () => {
    render(<WelcomeModal open onOpenChange={() => {}} userId="user-1" />)
    expect(screen.getByTestId('business-intake-cta')).toBeInTheDocument()
    expect(screen.getByTestId('business-intake-start')).toHaveTextContent(
      'Start Business Intake'
    )
  })

  it('routes to /onboarding/wizard when the CTA is clicked', async () => {
    render(<WelcomeModal open onOpenChange={() => {}} userId="user-1" />)
    fireEvent.click(screen.getByTestId('business-intake-start'))
    await waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith('/onboarding/wizard')
    })
  })
})
