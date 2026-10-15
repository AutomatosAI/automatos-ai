/**
 * PRD-260 P260-FX-05 (M41, review #177) — the running step keeps its focus ring and hover.
 *
 * The step rows' `dark:bg-blue-950/20` was dead until PRD-260's darkMode cherry-pick made
 * every dark: class render. Tailwind emits the dark variant after hover:/focus: at the same
 * specificity, so in every dark tone it replaced the running row's hover and its only
 * keyboard-focus indicator (`focus:bg-muted/50`). The row now marks the running step with
 * the info token in every tone, and shows focus as a ring.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { WorkflowSteps } from '../WorkflowSteps'
import type { WorkflowStep } from '../../types'

const RUNNING: WorkflowStep = { id: 'fetch', name: 'Fetch the orders', type: 'action', status: 'running', duration: 1200 }
const FAILED: WorkflowStep = {
  id: 'send',
  name: 'Send the summary',
  type: 'action',
  status: 'failed',
  duration: 300,
  error: 'SMTP refused the message',
}

beforeEach(() => {
  // jsdom has no ResizeObserver; the list's Radix scroll area observes its viewport.
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('WorkflowSteps rows (P260-FX-05)', () => {
  it("a running step's row keeps a focus ring and its hover, with no dark: background over them", () => {
    render(<WorkflowSteps steps={[RUNNING, FAILED]} />)
    const row = screen.getByRole('button', { name: /Fetch the orders/ })
    expect(row.className).toContain('focus-visible:ring-2')
    expect(row.className).toContain('focus-visible:ring-ring')
    expect(row.className).toContain('hover:bg-muted/50')
    expect(row.className).toContain('bg-info/10')
    expect(row.className).not.toMatch(/(^|\s)dark:bg-/)
    expect(row).toHaveTextContent('Currently running')
  })

  it('a step that is not running carries no running tint', () => {
    render(<WorkflowSteps steps={[RUNNING, FAILED]} />)
    const row = screen.getByRole('button', { name: /Send the summary/ })
    expect(row.className).not.toContain('bg-info/10')
    expect(row.className).toContain('focus-visible:ring-2')
    expect(row).toHaveTextContent('SMTP refused the message')
  })

  it("clicking a row still opens that step's detail, its error in foreground text in dark tones", () => {
    render(<WorkflowSteps steps={[RUNNING, FAILED]} />)
    fireEvent.click(screen.getByRole('button', { name: /Send the summary/ }))
    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveTextContent('Send the summary')
    const heading = screen.getByRole('heading', { name: 'Error' })
    expect(heading.className).toContain('dark:text-foreground')
    expect(heading.className).not.toMatch(/dark:text-destructive\//)
  })
})
