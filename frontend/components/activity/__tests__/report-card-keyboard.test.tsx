/**
 * Issue #1090 — a report card opens from the keyboard, and the keys of its own
 * Download and View buttons stay those buttons' own.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'
import type { AgentReport } from '@/hooks/use-reports-api'
import { ReportCard } from '../report-card'

const REPORT = {
  id: 'r-7f3a', agent_name: 'Words', report_type: 'standup', title: 'Supplier terms',
  summary: 'Terms agreed.', status: 'ok', metrics: {}, grade: null, graded_at: null, created_at: '',
} as unknown as AgentReport

afterEach(() => cleanup())

describe('ReportCard keyboard', () => {
  it('opens with Enter and Space like a click, and leaves its buttons alone', () => {
    const onView = vi.fn()
    render(<ReportCard report={REPORT} onView={onView} />)
    const card = screen.getByText('Supplier terms').closest('[role="button"]') as HTMLElement
    expect(card).toHaveAttribute('tabindex', '0')

    fireEvent.click(card)
    expect(onView).toHaveBeenCalledWith(REPORT)

    onView.mockClear()
    fireEvent.keyDown(card, { key: 'Enter' })
    expect(onView).toHaveBeenCalledWith(REPORT)

    onView.mockClear()
    fireEvent.keyDown(card, { key: ' ' })
    expect(onView).toHaveBeenCalledWith(REPORT)

    onView.mockClear()
    fireEvent.keyDown(screen.getByRole('button', { name: 'View' }), { key: 'Enter' })
    expect(onView).not.toHaveBeenCalled()
  })
})
