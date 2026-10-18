/**
 * PRD-261 A-04a (A0 plan §7 Q7) — the page boundary inside a theme's frame: a page that throws
 * draws the core's error UI in its place, with a plain message that never shows the error's text,
 * logs the error, and draws the page again on "Try again".
 */
import { useState } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PAGE_ERROR_TEST_ID, PAGE_ERROR_TEXT, PAGE_ERROR_TITLE, PageBoundary } from '../page-boundary'

const SECRET_DETAIL = 'internal detail that must not reach the screen'

const failing = { current: true }

function FlakyPage() {
  const [label] = useState('page drawn')
  if (failing.current) throw new Error(SECRET_DETAIL)
  return <p>{label}</p>
}

afterEach(() => {
  failing.current = true
  vi.restoreAllMocks()
})

describe('PRD-261 A-04a — PageBoundary', () => {
  it('draws its page when nothing throws', () => {
    failing.current = false
    render(<PageBoundary><FlakyPage /></PageBoundary>)
    expect(screen.getByText('page drawn')).toBeInTheDocument()
    expect(screen.queryByTestId(PAGE_ERROR_TEST_ID)).toBeNull()
  })

  it("draws the core's error UI with a plain message, and logs the error", () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    render(<PageBoundary><FlakyPage /></PageBoundary>)
    const shown = screen.getByTestId(PAGE_ERROR_TEST_ID)
    expect(shown).toHaveTextContent(PAGE_ERROR_TITLE)
    expect(shown).toHaveTextContent(PAGE_ERROR_TEXT)
    expect(shown).not.toHaveTextContent(SECRET_DETAIL)
    expect(error).toHaveBeenCalledWith('[theme-host]', 'page', expect.any(Error))
  })

  it('draws the page again on "Try again"', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    render(<PageBoundary><FlakyPage /></PageBoundary>)
    failing.current = false
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(screen.getByText('page drawn')).toBeInTheDocument()
    expect(screen.queryByTestId(PAGE_ERROR_TEST_ID)).toBeNull()
  })
})
