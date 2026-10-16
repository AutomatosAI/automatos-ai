/**
 * PRD-261 A-01 — a frame that throws while drawing is replaced by the fallback, and the host is
 * told once, with the error.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ThemeErrorBoundary } from '../theme-error-boundary'

const FAILURE = new Error('deliberate render failure')

function Throws(): React.ReactElement {
  throw FAILURE
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PRD-261 A-01 — ThemeErrorBoundary', () => {
  it('renders its children while they draw', () => {
    render(
      <ThemeErrorBoundary fallback={<p>fallback</p>} onError={vi.fn()}>
        <p>frame</p>
      </ThemeErrorBoundary>,
    )
    expect(screen.getByText('frame')).toBeInTheDocument()
    expect(screen.queryByText('fallback')).toBeNull()
  })

  it('renders the fallback when a child throws, and calls onError once with the error', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const onError = vi.fn()
    render(
      <ThemeErrorBoundary fallback={<p>fallback</p>} onError={onError}>
        <Throws />
      </ThemeErrorBoundary>,
    )
    expect(screen.getByText('fallback')).toBeInTheDocument()
    expect(onError).toHaveBeenCalledTimes(1)
    expect(onError).toHaveBeenCalledWith(FAILURE)
  })
})
