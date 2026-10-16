/**
 * PRD-261 A-01 — the frame slot draws the core's chrome unchanged with no theme host or no active
 * frame, the theme's frame around the page when one is active, and the chrome again (telling the
 * host) when the frame throws.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ThemeFrameSlot } from '../theme-frame-slot'
import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'
import type { FrameComponent } from '@/lib/theme-host/types'

const FAILURE = new Error('deliberate frame failure')

function HelloFrame({ children }: { children?: React.ReactNode }) {
  return <section data-testid="theme-frame">{children}</section>
}

function BrokenFrame(): React.ReactElement {
  throw FAILURE
}

function hostWith(frame: FrameComponent | null, reportRenderFailure = vi.fn()): ThemeHostContextValue {
  return { status: frame ? 'active' : 'failed', reason: frame ? null : 'load', frame, reportRenderFailure }
}

function renderSlot(host: ThemeHostContextValue | null) {
  const slot = (
    <ThemeFrameSlot fallback={<div data-testid="core-shell"><p>page</p></div>}>
      <p>page</p>
    </ThemeFrameSlot>
  )
  return render(host === null ? slot : <ThemeHostContext.Provider value={host}>{slot}</ThemeHostContext.Provider>)
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PRD-261 A-01 — ThemeFrameSlot', () => {
  it('draws the fallback unchanged with no theme host', () => {
    const { container } = renderSlot(null)
    expect(container.innerHTML).toBe('<div data-testid="core-shell"><p>page</p></div>')
  })

  it('draws the fallback when the host has no active frame', () => {
    renderSlot(hostWith(null))
    expect(screen.getByTestId('core-shell')).toBeInTheDocument()
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })

  it("draws the active theme's frame around the page, without the core's chrome", () => {
    renderSlot(hostWith(HelloFrame))
    expect(screen.getByTestId('theme-frame')).toHaveTextContent('page')
    expect(screen.queryByTestId('core-shell')).toBeNull()
  })

  it('draws the fallback and tells the host when the frame throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const reportRenderFailure = vi.fn()
    renderSlot(hostWith(BrokenFrame, reportRenderFailure))
    expect(screen.getByTestId('core-shell')).toBeInTheDocument()
    expect(reportRenderFailure).toHaveBeenCalledWith(FAILURE)
  })
})
