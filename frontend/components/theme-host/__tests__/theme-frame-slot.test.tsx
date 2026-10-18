/**
 * PRD-261 A-01, A-04a — the frame slot draws the core's chrome unchanged with no theme host or no
 * active frame, nothing while a theme is still loading under the theme layout (so the page never
 * mounts twice), the theme's frame around the page when one is active (with frame@1's props), and
 * the chrome again (telling the host) when the frame throws. A page that throws inside the frame
 * draws the core's error UI there, keeps the frame, and never tells the host.
 */
import React from 'react'
import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { FrameComponent, FrameProps, ThemeHostStatus } from '@/lib/theme-host/types'
import type { ThemeLayout } from '@/lib/theme-host/registry-types'

const layout = vi.hoisted(() => ({ current: 'fallback' as ThemeLayout }))
vi.mock('@/hooks/use-studio-theme', () => ({ useThemeLayout: () => layout.current }))

import { PAGE_ERROR_TEST_ID } from '../page-boundary'
import { ThemeFrameSlot } from '../theme-frame-slot'
import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'

const FAILURE = new Error('deliberate frame failure')
const PAGE_FAILURE = new Error('deliberate page failure')

function HelloFrame({ children, fullBleed, autoWidget }: FrameProps) {
  return (
    <section data-testid="theme-frame" data-full-bleed={String(fullBleed)}>
      {children}
      {autoWidget}
    </section>
  )
}

function BrokenFrame(): React.ReactElement {
  throw FAILURE
}

function BrokenPage(): React.ReactElement {
  throw PAGE_FAILURE
}

function hostWith(frame: FrameComponent | null, reportRenderFailure = vi.fn(), status?: ThemeHostStatus): ThemeHostContextValue {
  return { status: status ?? (frame ? 'active' : 'failed'), reason: frame ? null : 'load', frame, reportRenderFailure }
}

function renderSlot(host: ThemeHostContextValue | null, page: React.ReactNode = <p>page</p>) {
  const slot = (
    <ThemeFrameSlot fallback={<div data-testid="core-shell"><p>page</p></div>} fullBleed autoWidget={<aside data-testid="auto-widget" />}>
      {page}
    </ThemeFrameSlot>
  )
  return render(host === null ? slot : <ThemeHostContext.Provider value={host}>{slot}</ThemeHostContext.Provider>)
}

beforeEach(() => {
  layout.current = 'fallback'
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('PRD-261 A-04a — ThemeFrameSlot', () => {
  it('draws the fallback unchanged with no theme host', () => {
    const { container } = renderSlot(null)
    expect(container.innerHTML).toBe('<div data-testid="core-shell"><p>page</p></div>')
  })

  it('draws the fallback when the host has no active frame', () => {
    renderSlot(hostWith(null))
    expect(screen.getByTestId('core-shell')).toBeInTheDocument()
    expect(screen.queryByTestId('theme-frame')).toBeNull()
  })

  it('draws nothing while the theme loads under the theme layout', () => {
    layout.current = 'theme'
    const { container } = renderSlot(hostWith(null, vi.fn(), 'loading'))
    expect(container.innerHTML).toBe('')
  })

  it('draws the fallback while the theme loads under the fallback layout', () => {
    layout.current = 'fallback'
    renderSlot(hostWith(null, vi.fn(), 'loading'))
    expect(screen.getByTestId('core-shell')).toBeInTheDocument()
  })

  it("draws the active theme's frame around the page, with frame@1's props, without the core's chrome", () => {
    layout.current = 'theme'
    renderSlot(hostWith(HelloFrame))
    expect(screen.getByTestId('theme-frame')).toHaveTextContent('page')
    expect(screen.getByTestId('theme-frame')).toHaveAttribute('data-full-bleed', 'true')
    expect(screen.getByTestId('auto-widget')).toBeInTheDocument()
    expect(screen.queryByTestId('core-shell')).toBeNull()
  })

  it('draws the fallback and tells the host when the frame throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const reportRenderFailure = vi.fn()
    renderSlot(hostWith(BrokenFrame, reportRenderFailure))
    expect(screen.getByTestId('core-shell')).toBeInTheDocument()
    expect(reportRenderFailure).toHaveBeenCalledWith(FAILURE)
  })

  it("keeps the frame and shows the core's error UI when the page throws, and never tells the host", () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const reportRenderFailure = vi.fn()
    renderSlot(hostWith(HelloFrame, reportRenderFailure), <BrokenPage />)
    expect(screen.getByTestId('theme-frame')).toBeInTheDocument()
    expect(screen.getByTestId('theme-frame')).toContainElement(screen.getByTestId(PAGE_ERROR_TEST_ID))
    expect(screen.queryByTestId('core-shell')).toBeNull()
    expect(reportRenderFailure).not.toHaveBeenCalled()
  })
})
