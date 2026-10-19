/**
 * PRD-261 A-04a — helpers for the boundary tests: a page that calls Next's notFound() or
 * redirect() while drawing (the real functions, so the errors carry Next's own digests), and a
 * boundary above the one under test that stands in for Next's, recording what reaches it.
 */
import React from 'react'
import { notFound, redirect } from 'next/navigation'

export const REDIRECT_TARGET = '/agents'
export const OUTER_CAUGHT_TEST_ID = 'outer-boundary-caught'

export function CallsNotFound(): React.ReactElement {
  return notFound()
}

export function CallsRedirect(): React.ReactElement {
  return redirect(REDIRECT_TARGET)
}

/** The router errors a page can throw, by name, for it.each. */
export const ROUTER_ERROR_PAGES: Array<[string, () => React.ReactElement]> = [
  ['notFound()', CallsNotFound],
  ['redirect()', CallsRedirect],
]

interface OuterProps {
  children: React.ReactNode
  onCaught: (error: unknown) => void
}

/** Stands in for Next's boundary above the theme's: whatever is rethrown lands here. */
export class OuterBoundary extends React.Component<OuterProps, { caught: boolean }> {
  constructor(props: OuterProps) {
    super(props)
    this.state = { caught: false }
  }

  static getDerivedStateFromError(): { caught: boolean } {
    return { caught: true }
  }

  componentDidCatch(error: unknown) {
    this.props.onCaught(error)
  }

  render() {
    return this.state.caught ? <p data-testid={OUTER_CAUGHT_TEST_ID}>caught above</p> : this.props.children
  }
}
