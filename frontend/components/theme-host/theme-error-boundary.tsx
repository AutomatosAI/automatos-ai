'use client'

import { unstable_rethrow } from 'next/navigation'
import React from 'react'

interface Props {
  children: React.ReactNode
  /** Rendered once a child throws: the core's own chrome. */
  fallback: React.ReactNode
  /** Told about the error once, so the host can drop the theme. */
  onError: (error: unknown) => void
}

interface State {
  hasError: boolean
}

/**
 * PRD-261 A-01 — catches a render throw from a theme's frame and draws the fallback instead (on
 * the pattern of components/missions/field-viz-error-boundary.tsx, which only logs). The host's
 * onError takes the theme down, so the fallback stays. Next's own control-flow errors (notFound(),
 * redirect(), a bailout to client rendering) are rethrown to Next's boundaries above, as Next's own
 * error boundary does: they never take the theme down.
 */
export class ThemeErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: unknown): State {
    unstable_rethrow(error)
    return { hasError: true }
  }

  componentDidCatch(error: unknown) {
    this.props.onError(error)
  }

  render() {
    return this.state.hasError ? this.props.fallback : this.props.children
  }
}
