'use client'

import React from 'react'

interface Props {
  children: React.ReactNode
  /** Rendered when a child throws — for the field viz this is the 2D renderer. */
  fallback: React.ReactNode
}

interface State {
  hasError: boolean
}

/**
 * Catches a render/runtime throw from the primary (3D) field renderer and
 * swaps in the `fallback` (the 2D renderer). The old viz mounted a raw
 * <Canvas> from @react-three/fiber v9, which peer-requires React 19 while the
 * app is on React 18 — it crashed at mount with NO boundary, so the Field tab
 * was a blank panel. This makes the 3D→2D degrade graceful.
 */
export class FieldVizErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: unknown) {
    // eslint-disable-next-line no-console
    console.error('[MissionField] 3D renderer failed, falling back to 2D:', error)
  }

  render() {
    return this.state.hasError ? this.props.fallback : this.props.children
  }
}
