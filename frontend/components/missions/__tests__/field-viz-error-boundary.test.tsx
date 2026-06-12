import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { FieldVizErrorBoundary } from '../field-viz-error-boundary'

function Boom(): JSX.Element {
  throw new Error('WebGL context lost')
}

describe('FieldVizErrorBoundary', () => {
  it('falls back to the 2D renderer when the 3D child throws', () => {
    // React logs caught render errors to console.error — silence the noise.
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <FieldVizErrorBoundary fallback={<div>FALLBACK_2D</div>}>
        <Boom />
      </FieldVizErrorBoundary>,
    )
    expect(screen.getByText('FALLBACK_2D')).toBeInTheDocument()
    spy.mockRestore()
  })

  it('renders the primary child when nothing throws', () => {
    render(
      <FieldVizErrorBoundary fallback={<div>FALLBACK_2D</div>}>
        <div>PRIMARY_3D</div>
      </FieldVizErrorBoundary>,
    )
    expect(screen.getByText('PRIMARY_3D')).toBeInTheDocument()
    expect(screen.queryByText('FALLBACK_2D')).not.toBeInTheDocument()
  })
})
