/**
 * PRD-180 S2 (F038) — the fabricated sidebar stats are gone.
 *
 * Pins: the sidebar renders no hardcoded telemetry literals (tick 5s,
 * $/dec $0.0027, cache 68%) and no fabricated version string (v0.11). Those were
 * made-up numbers, not real metrics; a lie in the chrome corrodes trust in every
 * real control, so they are deleted (not wired to a placeholder).
 */
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

vi.mock('next/navigation', () => ({
  usePathname: () => '/command-center',
}))

import { StudioSidebar } from '../studio-sidebar'

describe('StudioSidebar — fabricated stats removed (F038)', () => {
  it('renders none of the fabricated telemetry literals', () => {
    const { container } = render(<StudioSidebar />)
    const text = container.textContent ?? ''
    // The exact fabricated values that used to render.
    expect(text).not.toContain('$0.0027')
    expect(text).not.toContain('68%')
    expect(text).not.toMatch(/\btick\b/)
    expect(text).not.toContain('$/dec')
  })

  it('renders no fabricated version string', () => {
    const { container } = render(<StudioSidebar />)
    expect(container.textContent ?? '').not.toContain('v0.11')
  })

  it('no longer exposes a showStats prop path (dead prop removed)', async () => {
    // Source-level guard: the dead `showStats` prop and its stats block are gone.
    const fs = await import('node:fs')
    const path = await import('node:path')
    const src = fs.readFileSync(
      path.resolve(__dirname, '../studio-sidebar.tsx'),
      'utf8',
    )
    expect(src).not.toContain('showStats')
    expect(src).not.toContain('sh-mini-stats')
  })
})
