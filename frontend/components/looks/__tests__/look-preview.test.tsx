/**
 * PRD-261 A-13b (F10, WCAG 2.2.2) — the Looks page's preview Auto is the one Auto on /looks: his
 * clip plays and his ring pulses, and nothing else on the page stops him. So he keeps his own Pause
 * Auto control; a later pauseControl={false} here would leave the page moving with no way to stop it.
 * jsdom plays no media: play() is a mock that resolves, as in the mascot tests.
 */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { LookPreview, PREVIEW_CAPTION } from '@/components/looks/look-preview'
import { PAUSE_AUTO } from '@/components/mascot/mascot-pause-button'
import { PAUSE_KEY } from '@/lib/mascot/pause'

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
})

afterEach(() => {
  cleanup()
  window.localStorage.removeItem(PAUSE_KEY)
  vi.restoreAllMocks()
})

describe('LookPreview — Auto can be paused where he moves (WCAG 2.2.2)', () => {
  it('offers the Pause Auto control under the preview Auto', () => {
    render(<LookPreview />)
    expect(screen.getByRole('img', { name: PREVIEW_CAPTION })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: PAUSE_AUTO })).toHaveAttribute('aria-pressed', 'false')
  })
})
