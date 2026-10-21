/**
 * PRD-261 W-L1 (A-19e2's finding; P260-FX-25, M69) — useIsTabletOrBelow knows the width on a
 * browser mount's first render. It used to start false and flip in an effect, so on a phone every
 * client navigation drew MainLayout's desktop branch for one render, where the theme host hands
 * an installed theme's frame over (components/theme-host/__tests__/theme-frame-below-min-width
 * proves the frame never renders there). The server and hydration read false, as before: the
 * server's HTML is unchanged and hydrating it on a phone is no mismatch.
 */
import { act, render, screen } from '@testing-library/react'
import { renderToString } from 'react-dom/server'
import { hydrateRoot, type Root } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useIsTabletOrBelow } from '@/hooks/use-mobile'

import { LAPTOP_WIDTH_PX, PHONE_WIDTH_PX, stubWindowWidth, type WindowWidth } from './window-width'

const NARROW = 'tablet or below'
const WIDE = 'wider'
/** Just below and at the 1024 px band (hooks/use-mobile.ts TABLET_BREAKPOINT). */
const BELOW_BAND_PX = 1023
const AT_BAND_PX = 1024

/** Records every answer useIsTabletOrBelow gives, render by render. */
function BandProbe({ seen }: { readonly seen: boolean[] }) {
  const below = useIsTabletOrBelow()
  seen.push(below)
  return <p data-testid="band">{below ? NARROW : WIDE}</p>
}

let width: WindowWidth | null = null

afterEach(() => {
  width?.restore()
  width = null
})

describe('useIsTabletOrBelow in the browser', () => {
  it('is true on a phone from the very first render, never false for a render', () => {
    width = stubWindowWidth(PHONE_WIDTH_PX)
    const seen: boolean[] = []
    render(<BandProbe seen={seen} />)
    expect(screen.getByTestId('band')).toHaveTextContent(NARROW)
    expect(seen.length).toBeGreaterThan(0)
    expect(new Set(seen)).toEqual(new Set([true]))
  })

  it('is false on a laptop from the very first render', () => {
    width = stubWindowWidth(LAPTOP_WIDTH_PX)
    const seen: boolean[] = []
    render(<BandProbe seen={seen} />)
    expect(new Set(seen)).toEqual(new Set([false]))
  })

  it('follows the window across 1024 px', () => {
    width = stubWindowWidth(BELOW_BAND_PX)
    render(<BandProbe seen={[]} />)
    expect(screen.getByTestId('band')).toHaveTextContent(NARROW)
    act(() => width?.resize(AT_BAND_PX))
    expect(screen.getByTestId('band')).toHaveTextContent(WIDE)
    act(() => width?.resize(PHONE_WIDTH_PX))
    expect(screen.getByTestId('band')).toHaveTextContent(NARROW)
  })

  it('treats a browser without matchMedia as a wide one', () => {
    width = stubWindowWidth(PHONE_WIDTH_PX)
    Reflect.deleteProperty(window, 'matchMedia')
    render(<BandProbe seen={[]} />)
    expect(screen.getByTestId('band')).toHaveTextContent(WIDE)
  })
})

describe('useIsTabletOrBelow on the server and while hydrating', () => {
  it('is false on the server, as the HTML has always been drawn', () => {
    const seen: boolean[] = []
    expect(renderToString(<BandProbe seen={seen} />)).toContain(WIDE)
    expect(seen).toEqual([false])
  })

  it("hydrates the server's HTML on a phone without a mismatch, then reads the phone", async () => {
    const container = document.createElement('div')
    container.innerHTML = renderToString(<BandProbe seen={[]} />)
    document.body.appendChild(container)
    width = stubWindowWidth(PHONE_WIDTH_PX)
    const seen: boolean[] = []
    const onRecoverableError = vi.fn()
    const hydrated: { root: Root | null } = { root: null }
    await act(async () => {
      hydrated.root = hydrateRoot(container, <BandProbe seen={seen} />, { onRecoverableError })
    })
    try {
      expect(seen[0]).toBe(false)
      expect(seen[seen.length - 1]).toBe(true)
      expect(container).toHaveTextContent(NARROW)
      expect(onRecoverableError).not.toHaveBeenCalled()
    } finally {
      act(() => hydrated.root?.unmount())
      container.remove()
    }
  })
})
