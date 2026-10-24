/**
 * PRD-266 US-019c (§4.3 "Web", K17): the QR image in "Pair a phone".
 *
 * The pairing link is drawn as an SVG from the vendored encoder's module matrix
 * (lib/vendor/qrcodegen, Nayuki's v1.8.0). The level is M exactly, the quiet zone is four modules,
 * and dark modules sit on a light ground. Each dark module the SVG draws is one the encoder sets
 * for that text, and it draws all of them. Nothing is drawn on SaaS, and nothing once the code has
 * expired. The code, its countdown and the link stay beside the image as text.
 */
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { QrCode } from '@/lib/vendor/qrcodegen/qrcodegen'

const CODE = 'ABCD-EFGH'
const LINK = `https://studio.example.com/go/pair?api=https%3A%2F%2Fapi.example.com&code=${CODE}`
/** Byte mode at level M holds 2,331 bytes at most (version 40). */
const TOO_LONG = `${LINK}&pad=${'x'.repeat(3000)}`
const TEN_MINUTES_S = 600
const FINDER_SIDE = 7
const FINDER_RING = 2
const NO_OP = () => undefined
/** Let the encoder choose the mask, as the component does. */
const AUTO_MASK = -1

type Edition = 'local' | 'saas'

/** The QR module and the vendored encoder in `edition`, fresh from import, in one module registry. */
async function load(edition: Edition) {
  vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', edition)
  vi.resetModules()
  const qrModule = await import('../pair-phone-qr')
  const encoder = await import('@/lib/vendor/qrcodegen/qrcodegen')
  const segments = await import('@/lib/vendor/qrcodegen/qrsegment')
  return { ...qrModule, QrCode: encoder.QrCode, QrSegment: segments.QrSegment }
}

/** The dark modules the SVG's path draws, as "x,y" in the symbol (the quiet zone taken off). */
function drawnModules(svg: Element, quietZone: number): Set<string> {
  const path = svg.querySelector('path')?.getAttribute('d') ?? ''
  const squares = [...path.matchAll(/M(\d+),(\d+)h1v1h-1z/g)]
  return new Set(squares.map(([, x, y]) => `${Number(x) - quietZone},${Number(y) - quietZone}`))
}

/** The encoder's dark modules for `qr`, as "x,y". */
function encoderModules(qr: QrCode): Set<string> {
  const indices = Array.from({ length: qr.size }, (_, index) => index)
  return new Set(indices.flatMap((y) => indices.filter((x) => qr.getModule(x, y)).map((x) => `${x},${y}`)))
}

/** A finder pattern's 7×7 modules from (left, top): dark except the ring two modules in from its edge. */
function finderAt(left: number, top: number): Array<{ readonly at: string; readonly dark: boolean }> {
  const offsets = Array.from({ length: FINDER_SIDE }, (_, index) => index)
  const centre = (FINDER_SIDE - 1) / 2
  return offsets.flatMap((dy) => offsets.map((dx) => ({
    at: `${left + dx},${top + dy}`,
    dark: Math.max(Math.abs(dx - centre), Math.abs(dy - centre)) !== FINDER_RING,
  })))
}

afterEach(() => {
  cleanup()
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
})

describe('PairPhoneQr draws the pairing link', () => {
  it('as an SVG at level M, inside a four-module quiet zone, dark modules on a light ground', async () => {
    const { PairPhoneQr, encodePairingQr, QrCode: Encoder, QUIET_ZONE_MODULES, QR_DARK, QR_LIGHT, QR_LABEL } = await load('local')
    render(<PairPhoneQr payload={LINK} />)

    const svg = screen.getByRole('img', { name: QR_LABEL })
    const qr = encodePairingQr(LINK) as QrCode
    const side = qr.size + 2 * QUIET_ZONE_MODULES
    expect(QUIET_ZONE_MODULES).toBe(4)
    expect(qr.errorCorrectionLevel.ordinal).toBe(Encoder.Ecc.MEDIUM.ordinal)
    expect(svg.getAttribute('viewBox')).toBe(`0 0 ${side} ${side}`)
    const ground = svg.querySelector('rect')
    expect(ground?.getAttribute('width')).toBe(String(side))
    expect(ground?.getAttribute('height')).toBe(String(side))
    expect(ground?.getAttribute('fill')).toBe(QR_LIGHT)
    expect(svg.querySelector('path')?.getAttribute('fill')).toBe(QR_DARK)
  })

  it("with the encoder's symbol for that text: the same size and every dark module, no more", async () => {
    const { PairPhoneQr, QrCode: Encoder, QrSegment, QUIET_ZONE_MODULES, QR_LABEL } = await load('local')
    const direct = Encoder.encodeSegments(QrSegment.makeSegments(LINK), Encoder.Ecc.MEDIUM, Encoder.MIN_VERSION, Encoder.MAX_VERSION, AUTO_MASK, false)
    render(<PairPhoneQr payload={LINK} />)

    const svg = screen.getByRole('img', { name: QR_LABEL })
    const drawn = drawnModules(svg, QUIET_ZONE_MODULES)
    const expected = encoderModules(direct)
    expect(svg.getAttribute('viewBox')).toBe(`0 0 ${direct.size + 2 * QUIET_ZONE_MODULES} ${direct.size + 2 * QUIET_ZONE_MODULES}`)
    expect(expected.size).toBeGreaterThan(0)
    expect(drawn.size).toBe(expected.size)
    expect(drawn).toEqual(expected)
  })

  it('the right way round: a finder pattern in the top-left, top-right and bottom-left corners', async () => {
    const { PairPhoneQr, encodePairingQr, QUIET_ZONE_MODULES, QR_LABEL } = await load('local')
    render(<PairPhoneQr payload={LINK} />)

    const drawn = drawnModules(screen.getByRole('img', { name: QR_LABEL }), QUIET_ZONE_MODULES)
    const far = (encodePairingQr(LINK) as QrCode).size - FINDER_SIDE
    for (const [left, top] of [[0, 0], [far, 0], [0, far]]) {
      for (const cell of finderAt(left, top)) expect(drawn.has(cell.at)).toBe(cell.dark)
    }
  })

  it('draws nothing for a link too long for any version, and never logs the link (it holds the code)', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(NO_OP)
    const { PairPhoneQr, encodePairingQr } = await load('local')
    expect(encodePairingQr(TOO_LONG)).toBeNull()

    const { container } = render(<PairPhoneQr payload={TOO_LONG} />)
    expect(container.querySelector('svg')).toBeNull()
    expect(warn).toHaveBeenCalled()
    expect(JSON.stringify(warn.mock.calls)).not.toContain(CODE)
  })

  it('draws nothing on SaaS, where nothing pairs', async () => {
    const { PairPhoneQr } = await load('saas')
    const { container } = render(<PairPhoneQr payload={LINK} />)
    expect(container.querySelector('svg')).toBeNull()
  })
})

describe('Pair a phone with the QR image', () => {
  const pairing = { code: CODE, expires_at: new Date(Date.now() + TEN_MINUTES_S * 1000).toISOString(), qr_payload: LINK }

  async function renderPanel(secondsLeft: number) {
    vi.stubEnv('NEXT_PUBLIC_AUTH_EDITION', 'local')
    vi.resetModules()
    const { PairPhonePanel, CODE_EXPIRED } = await import('../pair-phone-panel')
    render(<PairPhonePanel pairing={pairing} secondsLeft={secondsLeft} minting={false} error={null} onMint={NO_OP} />)
    return { CODE_EXPIRED }
  }

  it('shows it while the code is live, with the code, the countdown and the link beside it as text', async () => {
    await renderPanel(TEN_MINUTES_S)

    const qr = screen.getByTestId('pairing-qr')
    expect(screen.getByTestId('pairing-code')).toHaveTextContent(CODE)
    expect(screen.getByTestId('pairing-countdown')).toHaveTextContent('Expires in 10:00')
    expect(screen.getByTestId('pairing-link')).toHaveTextContent(LINK)
    expect(qr.parentElement?.contains(screen.getByTestId('pairing-code'))).toBe(true)
  })

  it('draws no QR image once the code has expired', async () => {
    const { CODE_EXPIRED } = await renderPanel(0)

    expect(screen.queryByTestId('pairing-qr')).toBeNull()
    expect(screen.getByText(CODE_EXPIRED)).toBeInTheDocument()
  })
})
