'use client'

/**
 * PRD-266 US-019c (§4.3 "Web", K17): the QR image in "Pair a phone". It encodes the pairing link
 * (qr_payload, the universal link `<web address>/go/pair?api=<API address>&code=<code>`) with the
 * vendored encoder (lib/vendor/qrcodegen, Nayuki's v1.8.0) and draws its module matrix as one SVG
 * path. No markup is injected.
 *
 * The error correction level is M exactly: the encoder may not raise it. A quiet zone of four light
 * modules surrounds the symbol. The modules are dark on a light ground in both themes, because a
 * symbol in the theme's colours would invert in dark mode and many scanners read only dark on
 * light. Pairing exists only on a local install, so on SaaS this draws nothing. The panel draws it
 * only while the code is live, with the code, its countdown and the link beside it as text.
 */
import { useMemo } from 'react'

import { isLocal } from '@/lib/auth-edition'
import { QrCode } from '@/lib/vendor/qrcodegen/qrcodegen'
import { QrSegment } from '@/lib/vendor/qrcodegen/qrsegment'

/** The light margin, in modules, that ISO/IEC 18004 puts around a symbol. */
export const QUIET_ZONE_MODULES = 4
/** Dark on light in every theme, so a camera reads it in dark mode too. */
export const QR_DARK = '#000000'
export const QR_LIGHT = '#ffffff'
export const QR_LABEL = 'QR code of the pairing link'
/** The drawn width and height, in CSS pixels. */
const QR_SIZE_PX = 176
/** Let the encoder choose the mask with the lowest penalty. */
const AUTO_MASK = -1
const KEEP_ECC_LEVEL = false

const indices = (count: number): number[] => Array.from({ length: count }, (_, index) => index)

/** The link as a QR Code at level M, or null when it is too long for any version (never logged: it holds the code). */
export function encodePairingQr(payload: string): QrCode | null {
  try {
    const segments = QrSegment.makeSegments(payload)
    return QrCode.encodeSegments(segments, QrCode.Ecc.MEDIUM, QrCode.MIN_VERSION, QrCode.MAX_VERSION, AUTO_MASK, KEEP_ECC_LEVEL)
  } catch (error) {
    if (!(error instanceof RangeError)) throw error
    // console, as the rest of components/settings does: '@/lib/logger' resolves to the legacy CommonJS
    // lib/logger.js under vitest (.js before .ts) and has no named `logger` there.
    console.warn(`The pairing link was not drawn as a QR code: ${error.message}`)
    return null
  }
}

/** One path over the dark modules: a 1×1 square each, offset by the quiet zone. */
export function qrModulePath(qr: QrCode): string {
  return indices(qr.size)
    .flatMap((y) => indices(qr.size)
      .filter((x) => qr.getModule(x, y))
      .map((x) => `M${x + QUIET_ZONE_MODULES},${y + QUIET_ZONE_MODULES}h1v1h-1z`))
    .join('')
}

export function PairPhoneQr({ payload }: { payload: string }) {
  const qr = useMemo(() => (isLocal ? encodePairingQr(payload) : null), [payload])
  if (!qr) return null
  const side = qr.size + 2 * QUIET_ZONE_MODULES
  return (
    <svg
      role="img"
      aria-label={QR_LABEL}
      data-testid="pairing-qr"
      viewBox={`0 0 ${side} ${side}`}
      width={QR_SIZE_PX}
      height={QR_SIZE_PX}
      shapeRendering="crispEdges"
      className="shrink-0 rounded"
    >
      <rect width={side} height={side} fill={QR_LIGHT} />
      <path d={qrModulePath(qr)} fill={QR_DARK} />
    </svg>
  )
}
