'use client'

/**
 * PRD-266 US-005b — the "Pair this browser" gate (§4.3), mounted by LocalAuthProvider.
 *
 * On a local install in paired mode a browser with no device token sees PairBrowserScreen
 * instead of the app; once it pairs, or whenever it holds a token, it sees the app. In SaaS, and
 * on a local install that is off or open, it is the app, unchanged (hooks/use-mobile-devices.ts).
 *
 * The public paths stay open to an unpaired browser: /go/pair is the page a QR code scanned
 * without the app opens (PRD-266 W3 US-018d), so it must render before any pairing.
 */
import { usePathname } from 'next/navigation'
import type { ReactNode } from 'react'

import { useBrowserPairing } from '@/hooks/use-mobile-devices'

import { PairBrowserScreen } from './pair-browser-screen'

/** Path prefixes the gate never covers. */
export const PAIRING_GATE_PUBLIC_PREFIXES: readonly string[] = ['/go/']

/** Is `path` one an unpaired browser may open? */
export function isPairingGatePublicPath(path: string | null): boolean {
  return path !== null && PAIRING_GATE_PUBLIC_PREFIXES.some((prefix) => path.startsWith(prefix))
}

export function PairBrowserGate({ children }: { children: ReactNode }) {
  const pathname = usePathname()
  const { needsPairing, keepPairing } = useBrowserPairing()
  if (!needsPairing || isPairingGatePublicPath(pathname)) return <>{children}</>
  return <PairBrowserScreen onPaired={keepPairing} />
}
