'use client'

import { useRef } from 'react'
import { PairBrowserGate } from '@/components/settings/phones-devices/pair-browser-gate'
import { apiClient } from '@/lib/api-client'
import { readDeviceToken } from '@/lib/device-token'

/**
 * PRD-175 (F008) — the `local` edition counterpart to ClerkApiClientProvider.
 *
 * It reuses the *same* token seam (`apiClient.setClerkTokenGetter`, api-client.ts)
 * — the "one-function seam". No Clerk symbol is imported here, so `local` mounts
 * and renders with zero Clerk env / no publishable key.
 *
 * PRD-266 US-005b: the getter returns this browser's device token (lib/device-token.ts,
 * read from browser storage on every call), or `null` when it has none. With no bearer
 * on the wire the backend falls through to its single local identity (`hybrid.py`
 * anonymous lane), exactly as before; on an install with LOCAL_REMOTE_ACCESS=paired
 * that lane is closed, and PairBrowserGate shows "Pair this browser" until the browser
 * holds a token.
 *
 * This is a *facade over the existing seam*, not a second source of truth for
 * identity — the backend remains the only authority for who the user is.
 */
export function LocalAuthProvider({ children }: { children: React.ReactNode }) {
  // Install synchronously during render (mirrors ClerkApiClientProvider) so
  // React Query can't fire a request before the getter is set.
  const configuredRef = useRef(false)
  if (!configuredRef.current) {
    apiClient.setClerkTokenGetter(async () => readDeviceToken())
    configuredRef.current = true
  }

  return <PairBrowserGate>{children}</PairBrowserGate>
}
