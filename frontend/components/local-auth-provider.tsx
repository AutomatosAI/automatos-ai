'use client'

import { useRef } from 'react'
import { apiClient } from '@/lib/api-client'

/**
 * PRD-175 (F008) — the `local` edition counterpart to ClerkApiClientProvider.
 *
 * It reuses the *same* token seam (`apiClient.setClerkTokenGetter`, api-client.ts)
 * — the "one-function seam" — but installs a NO-OP getter that always resolves
 * `null`. With no bearer on the wire, the backend falls through to its single
 * local identity (`hybrid.py` anonymous dev-fallback, gated by REQUIRE_AUTH which
 * `AUTH_EDITION=local` forces false). No Clerk symbol is imported here, so `local`
 * mounts and renders with zero Clerk env / no publishable key.
 *
 * This is a *facade over the existing seam*, not a second source of truth for
 * identity — the backend remains the only authority for who the user is.
 */
export function LocalAuthProvider({ children }: { children: React.ReactNode }) {
  // Install synchronously during render (mirrors ClerkApiClientProvider) so
  // React Query can't fire a request before the getter is set.
  const configuredRef = useRef(false)
  if (!configuredRef.current) {
    apiClient.setClerkTokenGetter(async () => null)
    configuredRef.current = true
  }

  return <>{children}</>
}
