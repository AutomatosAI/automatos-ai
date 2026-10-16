'use client'

import type { ReactNode } from 'react'
import { ThemeErrorBoundary } from './theme-error-boundary'
import { useThemeHost } from './theme-host-context'

interface ThemeFrameSlotProps {
  /** The core's own chrome, drawn when no theme frame is active, or when the frame throws. */
  fallback: ReactNode
  /** The page, drawn inside the theme's frame. */
  children: ReactNode
}

/**
 * PRD-261 A-01 — the frame slot. With no theme host (every build with the spike off) or no active
 * frame, it draws `fallback` as it is. With an active frame, the frame draws around the page,
 * inside a boundary that falls back to `fallback` and tells the host if the frame throws.
 */
export function ThemeFrameSlot({ fallback, children }: ThemeFrameSlotProps) {
  const host = useThemeHost()
  const Frame = host?.frame ?? null
  if (host === null || Frame === null) return <>{fallback}</>
  return (
    <ThemeErrorBoundary fallback={fallback} onError={host.reportRenderFailure}>
      <Frame>{children}</Frame>
    </ThemeErrorBoundary>
  )
}
