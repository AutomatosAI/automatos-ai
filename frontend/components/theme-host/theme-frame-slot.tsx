'use client'

import type { ReactNode } from 'react'

import { useThemeLayout } from '@/hooks/use-studio-theme'

import { PageBoundary } from './page-boundary'
import { ThemeErrorBoundary } from './theme-error-boundary'
import { useThemeHost } from './theme-host-context'

interface ThemeFrameSlotProps {
  /** The core's own chrome, drawn when no theme frame is active, or when the frame throws. */
  fallback: ReactNode
  /** The page, drawn inside the theme's frame. */
  children: ReactNode
  /** frame@1 props (plan §2.2): MainLayout's full-bleed flag and Auto's widget, handed to the frame. */
  fullBleed?: boolean
  autoWidget?: ReactNode
}

/**
 * PRD-261 A-01, A-04a — the frame slot. With no theme host or no active frame, it draws
 * `fallback` as it is. While the host is still loading a theme under the theme layout it draws
 * nothing, as MainLayout does while the layout is pending (P260-FX-25, M69): the page would mount
 * in the fallback and again in the frame, running its effects twice. With an active frame, the
 * frame draws around the page inside a boundary that falls back to `fallback` and tells the host
 * if the frame throws; the page itself sits inside its own boundary (PageBoundary), so a page
 * that throws shows the core's error UI inside the frame and never takes the theme down.
 */
export function ThemeFrameSlot({ fallback, children, fullBleed, autoWidget }: ThemeFrameSlotProps) {
  const host = useThemeHost()
  const layout = useThemeLayout()
  const Frame = host?.frame ?? null
  if (host?.status === 'loading' && layout === 'theme') return null
  if (host === null || Frame === null) return <>{fallback}</>
  return (
    <ThemeErrorBoundary fallback={fallback} onError={host.reportRenderFailure}>
      <Frame fullBleed={fullBleed} autoWidget={autoWidget}>
        <PageBoundary>{children}</PageBoundary>
      </Frame>
    </ThemeErrorBoundary>
  )
}
