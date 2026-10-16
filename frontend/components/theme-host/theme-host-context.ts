'use client'

/**
 * PRD-261 A-01 — the theme host's context, apart from its provider, so the frame slot (and so
 * MainLayout) reads the active frame without importing the provider, the notices or safe mode.
 */
import { createContext, useContext } from 'react'
import type { FrameComponent, ThemeFailureReason, ThemeHostStatus } from '@/lib/theme-host/types'

export interface ThemeHostContextValue {
  readonly status: ThemeHostStatus
  readonly reason: ThemeFailureReason | null
  /** The active theme's frame, or null: then the core's chrome draws. */
  readonly frame: FrameComponent | null
  /** Called by the frame slot's boundary when the frame throws while drawing. */
  readonly reportRenderFailure: (error: unknown) => void
}

export const ThemeHostContext = createContext<ThemeHostContextValue | null>(null)

/** Null outside the provider: in every build with the spike off, and in isolated mounts. */
export function useThemeHost(): ThemeHostContextValue | null {
  return useContext(ThemeHostContext)
}
