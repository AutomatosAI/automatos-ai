'use client'

/**
 * PRD-261 A-01, A-12b — the theme host's context, apart from its provider, so the frame slot (and
 * so MainLayout) and the placement slots read what the active theme registered without importing
 * the provider, the notices or safe mode.
 */
import { createContext, useContext } from 'react'
import type { SlotId } from '@/lib/theme-host/registry-types'
import type { FrameComponent, ThemeFailureReason, ThemeHostStatus } from '@/lib/theme-host/types'

export interface ThemeHostContextValue {
  readonly status: ThemeHostStatus
  readonly reason: ThemeFailureReason | null
  /** The active theme's frame, or null: then the core's chrome draws. */
  readonly frame: FrameComponent | null
  /** Every slot the active theme registered (frame@1 included), or null with no active theme. */
  readonly slots: ReadonlyMap<SlotId, unknown> | null
  /** Called by the frame slot's boundary when the frame throws while drawing. */
  readonly reportRenderFailure: (error: unknown) => void
}

export const ThemeHostContext = createContext<ThemeHostContextValue | null>(null)

/** Null outside the provider: in every build with the spike off, and in isolated mounts. */
export function useThemeHost(): ThemeHostContextValue | null {
  return useContext(ThemeHostContext)
}
