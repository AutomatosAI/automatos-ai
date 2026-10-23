'use client'

/**
 * PRD-261 A-04b — what the active theme registered, for the theme slots (ThemeSlot, ThemeSlotPage,
 * ThemeSlotRoute): an installed theme's registrations, which the theme host hands over once it has
 * loaded the theme's code (theme-host-provider.tsx); the frame slot reads the host's own
 * (theme-frame-slot.tsx). No theme is compiled into the app (A-19f1), so there is no other source.
 *
 * The layout says where the theme draws: 'none' (Classic, Studio, or an installed theme that is
 * off, failed or in safe mode: the core draws), 'pending' (the width is not known yet, the server
 * and hydration; or the host is still loading an installed theme, whose registrations are not
 * known yet), 'narrow' (below the theme's minWidth: the core draws inside the fallback chrome) and
 * 'theme'. An installed theme draws where MainLayout draws its frame (theme-frame-slot.tsx): from
 * the desktop width up (INSTALLED_FRAME_MIN_WIDTH_PX); below it, as any theme below its minWidth,
 * the core's pages, slots and notes ('narrow'), so a phone never gets a theme page cut for a frame
 * it does not have. Its registry entry (W-A6b, lib/theme-host/installed-entry.ts) names that width
 * as its minWidth; a theme with no entry (its boot carried none) reads the same width here.
 */
import { useCallback, useSyncExternalStore, type ComponentType } from 'react'

import { useActiveTheme, useThemeLayout } from '@/hooks/use-studio-theme'
import { INSTALLED_FRAME_MIN_WIDTH_PX } from '@/lib/theme-host/installed-entry'
import type { SlotId, ThemeLayout } from '@/lib/theme-host/registry-types'
import type { ThemeRegistrations } from '@/lib/theme-host/types'

import { useThemeHost, type ThemeHostContextValue } from './theme-host-context'

export type ThemeSlotLayout = 'none' | 'pending' | 'narrow' | 'theme'

export interface ActiveThemeSlots {
  readonly layout: ThemeSlotLayout
  /** What the active theme registered; null with no theme, and while an installed theme loads. */
  readonly registrations: ThemeRegistrations | null
}

const NO_THEME: ActiveThemeSlots = { layout: 'none', registrations: null }
const INSTALLED_LOADING: ActiveThemeSlots = { layout: 'pending', registrations: null }
const NO_SLOTS: ReadonlyMap<SlotId, unknown> = new Map()
const NO_PAGES: ReadonlyMap<string, unknown> = new Map()

export { INSTALLED_FRAME_MIN_WIDTH_PX }
const BELOW_INSTALLED_FRAME_QUERY = `(max-width: ${INSTALLED_FRAME_MIN_WIDTH_PX - 1}px)`

function hasMatchMedia(): boolean {
  return typeof window.matchMedia === 'function'
}

/** Below the frame's width now; a browser that can't say is a wide one, as hooks/use-mobile.ts reads it. */
function belowInstalledFrameNow(): boolean | null {
  return hasMatchMedia() && window.matchMedia(BELOW_INSTALLED_FRAME_QUERY).matches
}

/** The server has no window: not known. */
const widthOnServer = (): boolean | null => null

/** 'theme' from the desktop width up, 'narrow' below it, 'pending' on the server and while hydrating (M69). */
function useInstalledFrameLayout(): ThemeSlotLayout {
  const subscribe = useCallback((onChange: () => void) => {
    if (!hasMatchMedia()) return () => undefined
    const query = window.matchMedia(BELOW_INSTALLED_FRAME_QUERY)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])
  const below = useSyncExternalStore(subscribe, belowInstalledFrameNow, widthOnServer)
  if (below === null) return 'pending'
  return below ? 'narrow' : 'theme'
}

function layoutOf(layout: ThemeLayout): ThemeSlotLayout {
  if (layout === 'fallback') return 'narrow'
  return layout
}

/** The theme host's registrations; the provider hands over the frame, slots, pages and routes it loaded. */
function hostRegistrations(host: ThemeHostContextValue): ThemeRegistrations | null {
  if (host.frame === null) return null
  return {
    frame: host.frame,
    slots: host.slots ?? NO_SLOTS,
    pages: host.pages ?? NO_PAGES,
    routes: host.routes ?? NO_PAGES,
  }
}

function installedSlots(host: ThemeHostContextValue | null, layout: ThemeSlotLayout): ActiveThemeSlots {
  if (host === null) return NO_THEME
  if (host.status === 'loading') return INSTALLED_LOADING
  const registrations = host.status === 'active' ? hostRegistrations(host) : null
  return registrations === null ? NO_THEME : { layout, registrations }
}

/** The active theme's registrations and where it draws them (see the module comment). */
export function useActiveThemeSlots(): ActiveThemeSlots {
  const entry = useActiveTheme()
  const layout = useThemeLayout()
  const host = useThemeHost()
  const installedFrameLayout = useInstalledFrameLayout()
  return installedSlots(host, entry === null ? installedFrameLayout : layoutOf(layout))
}

/** A registered component as React renders it, or null for nothing registered. */
export function registeredComponent(registered: unknown): ComponentType | null {
  return registered === undefined || registered === null ? null : (registered as ComponentType)
}

