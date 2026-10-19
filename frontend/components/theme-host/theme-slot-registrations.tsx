'use client'

/**
 * PRD-261 A-04b — what the active theme registered, for the theme slots (ThemeSlot, ThemeSlotPage,
 * ThemeSlotRoute) and, through ThemeSlotScope, the frame slot. Builtin Play and an installed theme
 * plug in the same way: a builtin theme's registrations are module constants
 * (lib/theme-host/builtin/slots.ts), known on the first render; an installed theme's come from the
 * theme host once it has loaded the theme's code (theme-host-provider.tsx).
 *
 * The layout says where the theme draws: 'none' (Classic, Studio, or an installed theme that is
 * off, failed or in safe mode: the core draws), 'pending' (the width is not known yet, the server
 * and hydration; or the host is still loading an installed theme, whose registrations are not
 * known yet), 'narrow' (below the theme's minWidth: the core draws inside the fallback chrome) and
 * 'theme'. An installed theme with no registry entry yet (A-02c adds them) draws wherever the host
 * shows it, as its frame does (theme-frame-slot.tsx).
 */
import { useMemo, type ComponentType, type ReactNode } from 'react'

import { useActiveTheme, useThemeLayout } from '@/hooks/use-studio-theme'
import { builtinRegistrationsOf } from '@/lib/theme-host/builtin/slots'
import type { SlotId, ThemeEntry, ThemeLayout } from '@/lib/theme-host/registry-types'
import type { ThemeRegistrations } from '@/lib/theme-host/types'

import { ThemeHostContext, useThemeHost, type ThemeHostContextValue } from './theme-host-context'

export type ThemeSlotLayout = 'none' | 'pending' | 'narrow' | 'theme'

export interface ActiveThemeSlots {
  readonly layout: ThemeSlotLayout
  /** What the active theme registered; null with no theme, and while an installed theme loads. */
  readonly registrations: ThemeRegistrations | null
}

const LOG_PREFIX = '[theme-host]'
const NO_THEME: ActiveThemeSlots = { layout: 'none', registrations: null }
const INSTALLED_LOADING: ActiveThemeSlots = { layout: 'pending', registrations: null }
const NO_SLOTS: ReadonlyMap<SlotId, unknown> = new Map()
const NO_PAGES: ReadonlyMap<string, unknown> = new Map()

/**
 * The theme host's pages and routes. The context carries them once the provider hands over the
 * whole load result (the W-A2 seam: theme-host-provider.tsx, the A-04c lane's file); until then an
 * installed theme's pages and routes read as none, and its frame and slots still draw.
 */
type HostWithPages = ThemeHostContextValue & {
  readonly pages?: ReadonlyMap<string, unknown> | null
  readonly routes?: ReadonlyMap<string, unknown> | null
}

function layoutOf(layout: ThemeLayout): ThemeSlotLayout {
  if (layout === 'fallback') return 'narrow'
  return layout
}

function hostRegistrations(host: HostWithPages): ThemeRegistrations | null {
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
  const builtin = builtinRegistrationsOf(entry)
  if (builtin !== null) return { layout: layoutOf(layout), registrations: builtin }
  return installedSlots(host, entry === null ? 'theme' : layoutOf(layout))
}

/** A registered component as React renders it, or null for nothing registered. */
export function registeredComponent(registered: unknown): ComponentType | null {
  return registered === undefined || registered === null ? null : (registered as ComponentType)
}

/** A builtin theme's frame failing is logged; the frame slot's boundary keeps the core's chrome. */
function logBuiltinRenderFailure(error: unknown): void {
  console.error(LOG_PREFIX, 'render', error)
}

function builtinHostValue(registrations: ThemeRegistrations | null): ThemeHostContextValue | null {
  if (registrations === null) return null
  return {
    status: 'active',
    reason: null,
    frame: registrations.frame,
    slots: registrations.slots,
    reportRenderFailure: logBuiltinRenderFailure,
  }
}

function useBuiltinHostValue(entry: ThemeEntry | null): ThemeHostContextValue | null {
  const registrations = builtinRegistrationsOf(entry)
  return useMemo(() => builtinHostValue(registrations), [registrations])
}

/**
 * Hands the frame slot (and the placement slots inside it) a builtin theme's registrations, as the
 * theme host hands them an installed theme's, while the builtin theme draws (its layout is
 * 'theme'). Under any other style, or below the theme's width, the host's own value stands.
 */
export function ThemeSlotScope({ children }: { readonly children: ReactNode }) {
  const value = useBuiltinHostValue(useActiveTheme())
  const layout = useThemeLayout()
  if (value === null || layout !== 'theme') return <>{children}</>
  return <ThemeHostContext.Provider value={value}>{children}</ThemeHostContext.Provider>
}
