'use client'

/**
 * PRD-261 A-13c (PRD §4.3 item 5) — the host mounts the active theme's look runtime, at every width,
 * whenever the theme declares looks (it fills the look editor slot, PA-14).
 *
 * The runtime keeps the signed-in user's looks in this workspace (AccountLookScope: the scope; the
 * LookProvider: this browser's copy and the server's), paints the active look on <html> and hands
 * the tone its look sets up to ThemeProvider's forcedTheme (components/providers.tsx). Nothing here
 * reads the width: below the theme's minWidth its frame and pages give way to the core's
 * (useThemeLayout), but the look and the forced tone still paint, so Studio's tokens and dark:
 * classes match the look's ground there too (PRD-260 K4).
 *
 * Classic, Studio and a theme without looks get no look runtime: no scope, no storage read, no look
 * on <html>, nothing for the SDK. The runtime's components stay mounted either way and are only
 * switched off (`enabled`), so the tree above the page has one shape: a style switched in place
 * (Studio and builtin Play, or the server's stored choice) never remounts the page below (M69).
 * Leaving a theme with looks switches it off: the LookProvider takes the look off <html>, and the
 * host gives the tone back to the user and deletes the forced tone's cookie.
 *
 * The LookProvider hands the runtime to the SDK's `looks` area (lib/theme-sdk/looks.ts) as a
 * theme-neutral copy (lib/theme-host/builtin/play-look-bridge.tsx): every look a theme hands in is
 * checked against the theme's look spec first. In Phase A the one look runtime the host has is the
 * core-owned one Play's looks run on (PA-9: lib/look-kit bound to Play's catalogue in
 * lib/looks/engine.ts, lib/looks/transfer.ts for files); PRD-261 A-14d moved its files out of Play's
 * folders (./account-look-scope.tsx, lib/theme-host/builtin/play-look-{provider,store,bridge}).
 *
 * PRD-261 W-A6: an installed theme gets no look runtime yet (its style has no registry entry, so
 * useActiveTheme() is null: gap G1 of e2e/browser/parity-checklist.md). While the host loads or
 * runs one, the switched-off LookProvider leaves <html> as the root layout's boot painted it (the
 * theme's prepaint table, under the same names as Play's look), so the installed theme keeps its
 * first-paint look after hydration instead of losing every look variable. A failed load, safe mode
 * and every other style clear as before.
 */
import { useEffect, type ReactNode } from 'react'

import { useActiveTheme } from '@/hooks/use-studio-theme'
import type { ThemeTone } from '@/lib/theme-host/boot'
import { LookProvider } from '@/lib/theme-host/builtin/play-look-provider'
import { clearPlayToneCookie } from '@/lib/theme-host/builtin/play-tone-cookie'
import type { SlotId, ThemeEntry } from '@/lib/theme-host/registry-types'
import type { ThemeHostStatus } from '@/lib/theme-host/types'

import { AccountLookScope } from './account-look-scope'
import { useThemeHost } from './theme-host-context'

/** The slot a theme with looks fills (PA-14: every installable theme ships one). */
const LOOK_EDITOR_SLOT: SlotId = 'lookEditor@1'

/** The host's states in which an installed theme owns <html>'s first paint: it is loading or running. */
const HOST_RUNS_INSTALLED_THEME: readonly ThemeHostStatus[] = ['loading', 'active']

export interface ThemeLookRuntimeProps {
  readonly children: ReactNode
  /** Receives the tone the look forces, and undefined when none is forced: ThemeProvider's forcedTheme. */
  readonly onToneChange: (tone: ThemeTone | undefined) => void
}

/** Whether a theme declares looks: it fills the look editor slot. Classic and Studio (null) never do. */
export function declaresLooks(theme: ThemeEntry | null): boolean {
  return theme !== null && theme.slots.includes(LOOK_EDITOR_SLOT)
}

/** With no look runtime mounted, nothing forces the tone: the user's own returns, and the cookie goes. */
function useToneReleased(mounted: boolean, onToneChange: (tone: ThemeTone | undefined) => void): void {
  useEffect(() => {
    if (mounted) return
    clearPlayToneCookie()
    onToneChange(undefined)
  }, [mounted, onToneChange])
}

/**
 * With no look runtime, whether <html> keeps the root layout's first paint: only while the host
 * loads or runs an installed theme (it is off under Classic, Studio and builtin Play).
 */
function useKeepsFirstPaint(mounted: boolean): boolean {
  const status = useThemeHost()?.status ?? null
  return !mounted && status !== null && HOST_RUNS_INSTALLED_THEME.includes(status)
}

export function ThemeLookRuntime({ children, onToneChange }: ThemeLookRuntimeProps) {
  const mounted = declaresLooks(useActiveTheme())
  const keepFirstPaint = useKeepsFirstPaint(mounted)
  useToneReleased(mounted, onToneChange)
  // One shape for both values of `mounted`: a different tree here would remount every page (M69).
  return (
    <AccountLookScope enabled={mounted}>
      <LookProvider onToneChange={onToneChange} enabled={mounted} keepFirstPaint={keepFirstPaint}>
        {children}
      </LookProvider>
    </AccountLookScope>
  )
}
