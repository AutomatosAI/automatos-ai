'use client'

/**
 * PRD-261 A-13c (PRD §4.3 item 5) — the host mounts the active theme's look runtime, at every width,
 * whenever the theme declares looks (it fills the look editor slot, PA-14).
 *
 * The runtime keeps the signed-in user's looks in this workspace (AccountLookScope: the scope; the
 * look provider: this browser's copy and the server's), paints the active look on <html> and hands
 * the tone its look sets up to ThemeProvider's forcedTheme (components/providers.tsx). Nothing here
 * reads the width: below the theme's minWidth its frame and pages give way to the core's
 * (useThemeLayout), but the look and the forced tone still paint, so Studio's tokens and dark:
 * classes match the look's ground there too (PRD-260 K4).
 *
 * Classic, Studio and a theme without looks get no look runtime: no scope, no storage read, no look
 * on <html>, nothing for the SDK. The runtime's components stay mounted either way and are only
 * switched off (`enabled`), so the tree above the page has one shape: a style switched in place
 * (Studio and builtin Play, or the server's stored choice) never remounts the page below (M69).
 * Leaving a theme with looks switches it off: the look comes off <html>, and the host gives the tone
 * back to the user and deletes the forced tone's cookie.
 *
 * PRD-261 A-19d2 — two runtimes, file-disjoint, until W-A8 deletes builtin Play:
 * - builtin Play keeps its own (lib/theme-host/builtin/play-look-{provider,store,bridge}, Play's
 *   look engine and its `data-pl-motion` and `automatos-play-tone` marks), unchanged;
 * - an installed theme with looks runs on the generic one (lib/theme-host/looks/provider.tsx): its
 *   looks come from its own boot data (the prepaint table, the guard and the presets), it paints
 *   `data-theme-motion` and `--theme-motion`, and its look's tone follows the table's tone rule into
 *   `automatos-theme-tone` and forcedTheme. Play's package is such a theme: it no longer gets the
 *   builtin runtime because its <html> class is Play's.
 * While the host loads or runs an installed theme, a runtime that paints no look leaves <html> as the
 * root layout's boot painted it (W-A6: the theme's prepaint table), so the theme keeps its first
 * paint after hydration. A failed load, safe mode and every other style clear as before: the runtime
 * stops for an installed theme the moment the host stops running it, without waiting for the host to
 * release its entry. The SDK's `looks` area reads whichever runtime runs (the generic bridge hands on
 * builtin Play's while it is off).
 *
 * PB-8a (the W-E3 seam review): the host's one workspace retry (./use-workspace-retry.ts) is called
 * here, whenever the host runs an installed theme or a builtin theme with looks is the style.
 */
import { useEffect, type ReactNode } from 'react'

import { useActiveTheme } from '@/hooks/use-studio-theme'
import type { ThemeTone } from '@/lib/theme-host/boot'
import { LookProvider } from '@/lib/theme-host/builtin/play-look-provider'
import { clearPlayToneCookie } from '@/lib/theme-host/builtin/play-tone-cookie'
import { ThemeLookProvider } from '@/lib/theme-host/looks/provider'
import type { SlotId, ThemeEntry, ThemeTonePolicy } from '@/lib/theme-host/registry-types'
import { THEME_HOST_RUNNING_STATUSES } from '@/lib/theme-host/types'

import { AccountLookScope } from './account-look-scope'
import { useLookScope } from './look-scope'
import { useThemeHost } from './theme-host-context'
import { ThemeStateScope } from './theme-state-scope'
import { useHostWorkspaceRetry } from './use-workspace-retry'

/** The slot a theme with looks fills (PA-14: every installable theme ships one). */
const LOOK_EDITOR_SLOT: SlotId = 'lookEditor@1'

/** No installed theme with looks is the style: the user sets the tone. */
const USER_TONE: ThemeTonePolicy = 'user'

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
 * Whether the host loads or runs an installed theme. A look runtime that paints no look then leaves
 * <html> as the root layout's first paint (it is off under Classic, Studio and builtin Play).
 */
function useHostRunsTheme(): boolean {
  const status = useThemeHost()?.status ?? null
  return status !== null && THEME_HOST_RUNNING_STATUSES.includes(status)
}

/**
 * Whether the active theme gets a look runtime: it declares looks and, for an installed one, the
 * host still loads or runs it. The host's release reaches the style context one commit after it
 * enters safe mode or fails, and the runtime must not repaint the look the host just took off.
 */
function runsLooks(theme: ThemeEntry | null, hostRunsTheme: boolean): boolean {
  return declaresLooks(theme) && (theme?.kind !== 'installed' || hostRunsTheme)
}

/** Which runtime a theme with looks gets: builtin Play's own until W-A8, or the generic one for an installed theme. */
interface LookRuntimes {
  /** A builtin theme with looks is the style: builtin Play's runtime runs. */
  readonly builtin: boolean
  /** The installed theme with looks that is the style, whose looks the generic runtime runs; null otherwise. */
  readonly installed: ThemeEntry | null
}

function lookRuntimesOf(theme: ThemeEntry | null): LookRuntimes {
  if (theme === null || !declaresLooks(theme)) return { builtin: false, installed: null }
  return theme.kind === 'installed' ? { builtin: false, installed: theme } : { builtin: true, installed: null }
}

interface InstalledLooksProps {
  readonly children: ReactNode
  readonly theme: ThemeEntry | null
  /** The host loads or runs the theme. */
  readonly enabled: boolean
  readonly keepFirstPaint: boolean
  readonly onToneChange: (tone: ThemeTone | undefined) => void
}

/** The generic look runtime, below AccountLookScope: the looks are the signed-in user's in this workspace. */
function InstalledLooks(props: InstalledLooksProps) {
  const { theme, enabled, keepFirstPaint, onToneChange } = props
  const scope = useLookScope()
  return (
    <ThemeLookProvider
      themeId={theme?.themeId ?? null}
      enabled={enabled}
      tonePolicy={theme?.tone ?? USER_TONE}
      scope={scope}
      onToneChange={onToneChange}
      keepFirstPaint={keepFirstPaint}
    >
      {props.children}
    </ThemeLookProvider>
  )
}

export function ThemeLookRuntime({ children, onToneChange }: ThemeLookRuntimeProps) {
  const keepFirstPaint = useHostRunsTheme()
  const theme = useActiveTheme()
  const mounted = runsLooks(theme, keepFirstPaint)
  const { builtin, installed } = lookRuntimesOf(theme)
  useToneReleased(mounted, onToneChange)
  useHostWorkspaceRetry(keepFirstPaint || builtin)
  // One shape for every style: a different tree here would remount every page (M69).
  return (
    <AccountLookScope enabled={mounted}>
      <LookProvider onToneChange={builtin ? onToneChange : undefined} enabled={builtin} keepFirstPaint={keepFirstPaint}>
        <InstalledLooks theme={installed} enabled={mounted} keepFirstPaint={keepFirstPaint} onToneChange={onToneChange}>
          <ThemeStateScope>{children}</ThemeStateScope>
        </InstalledLooks>
      </LookProvider>
    </AccountLookScope>
  )
}
