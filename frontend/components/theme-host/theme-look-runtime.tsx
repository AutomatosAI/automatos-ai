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
 * LookRuntimeBridge hands the runtime to the SDK's `looks` area (lib/theme-sdk/looks.ts) as a
 * theme-neutral copy: every look a theme hands in is checked against the theme's look spec first.
 * In Phase A the one look runtime the host has is the core-owned one Play's looks run on (PA-9:
 * lib/look-kit bound to Play's catalogue in lib/looks/engine.ts, lib/looks/transfer.ts for files).
 */
import { useEffect, useMemo, type ReactNode } from 'react'

import { AccountLookScope } from '@/components/looks/account-look-scope'
import { LookProvider, useLook as useRuntimeLook, type LookContextValue } from '@/components/looks/look-provider'
import { useActiveTheme } from '@/hooks/use-studio-theme'
import { validateLook } from '@/lib/looks/engine'
import { clearPlayToneCookie } from '@/lib/looks/tone'
import { importLook, LOOK_FILE_TYPE, lookFileName, lookJson } from '@/lib/looks/transfer'
import type { Look, ValidationResult } from '@/lib/looks/types'
import type { ThemeTone } from '@/lib/theme-host/boot'
import type { SlotId, ThemeEntry } from '@/lib/theme-host/registry-types'
import { LookRuntimeContext, registerLookRuntime, type HostLookRuntime } from '@/lib/theme-sdk/internal/look-runtime'
import type { HostLook, HostLookFile, HostLookResult, HostLookState } from '@/lib/theme-sdk/looks'

/** The slot a theme with looks fills (PA-14: every installable theme ships one). */
const LOOK_EDITOR_SLOT: SlotId = 'lookEditor@1'

export interface ThemeLookRuntimeProps {
  readonly children: ReactNode
  /** Receives the tone the look forces, and undefined when none is forced: ThemeProvider's forcedTheme. */
  readonly onToneChange: (tone: ThemeTone | undefined) => void
}

/** Whether a theme declares looks: it fills the look editor slot. Classic and Studio (null) never do. */
export function declaresLooks(theme: ThemeEntry | null): boolean {
  return theme !== null && theme.slots.includes(LOOK_EDITOR_SLOT)
}

/** A look as the SDK hands it out: a frozen copy, so a theme can't change the runtime's. */
function hostLook(look: Look): HostLook {
  const copy: HostLook = { ...look }
  return Object.freeze(copy)
}

function hostResult(result: ValidationResult): HostLookResult {
  if (!result.ok) return Object.freeze({ ok: false as const, errors: result.errors })
  return Object.freeze({ ok: true as const, look: hostLook(result.look) })
}

/** The look as a file, once it passes the theme's checks; null otherwise. */
function lookFile(input: unknown): HostLookFile | null {
  const checked = validateLook(input)
  if (!checked.ok) return null
  return Object.freeze({ fileName: lookFileName(checked.look), text: lookJson(checked.look), type: LOOK_FILE_TYPE })
}

/** What useLook() hands a theme: the runtime's state as copies, its actions behind the theme's checks. */
function hostState(runtime: LookContextValue): HostLookState {
  const setLook = (input: unknown): HostLookResult => {
    const checked = validateLook(input)
    if (checked.ok) runtime.setLook(checked.look)
    return hostResult(checked)
  }
  return Object.freeze({
    available: true,
    look: hostLook(runtime.look),
    saved: Object.freeze(runtime.saved.map(hostLook)),
    loaded: runtime.loaded,
    persisted: runtime.persisted,
    tone: runtime.tone ?? null,
    setLook,
    deleteLook: (id: string) => runtime.deleteLook(id),
    renameLook: (id: string, name: string) => hostResult(runtime.renameLook(id, name)),
  })
}

/** The runtime as the SDK's looks area reads it. */
function hostLookRuntime(runtime: LookContextValue): HostLookRuntime {
  const saveLook = (input: unknown): HostLookResult => {
    const checked = validateLook(input)
    return hostResult(checked.ok ? runtime.saveLook(checked.look) : checked)
  }
  return Object.freeze({
    state: hostState(runtime),
    saveLook,
    exportLook: lookFile,
    importLook: (text: string) => hostResult(importLook(text, runtime.saved)),
  })
}

interface LookRuntimeBridgeProps {
  readonly children: ReactNode
  /** False while no look runtime is on: the SDK gets none (null), and nothing is registered. */
  readonly enabled: boolean
}

/** Gives the SDK the mounted runtime: to useLook through context, to the plain functions as the current one. */
function LookRuntimeBridge({ children, enabled }: LookRuntimeBridgeProps) {
  const value = useRuntimeLook()
  const runtime = useMemo(() => (enabled ? hostLookRuntime(value) : null), [enabled, value])
  useEffect(() => (runtime === null ? undefined : registerLookRuntime(runtime)), [runtime])
  return <LookRuntimeContext.Provider value={runtime}>{children}</LookRuntimeContext.Provider>
}

/** With no look runtime mounted, nothing forces the tone: the user's own returns, and the cookie goes. */
function useToneReleased(mounted: boolean, onToneChange: (tone: ThemeTone | undefined) => void): void {
  useEffect(() => {
    if (mounted) return
    clearPlayToneCookie()
    onToneChange(undefined)
  }, [mounted, onToneChange])
}

export function ThemeLookRuntime({ children, onToneChange }: ThemeLookRuntimeProps) {
  const mounted = declaresLooks(useActiveTheme())
  useToneReleased(mounted, onToneChange)
  // One shape for both values of `mounted`: a different tree here would remount every page (M69).
  return (
    <AccountLookScope enabled={mounted}>
      <LookProvider onToneChange={onToneChange} enabled={mounted}>
        <LookRuntimeBridge enabled={mounted}>{children}</LookRuntimeBridge>
      </LookProvider>
    </AccountLookScope>
  )
}
