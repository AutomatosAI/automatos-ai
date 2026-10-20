'use client'

/**
 * PRD-261 A-13c, A-14d — the look runtime handed to the SDK's `looks` area (lib/theme-sdk/looks.ts)
 * as a theme-neutral copy: every look a theme hands in is checked against the theme's look spec
 * first. In Phase A the one look runtime the host has is the core-owned one Play's looks run on
 * (PA-9: lib/look-kit bound to Play's catalogue in lib/looks/engine.ts, lib/looks/transfer.ts for
 * files).
 *
 * The LookProvider (./play-look-provider.tsx) renders this bridge around everything below it, so
 * wherever the look runtime is mounted, the SDK's useLook() reads it: the host's ThemeLookRuntime
 * (components/theme-host/theme-look-runtime.tsx) and every test that mounts the provider alike.
 * Since A-14d, Play's own look editor reads the looks only this way (components/looks/use-play-look.ts).
 */
import { useEffect, useMemo, type ReactNode } from 'react'

import { validateLook } from '@/lib/looks/engine'
import { importLook, LOOK_FILE_TYPE, lookFileName, lookJson } from '@/lib/looks/transfer'
import type { Look, ValidationResult } from '@/lib/looks/types'
import { LookRuntimeContext, registerLookRuntime, type HostLookRuntime } from '@/lib/theme-sdk/internal/look-runtime'
import type { HostLook, HostLookFile, HostLookResult, HostLookState } from '@/lib/theme-sdk/looks'

import type { LookContextValue } from './play-look-provider'

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

export interface LookRuntimeBridgeProps {
  readonly children: ReactNode
  /** The LookProvider's value: the looks and their actions. */
  readonly value: LookContextValue
  /** False while no look runtime is on: the SDK gets none (null), and nothing is registered. */
  readonly enabled: boolean
}

/** Gives the SDK the mounted runtime: to useLook through context, to the plain functions as the current one. */
export function LookRuntimeBridge({ children, value, enabled }: LookRuntimeBridgeProps) {
  const runtime = useMemo(() => (enabled ? hostLookRuntime(value) : null), [enabled, value])
  useEffect(() => (runtime === null ? undefined : registerLookRuntime(runtime)), [runtime])
  return <LookRuntimeContext.Provider value={runtime}>{children}</LookRuntimeContext.Provider>
}
