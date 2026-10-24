'use client'

/**
 * PRD-261 A-19d2 (plan §2.2.4) — the generic look runtime handed to the SDK's `looks` area
 * (lib/theme-sdk/looks.ts, through lib/theme-sdk/internal/look-runtime.ts): to `useLook()` through
 * context, to the plain functions (`saveLook`, `listLooks`, `exportLook`, `importLook`) as the current
 * runtime. Every look a theme hands in is untrusted: the store checks it against the theme's look
 * rules (./store.ts setLook and saveLook, ./spec.ts checkLook), files go through ./transfer.ts, and
 * what the SDK hands out is a frozen copy, so a theme can't change the runtime's.
 *
 * While no installed theme with looks runs here (`source` null) the bridge hands its children the
 * runtime above it, unchanged, and registers nothing: the host keeps one tree shape (M69). Since
 * W-A8 no runtime is mounted above it (builtin Play's own left core), so they get none.
 */
import { useContext, useEffect, useMemo, type ReactNode } from 'react'

import type { ThemeTone } from '@/lib/theme-host/boot-types'
import { LookRuntimeContext, registerLookRuntime, type HostLookRuntime } from '@/lib/theme-sdk/internal/look-runtime'
import type { HostLook, HostLookResult, HostLookState } from '@/lib/theme-sdk/looks'

import type { ThemeLook, ThemeLookCheck, ThemeLookRules } from './spec'
import type { LookStore } from './store'
import { importLook, lookFile } from './transfer'

/** What the bridge hands the SDK: the running theme's store, its rules and the tone its look forces. */
export interface LookBridgeSource {
  readonly store: LookStore
  readonly rules: ThemeLookRules
  /** The tone the painted look forces; null when the theme's tone doesn't follow its look. */
  readonly tone: ThemeTone | null
}

/** A look as the SDK hands it out: a frozen copy. */
function hostLook(look: ThemeLook): HostLook {
  const copy: HostLook = { ...look }
  return Object.freeze(copy)
}

function hostResult(result: ThemeLookCheck): HostLookResult {
  if (!result.ok) return Object.freeze({ ok: false as const, errors: Object.freeze([...result.errors]) })
  return Object.freeze({ ok: true as const, look: hostLook(result.look) })
}

/** What useLook() hands a theme: the store's state as copies, its actions behind the theme's checks. */
function hostState({ store, tone }: LookBridgeSource): HostLookState {
  return Object.freeze({
    available: true,
    look: store.look === null ? null : hostLook(store.look),
    saved: Object.freeze(store.saved.map(hostLook)),
    loaded: store.loaded,
    persisted: store.persisted,
    tone,
    setLook: (look: HostLook) => hostResult(store.setLook(look)),
    deleteLook: (id: string) => store.deleteLook(id),
    renameLook: (id: string, name: string) => hostResult(store.renameLook(id, name)),
  })
}

/** The runtime as the SDK's looks area reads it. */
export function hostLookRuntime(source: LookBridgeSource): HostLookRuntime {
  const { store, rules } = source
  return Object.freeze({
    state: hostState(source),
    saveLook: (look: unknown) => hostResult(store.saveLook(look)),
    exportLook: (look: unknown) => lookFile(rules, look),
    importLook: (text: string) => hostResult(importLook(rules, text, store.saved)),
  })
}

export interface LookRuntimeBridgeProps {
  readonly children: ReactNode
  /** The running theme's looks; null while none runs here (the runtime above, if any, passes through). */
  readonly source: LookBridgeSource | null
}

export function LookRuntimeBridge({ children, source }: LookRuntimeBridgeProps) {
  const above = useContext(LookRuntimeContext)
  const own = useMemo(() => (source === null ? null : hostLookRuntime(source)), [source])
  useEffect(() => (own === null ? undefined : registerLookRuntime(own)), [own])
  return <LookRuntimeContext.Provider value={own ?? above}>{children}</LookRuntimeContext.Provider>
}
