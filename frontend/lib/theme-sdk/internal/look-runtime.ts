'use client'

/**
 * PRD-261 A-13c — the seam between the SDK's `looks` area and the look runtime the host mounts.
 *
 * The host mounts a look runtime only while the active theme declares looks
 * (components/theme-host/theme-look-runtime.tsx). That component hands the runtime to the SDK in two
 * ways: through LookRuntimeContext, which `useLook` reads (so a theme re-renders when the look
 * changes), and by registering it as the current one, which the plain functions `saveLook`,
 * `listLooks`, `exportLook` and `importLook` read (a theme calls them from an event handler, where
 * no hook can run). Nothing here knows a theme or a look format: the runtime checks every look
 * against its own theme's look spec.
 */
import { createContext } from 'react'

import type { HostLook, HostLookFile, HostLookResult, HostLookState } from '../looks'

/** What a mounted look runtime gives the SDK. Every input is untrusted theme data, checked by the runtime. */
export interface HostLookRuntime {
  /** What useLook() hands a theme: the active look, the saved looks and their actions. */
  readonly state: HostLookState
  /** Adds the look to the saved looks, or replaces the saved look with its id, once it passes the theme's checks. */
  saveLook(look: unknown): HostLookResult
  /** The look as a file the theme can offer for download, or null when it does not pass the theme's checks. */
  exportLook(look: unknown): HostLookFile | null
  /** Pasted or read text as a look ready to save under a fresh id, or the plain-word reasons it is refused. */
  importLook(text: string): HostLookResult
}

/** The runtime the host mounted for the active theme; null outside one (Classic, Studio, a theme without looks). */
export const LookRuntimeContext = createContext<HostLookRuntime | null>(null)

let current: HostLookRuntime | null = null

/** The runtime mounted now, for the SDK's plain functions; null when none is. */
export function currentLookRuntime(): HostLookRuntime | null {
  return current
}

/**
 * Makes `runtime` the current one. Returns the cleanup that removes it, only if it is still the
 * current one, so a runtime that replaced it is never dropped by the old one's cleanup.
 */
export function registerLookRuntime(runtime: HostLookRuntime): () => void {
  current = runtime
  return () => {
    if (current === runtime) current = null
  }
}
