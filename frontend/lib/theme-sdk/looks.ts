/**
 * PRD-261 A-13c — SDK area `looks`: the active look and the saved looks of the signed-in user in
 * this workspace, for the active theme.
 *
 * The host mounts the theme's look runtime at every width while the active theme declares looks
 * (components/theme-host/theme-look-runtime.tsx): it keeps the looks (this browser's copy and the
 * server's, GET/PUT /api/themes/looks/{themeId}), paints the active look on <html> and forces the
 * tone its look sets. This area is how a theme reads and changes them; it never holds the backend
 * client or storage itself. A look is data checked against the theme's own look spec, never
 * colours, CSS or URLs, so every look a theme hands in is checked before the runtime takes it.
 *
 * Outside a look runtime (Classic, Studio, a theme without looks, an isolated mount) `useLook()`
 * answers `available: false` with no look, `listLooks()` is empty, `exportLook()` is null, and
 * `saveLook()` and `importLook()` refuse in plain words. None of them throws.
 */
import { useContext } from 'react'

import { currentLookRuntime, LookRuntimeContext } from './internal/look-runtime'

/**
 * A look as a theme sees it: its identity, and beside it the theme's own knobs (its
 * looks/schema.json), keys and numbers only, which the theme reads as its own look type. No index
 * signature, so a theme's own look interface passes where a HostLook is asked for.
 */
export interface HostLook {
  readonly schema: string
  readonly id: string
  readonly name: string
  readonly sub: string
}

/** A look the theme's checks took (the look as kept), or the plain-word reasons they refused it. */
export type HostLookResult =
  | { readonly ok: true; readonly look: HostLook }
  | { readonly ok: false; readonly errors: readonly string[] }

/** A look as a file: what a theme's Export offers for download. */
export interface HostLookFile {
  readonly fileName: string
  readonly text: string
  /** The file's media type. */
  readonly type: string
}

export interface HostLookState {
  /** False outside a look runtime: then there is no look, nothing is saved and every action refuses. */
  readonly available: boolean
  /** The active look, possibly edited since it was picked; null outside a look runtime. */
  readonly look: HostLook | null
  /** The user's own saved looks (a theme's presets ship in its package). */
  readonly saved: readonly HostLook[]
  /** False until the looks have been read; the theme's first paint covers that moment. */
  readonly loaded: boolean
  /** False once the looks could not be kept (blocked storage, or the server refused them): say so. */
  readonly persisted: boolean
  /** The tone the active look forces, under a theme whose tone follows its look; null otherwise. */
  readonly tone: 'light' | 'dark' | null
  /** Makes the look the active one, once it passes the theme's checks. */
  setLook(look: HostLook): HostLookResult
  /** Deletes a saved look; deleting the active look goes back to the theme's default. */
  deleteLook(id: string): void
  renameLook(id: string, name: string): HostLookResult
}

/** The words every refusal outside a look runtime carries. */
const LOOKS_NOT_AVAILABLE = "Looks aren't available here."

const NOT_AVAILABLE: HostLookResult = Object.freeze({ ok: false as const, errors: Object.freeze([LOOKS_NOT_AVAILABLE]) })
const NO_LOOKS: readonly HostLook[] = Object.freeze([])

const NO_LOOK_STATE: HostLookState = Object.freeze({
  available: false,
  look: null,
  saved: NO_LOOKS,
  loaded: true,
  persisted: false,
  tone: null,
  setLook: () => NOT_AVAILABLE,
  deleteLook: () => undefined,
  renameLook: () => NOT_AVAILABLE,
})

/** The active look, the saved looks and their actions; the theme re-renders when they change. */
export function useLook(): HostLookState {
  return useContext(LookRuntimeContext)?.state ?? NO_LOOK_STATE
}

/** Adds the look to the saved looks, or replaces the saved look with its id. A preset can't be saved over. */
export function saveLook(look: HostLook): HostLookResult {
  return currentLookRuntime()?.saveLook(look) ?? NOT_AVAILABLE
}

/** The user's saved looks now, for code outside a render (an event handler). */
export function listLooks(): readonly HostLook[] {
  return currentLookRuntime()?.state.saved ?? NO_LOOKS
}

/** The look as a `.look.json` file to offer for download; null when it does not pass the theme's checks. */
export function exportLook(look: HostLook): HostLookFile | null {
  return currentLookRuntime()?.exportLook(look) ?? null
}

/** Pasted or read text as a look ready to save (under a fresh id), or why it is refused. Size-checked first. */
export function importLook(text: string): HostLookResult {
  return currentLookRuntime()?.importLook(text) ?? NOT_AVAILABLE
}
