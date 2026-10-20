/**
 * PRD-261 A-13c — SDK area `looks` (frontend/lib/theme-sdk/looks.ts): the active look and the saved looks of the
 * signed-in user in this workspace, for the active theme. The host mounts the theme's look runtime at every width
 * while the theme declares looks; it keeps the looks, paints the active one and forces the tone it sets. Every look
 * a theme hands in is checked against the theme's own look spec first. Outside a look runtime `useLook()` answers
 * `available: false` with no look, `listLooks()` is empty, `exportLook()` is null and the rest refuse; none throws.
 * `useLookScope()` is whose looks these are (the signed-in user in this workspace), for what a theme keeps per account.
 */

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

/** The active look, the saved looks and their actions; the theme re-renders when they change. */
export declare function useLook(): HostLookState
/** Adds the look to the saved looks, or replaces the saved look with its id. A preset can't be saved over. */
export declare function saveLook(look: HostLook): HostLookResult
/** The user's saved looks now, for code outside a render (an event handler). */
export declare function listLooks(): readonly HostLook[]
/** The look as a `.look.json` file to offer for download; null when it does not pass the theme's checks. */
export declare function exportLook(look: HostLook): HostLookFile | null
/** Pasted or read text as a look ready to save (under a fresh id), or why it is refused. Size-checked first. */
export declare function importLook(text: string): HostLookResult
/**
 * Whose looks these are: the signed-in user in this workspace, as one opaque string to key what the theme keeps
 * per account in this browser. Null until the account and the workspace are known (keep nothing then); outside the
 * host's account scope (an isolated mount) the host's default scope.
 */
export declare function useLookScope(): string | null
