/**
 * PRD-260 F5 (Gerard, 15 Oct) — looks, the active look and the seen
 * Deliverables record are kept per signed-in user and workspace, in this
 * browser: each storage key carries a scope naming both. Nothing goes to the
 * backend. The local edition has no user id (lib/auth-hooks.ts), so its one
 * user is NO_USER in the workspace it runs.
 *
 * What a browser kept before F5, under the unscoped keys, moves to the first
 * scope that reads it, once (moveUnscopedOnce): never to a second account, and
 * the unscoped keys are never deleted.
 */

/** The user part of a scope when there is no user id: the local edition's single user. */
export const NO_USER = 'local'
/** The workspace part of a scope when there is no workspace (outside a WorkspaceProvider). */
export const NO_WORKSPACE = 'default'
/** Between a key and its scope, and between a scope's workspace and user. */
export const SCOPE_SEPARATOR = ':'

export function lookScope(workspaceId: string | null | undefined, userId: string | null | undefined): string {
  return `${workspaceId || NO_WORKSPACE}${SCOPE_SEPARATOR}${userId || NO_USER}`
}

/** The scope where no account or workspace is known: a screen rendered without AccountLookScope (tests). */
export const DEFAULT_LOOK_SCOPE = lookScope(null, null)

/** `base` kept for one scope. */
export function scopedKey(base: string, scope: string): string {
  return `${base}${SCOPE_SEPARATOR}${scope}`
}

/** An unscoped key (before F5) and the scoped key it moves to. */
export interface KeyMove {
  readonly from: string
  readonly to: string
}

/**
 * Copies each unscoped key that holds something to its scoped key (a scoped
 * key already set is left alone), then writes `scope` under `marker`, so the
 * next scope to read finds them taken. Runs once per browser: nothing moves
 * after the marker is set, and nothing is deleted. A storage that refuses
 * throws to the caller, which reads what is there and logs why.
 */
export function moveUnscopedOnce(store: Storage, marker: string, scope: string, moves: readonly KeyMove[]): void {
  if (store.getItem(marker) !== null) return
  const held = moves.filter(({ from }) => store.getItem(from) !== null)
  if (held.length === 0) return
  held.forEach(({ from, to }) => {
    const value = store.getItem(from)
    if (value !== null && store.getItem(to) === null) store.setItem(to, value)
  })
  store.setItem(marker, scope)
}
