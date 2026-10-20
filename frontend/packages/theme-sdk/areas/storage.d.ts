/**
 * PRD-261 E3b — SDK area `storage` (frontend/lib/theme-sdk/storage.ts): small per-user state a theme keeps through
 * the host, never in the browser's storage directly. A key (`^[a-z][a-z0-9-]{0,47}$`) holds a JSON value, kept for
 * the active installed theme and the signed-in user (`within: 'account'`, the default) or the user in this workspace
 * (`within: 'workspace'`), in this browser (`lifetime: 'browser'`, the default) or for the tab's session
 * (`lifetime: 'session'`), within THEME_STATE_LIMITS. Outside a running installed theme reads answer `ready: false`
 * and writes refuse `not-ready`; on the server writes refuse `unavailable`; with the browser's storage blocked writes
 * answer `persisted: false` and the state lasts for the page. None throws. Every theme runs on the host's origin, so
 * the state is scoped per theme and per account, not isolated.
 */

/** A value a theme may keep: JSON only. */
export type ThemeStateValue =
  | null | boolean | number | string
  | readonly ThemeStateValue[] | { readonly [k: string]: ThemeStateValue }

/** Where a key is kept: `lifetime` 'browser' (default) or 'session'; `within` 'account' (default) or 'workspace'. */
export interface ThemeStateOptions {
  lifetime?: 'browser' | 'session'
  within?: 'account' | 'workspace'
}

/** A key's value; `ready` is false outside a running installed theme, and then the value is null. */
export interface ThemeStateRead {
  ready: boolean
  value: ThemeStateValue | null
}

/** A write or remove: done (`persisted` false while the browser blocks its storage), or refused in plain words. */
export type ThemeStateResult =
  | { ok: true; persisted: boolean }
  | { ok: false; reason: 'not-ready' | 'bad-key' | 'bad-value' | 'too-large' | 'unavailable'; message: string }

/** The key's value now, for code outside a render; the same object while the value is unchanged. */
export declare function readThemeState(key: string, options?: ThemeStateOptions): ThemeStateRead

/** Keeps `value` under `key`, once it is JSON within THEME_STATE_LIMITS. */
export declare function writeThemeState(key: string, value: ThemeStateValue, options?: ThemeStateOptions): ThemeStateResult

/** Forgets `key`; removing a key that is not there is done, and changes nothing. */
export declare function removeThemeState(key: string, options?: ThemeStateOptions): ThemeStateResult

/** Calls `listener` when the key's state may have changed, until the returned function is called. */
export declare function subscribeThemeState(key: string, listener: () => void, options?: ThemeStateOptions): () => void

/** The key's state for a component, re-rendering when it changes, with `set` and `remove` bound to the key. */
export declare function useThemeState(key: string, options?: ThemeStateOptions): {
  ready: boolean
  value: ThemeStateValue | null
  persisted: boolean
  set(value: ThemeStateValue): ThemeStateResult
  remove(): ThemeStateResult
}

/** The caps: the key pattern, keys and characters per theme blob, characters per value, nesting depth, blobs per lifetime. */
export declare const THEME_STATE_LIMITS: Readonly<{
  keyPattern: string
  keysPerBlob: 32
  valueChars: 32768
  blobChars: 65536
  depth: 8
  blobsPerLifetime: 8
}>
