/**
 * PRD-261 E3b — SDK area `storage`: small per-user state a theme keeps through the host (plan §2.2.1),
 * so a theme never touches the browser's storage itself (the kit's review refuses `localStorage`,
 * `sessionStorage` and `indexedDB` in a theme).
 *
 * A theme names a key (`^[a-z][a-z0-9-]{0,47}$`) and a JSON value. The host keeps it for the active
 * installed theme and the signed-in user (`within: 'account'`, the default) or the user in this
 * workspace (`within: 'workspace'`), in this browser until the theme removes it (`lifetime: 'browser'`,
 * the default) or for the tab's session (`lifetime: 'session'`). Values and blobs are capped by
 * THEME_STATE_LIMITS; a write over a cap is refused and changes nothing.
 *
 * Outside a running installed theme (Classic, Studio, safe mode, an isolated mount) reads answer
 * `{ready: false}` and writes refuse `not-ready`; on the server writes refuse `unavailable`. When the
 * browser blocks its storage, writes answer `{ok: true, persisted: false}` and the state lasts for the
 * page. None of these throws.
 *
 * Every theme runs on the host's origin: the state is scoped per theme and per account, not isolated.
 */
import { useCallback, useMemo, useSyncExternalStore } from 'react'

import {
  NOT_READY_SNAPSHOT,
  readThemeStateSnapshot,
  removeThemeStateValue,
  subscribeToThemeState,
  writeThemeStateValue,
  type ThemeStateSnapshot,
} from './internal/theme-state-runtime'
import type { ThemeStateFailureReason, ThemeStateValue } from './internal/theme-state-store'

export { THEME_STATE_LIMITS } from './internal/theme-state-store'
export type { ThemeStateValue } from './internal/theme-state-store'

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
  | { ok: false; reason: ThemeStateFailureReason; message: string }

/** The key's value now, for code outside a render; the same object while the value is unchanged. */
export function readThemeState(key: string, options?: ThemeStateOptions): ThemeStateRead {
  return readThemeStateSnapshot(key, options)
}

/** Keeps `value` under `key`, once it is JSON within THEME_STATE_LIMITS. */
export function writeThemeState(key: string, value: ThemeStateValue, options?: ThemeStateOptions): ThemeStateResult {
  return writeThemeStateValue(key, value, options)
}

/** Forgets `key`; removing a key that is not there is done, and changes nothing. */
export function removeThemeState(key: string, options?: ThemeStateOptions): ThemeStateResult {
  return removeThemeStateValue(key, options)
}

/** Calls `listener` when the key's state may have changed (here, in another tab, or the theme or user changed). */
export function subscribeThemeState(key: string, listener: () => void, options?: ThemeStateOptions): () => void {
  return subscribeToThemeState(key, listener, options)
}

function serverSnapshot(): ThemeStateSnapshot {
  return NOT_READY_SNAPSHOT
}

/** The key's state for a component, re-rendering when it changes, with `set` and `remove` bound to the key. */
export function useThemeState(key: string, options?: ThemeStateOptions): {
  ready: boolean
  value: ThemeStateValue | null
  persisted: boolean
  set(value: ThemeStateValue): ThemeStateResult
  remove(): ThemeStateResult
} {
  const lifetime = options?.lifetime
  const within = options?.within
  const subscribe = useCallback(
    (listener: () => void) => subscribeToThemeState(key, listener, { lifetime, within }),
    [key, lifetime, within],
  )
  const getSnapshot = useCallback(() => readThemeStateSnapshot(key, { lifetime, within }), [key, lifetime, within])
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, serverSnapshot)
  return useMemo(
    () => ({
      ready: snapshot.ready,
      value: snapshot.value,
      persisted: snapshot.persisted,
      set: (value: ThemeStateValue) => writeThemeStateValue(key, value, { lifetime, within }),
      remove: () => removeThemeStateValue(key, { lifetime, within }),
    }),
    [snapshot, key, lifetime, within],
  )
}
