/**
 * PRD-261 E3b — the theme-state runtime: what the SDK's `storage` area (lib/theme-sdk/storage.ts) reads
 * and writes through, over the pure store (./theme-state-store.ts, plan §2.2.1).
 *
 * - **The scope.** The host publishes `{themeId, userKey, workspaceKey}` while an installed theme runs
 *   (E3c, components/theme-host/theme-state-scope.tsx) and clears it otherwise. With none published
 *   (Classic, Studio, safe mode, an isolated mount) reads are not ready, writes refuse `not-ready` and
 *   nothing touches the browser's storage. On the server (no `window`) reads are not ready and writes
 *   refuse `unavailable`, whatever was published.
 * - **The storage.** `browser` is localStorage, `session` sessionStorage, reached only inside a guard.
 *   Storage that refuses (blocked site data, a private window, a full quota) is swapped for an
 *   in-memory one for the rest of the page, seeded with what could still be read; writes then answer
 *   `{ok: true, persisted: false}` and one `console.warn` is written in all.
 * - **Snapshots.** A key's snapshot is frozen and stays the same object while its value is the same,
 *   so `useSyncExternalStore` never loops and a re-render is only for a real change.
 * - **Subscribers.** Per key; one `storage` listener while any exist. Another tab's change notifies
 *   only the subscribers whose value changed; a `key: null` event (storage cleared) and a scope
 *   change notify all.
 *
 * Every theme runs on the host's origin: this scopes theme state, it does not isolate it.
 */
import {
  DEFAULT_THEME_STATE_LIFETIME,
  DEFAULT_THEME_STATE_WITHIN,
  isThemeStateKey,
  readThemeStateBlob,
  removeThemeStateEntry,
  THEME_STATE_KEY_PREFIX,
  THEME_STATE_LIMITS,
  themeStateScope,
  themeStateStorageKey,
  writeThemeStateEntry,
  type ThemeStateFailure,
  type ThemeStateFailureReason,
  type ThemeStateLifetime,
  type ThemeStateStorage,
  type ThemeStateStoreResult,
  type ThemeStateValue,
  type ThemeStateWithin,
} from './theme-state-store'

/** Whose theme state is read and written now: the installed theme, the user and (once known) the workspace. */
export interface ThemeStateRuntimeScope {
  readonly themeId: string
  readonly userKey: string
  readonly workspaceKey: string | null
}

/** One key's state as the SDK hands it out: frozen, the same object while nothing changed. */
export interface ThemeStateSnapshot {
  readonly ready: boolean
  readonly value: ThemeStateValue | null
  /** False while the value lives only in this page's memory (blocked storage) or nothing can be kept. */
  readonly persisted: boolean
}

export type ThemeStateRuntimeResult = { readonly ok: true; readonly persisted: boolean } | ThemeStateFailure

interface ResolvedOptions {
  readonly lifetime: ThemeStateLifetime
  readonly within: ThemeStateWithin
}

interface Subscriber {
  readonly key: string
  readonly options: ResolvedOptions
  readonly listener: () => void
}

interface CachedSnapshot {
  readonly raw: string | null
  readonly valueText: string
  readonly snapshot: ThemeStateSnapshot
}

const BROWSER_STORAGES: Readonly<Record<ThemeStateLifetime, 'localStorage' | 'sessionStorage'>> = Object.freeze({
  browser: 'localStorage',
  session: 'sessionStorage',
})
const LIFETIMES: readonly string[] = Object.freeze(Object.keys(BROWSER_STORAGES))
const WITHINS: readonly string[] = Object.freeze(['account', 'workspace'])
const STORAGE_EVENT = 'storage'
/** Every key of every blob in both lifetimes: the most snapshots a page can be holding. */
const SNAPSHOT_CACHE_SIZE = THEME_STATE_LIMITS.keysPerBlob * THEME_STATE_LIMITS.blobsPerLifetime * LIFETIMES.length
const BLOCKED_WARNING = "Automatos theme SDK: this browser's storage is blocked, so theme state is kept for this page only."

export const NOT_READY_SNAPSHOT: ThemeStateSnapshot = Object.freeze({ ready: false, value: null, persisted: false })
const BAD_KEY_SNAPSHOT: ThemeStateSnapshot = Object.freeze({ ready: true, value: null, persisted: false })
const NOOP_UNSUBSCRIBE = (): void => undefined

function failure(reason: ThemeStateFailureReason, message: string): ThemeStateFailure {
  return Object.freeze({ ok: false as const, reason, message })
}

const NOT_READY = failure('not-ready', "Theme state isn't ready here: only an installed theme, once its user is known, keeps state.")
const ON_THE_SERVER = failure('unavailable', 'Theme state is kept in the browser; there is none on the server.')
const BAD_OPTIONS = failure('bad-key', "Theme state options are lifetime 'browser' or 'session' and within 'account' or 'workspace'.")

let scope: ThemeStateRuntimeScope | null = null
let memoryStorages: Readonly<Partial<Record<ThemeStateLifetime, ThemeStateStorage>>> = Object.freeze({})
let warned = false
let listening = false
const subscribers = new Map<Subscriber, ThemeStateSnapshot>()
const snapshots = new Map<string, CachedSnapshot>()

function onTheServer(): boolean {
  return typeof window === 'undefined'
}

/** The theme's options with the defaults filled in, or null when they are not the SDK's options. */
function resolveThemeStateOptions(options: unknown): ResolvedOptions | null {
  if (options === undefined || options === null) return { lifetime: DEFAULT_THEME_STATE_LIFETIME, within: DEFAULT_THEME_STATE_WITHIN }
  if (typeof options !== 'object') return null
  const { lifetime = DEFAULT_THEME_STATE_LIFETIME, within = DEFAULT_THEME_STATE_WITHIN } = options as { lifetime?: unknown; within?: unknown }
  if (typeof lifetime !== 'string' || !LIFETIMES.includes(lifetime)) return null
  if (typeof within !== 'string' || !WITHINS.includes(within)) return null
  return { lifetime: lifetime as ThemeStateLifetime, within: within as ThemeStateWithin }
}

/** The storage key of the current scope's blob for these options, or null while the scope can't name one. */
function currentStorageKey(options: ResolvedOptions): string | null {
  if (scope === null) return null
  const part = themeStateScope(scope, options.within)
  return part === null ? null : themeStateStorageKey(options.lifetime, scope.themeId, part)
}

/** An in-memory Storage, in insertion order like the browser's. */
function memoryStorage(seed: ReadonlyMap<string, string>): ThemeStateStorage {
  const items = new Map(seed)
  return {
    get length() {
      return items.size
    },
    key: (index: number) => Array.from(items.keys())[index] ?? null,
    getItem: (key: string) => (items.has(key) ? (items.get(key) as string) : null),
    setItem: (key: string, value: string) => {
      items.set(String(key), String(value))
    },
    removeItem: (key: string) => {
      items.delete(key)
    },
  }
}

/** The theme-state items `storage` still lets us read; none when it refuses. */
function readableThemeState(storage: ThemeStateStorage | null): Map<string, string> {
  const found = new Map<string, string>()
  if (storage === null) return found
  try {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index)
      const raw = key !== null && key.startsWith(THEME_STATE_KEY_PREFIX) ? storage.getItem(key) : null
      if (key !== null && raw !== null) found.set(key, raw)
    }
  } catch {
    // The storage refuses reads too: the page starts from what was found before it refused.
  }
  return found
}

/** Swaps the lifetime's storage for memory for the rest of the page; warns once in all. */
function fallBackToMemory(lifetime: ThemeStateLifetime, refused: ThemeStateStorage | null): ThemeStateStorage {
  const memory = memoryStorage(readableThemeState(refused))
  memoryStorages = Object.freeze({ ...memoryStorages, [lifetime]: memory })
  if (!warned) {
    warned = true
    console.warn(BLOCKED_WARNING)
  }
  return memory
}

function browserStorage(lifetime: ThemeStateLifetime): ThemeStateStorage | null {
  try {
    return window[BROWSER_STORAGES[lifetime]] ?? null
  } catch {
    return null
  }
}

/** Runs `operation` on the lifetime's storage, and again on memory when the browser's refuses it. */
function onStorage<R extends { readonly ok: boolean }>(
  lifetime: ThemeStateLifetime,
  operation: (storage: ThemeStateStorage) => R | ThemeStateFailure,
): { readonly result: R | ThemeStateFailure; readonly persisted: boolean } {
  const memory = memoryStorages[lifetime]
  if (memory) return { result: operation(memory), persisted: false }
  const storage = browserStorage(lifetime)
  if (storage === null) return { result: operation(fallBackToMemory(lifetime, null)), persisted: false }
  const result = operation(storage)
  const refused = !result.ok && (result as ThemeStateFailure).reason === 'unavailable'
  return refused ? { result: operation(fallBackToMemory(lifetime, storage)), persisted: false } : { result, persisted: true }
}

/** The cached snapshot while the key's value and persistence are unchanged, else a new one (cached). */
function snapshotOf(storageKey: string, key: string, raw: string | null, value: ThemeStateValue | null, persisted: boolean): ThemeStateSnapshot {
  const id = `${storageKey}\n${key}`
  const cached = snapshots.get(id)
  if (cached && cached.raw === raw && cached.snapshot.persisted === persisted) return cached.snapshot
  const valueText = JSON.stringify(value)
  const same = cached !== undefined && cached.valueText === valueText && cached.snapshot.persisted === persisted
  const snapshot = same ? cached.snapshot : Object.freeze({ ready: true, value, persisted })
  snapshots.delete(id)
  snapshots.set(id, { raw, valueText, snapshot })
  if (snapshots.size > SNAPSHOT_CACHE_SIZE) snapshots.delete(snapshots.keys().next().value as string)
  return snapshot
}

function snapshotFor(key: string, options: ResolvedOptions): ThemeStateSnapshot {
  const storageKey = currentStorageKey(options)
  if (storageKey === null) return NOT_READY_SNAPSHOT
  if (!isThemeStateKey(key)) return BAD_KEY_SNAPSHOT
  const { result, persisted } = onStorage(options.lifetime, (storage) => readThemeStateBlob(storage, storageKey))
  if (!result.ok) return NOT_READY_SNAPSHOT
  const value = Object.prototype.hasOwnProperty.call(result.data, key) ? result.data[key] : null
  return snapshotOf(storageKey, key, result.raw, value, persisted)
}

/** One key's state now: not ready on the server, with no scope, or while the scope can't name the blob. */
export function readThemeStateSnapshot(key: string, options?: unknown): ThemeStateSnapshot {
  if (onTheServer()) return NOT_READY_SNAPSHOT
  const resolved = resolveThemeStateOptions(options)
  return resolved === null ? NOT_READY_SNAPSHOT : snapshotFor(key, resolved)
}

function notify(subscriber: Subscriber): void {
  try {
    subscriber.listener()
  } catch (error) {
    console.error(`Automatos theme SDK: a theme state listener for '${subscriber.key}' threw:`, error)
  }
}

/** Tells every subscriber, as after a scope change or a cleared storage. */
function notifyAll(): void {
  Array.from(subscribers.keys()).forEach((subscriber) => {
    subscribers.set(subscriber, snapshotFor(subscriber.key, subscriber.options))
    notify(subscriber)
  })
}

/** Tells the subscribers whose key lives in one of `storageKeys` and whose value changed. */
function notifyChanged(storageKeys: readonly string[]): void {
  Array.from(subscribers.entries()).forEach(([subscriber, last]) => {
    const storageKey = currentStorageKey(subscriber.options)
    if (storageKey === null || !storageKeys.includes(storageKey)) return
    const next = snapshotFor(subscriber.key, subscriber.options)
    if (next === last) return
    subscribers.set(subscriber, next)
    notify(subscriber)
  })
}

function onStorageEvent(event: StorageEvent): void {
  if (event.key === null) notifyAll()
  else if (event.key.startsWith(THEME_STATE_KEY_PREFIX)) notifyChanged([event.key])
}

function listenForOtherTabs(listen: boolean): void {
  if (listen === listening) return
  listening = listen
  if (listen) window.addEventListener(STORAGE_EVENT, onStorageEvent)
  else window.removeEventListener(STORAGE_EVENT, onStorageEvent)
}

/** Calls `listener` whenever the key's state may have changed, until the returned function is called. */
export function subscribeToThemeState(key: string, listener: () => void, options?: unknown): () => void {
  const resolved = resolveThemeStateOptions(options)
  if (onTheServer() || resolved === null || typeof listener !== 'function') return NOOP_UNSUBSCRIBE
  const subscriber: Subscriber = { key, options: resolved, listener }
  subscribers.set(subscriber, snapshotFor(key, resolved))
  listenForOtherTabs(true)
  return () => {
    subscribers.delete(subscriber)
    if (subscribers.size === 0) listenForOtherTabs(false)
  }
}

/** A store write or remove, bound to its key and value, run against one blob. */
type StoreChange = (storage: ThemeStateStorage, storageKey: string, now: number) => ThemeStateStoreResult

/** Runs a store write or remove against the current scope's blob, then tells the subscribers it touched. */
function change(options: unknown, operation: StoreChange): ThemeStateRuntimeResult {
  if (onTheServer()) return ON_THE_SERVER
  const resolved = resolveThemeStateOptions(options)
  if (resolved === null) return BAD_OPTIONS
  const storageKey = currentStorageKey(resolved)
  if (storageKey === null) return NOT_READY
  const now = Date.now()
  const { result, persisted } = onStorage(resolved.lifetime, (storage) => operation(storage, storageKey, now))
  if (!result.ok) return result
  notifyChanged([storageKey, ...result.evicted])
  return Object.freeze({ ok: true as const, persisted })
}

export function writeThemeStateValue(key: string, value: unknown, options?: unknown): ThemeStateRuntimeResult {
  return change(options, (storage, storageKey, now) => writeThemeStateEntry(storage, storageKey, key, value, now))
}

export function removeThemeStateValue(key: string, options?: unknown): ThemeStateRuntimeResult {
  return change(options, (storage, storageKey, now) => removeThemeStateEntry(storage, storageKey, key, now))
}

/** The scope published now; null when none is. */
export function currentThemeStateScope(): ThemeStateRuntimeScope | null {
  return scope
}

function sameScope(a: ThemeStateRuntimeScope | null, b: ThemeStateRuntimeScope): boolean {
  return a !== null && a.themeId === b.themeId && a.userKey === b.userKey && a.workspaceKey === b.workspaceKey
}

/**
 * Makes `next` the current scope and tells every subscriber. Returns the cleanup that clears it, only if it
 * is still the current one, so a scope that replaced it is never dropped by the old one's cleanup.
 */
export function publishThemeStateScope(next: ThemeStateRuntimeScope): () => void {
  if (!sameScope(scope, next)) {
    scope = Object.freeze({ themeId: next.themeId, userKey: next.userKey, workspaceKey: next.workspaceKey })
    notifyAll()
  }
  const published = scope
  return () => {
    if (scope === published) clearThemeStateScope()
  }
}

/** Clears the scope (Classic, Studio, safe mode, the user gone) and tells every subscriber. */
export function clearThemeStateScope(): void {
  if (scope === null) return
  scope = null
  notifyAll()
}

/** Tests only: forgets the scope, the in-memory storages, the warning, the subscribers and the snapshots. */
export function resetThemeStateRuntimeForTests(): void {
  scope = null
  memoryStorages = Object.freeze({})
  warned = false
  subscribers.clear()
  snapshots.clear()
  if (!onTheServer()) listenForOtherTabs(false)
}
