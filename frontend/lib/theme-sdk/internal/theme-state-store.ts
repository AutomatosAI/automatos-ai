/**
 * PRD-261 E3a — the theme-state store: the rules of the SDK's `storage` area (plan §2.2.1) as pure
 * functions over a Storage-like object the caller hands in; nothing here reads `window` (the E3b
 * runtime picks local or session storage, guards it and falls back to memory). One blob per theme,
 * lifetime and scope, `automatos-theme-state:v1:<b|s>:<themeId>:<scope>` (scope `u:<user>` or
 * `w:<workspace>:u:<user>`), holding `{"v":1,"at":<epoch ms>,"data":{<key>: <value>}}`. Values are
 * JSON only, capped by THEME_STATE_LIMITS (chars are UTF-16 units of the JSON text; depth counts
 * nested containers, `[[1]]` is 2). A blob that does not read back (corrupt, foreign, over a cap)
 * reads empty and the next write replaces it. Reads are deeply frozen and, for the same raw text,
 * the same objects (useSyncExternalStore). Every theme runs on the host's origin: this scopes, it
 * does not isolate.
 */

/** A value a theme may keep: JSON only. */
export type ThemeStateValue =
  | null | boolean | number | string
  | readonly ThemeStateValue[] | { readonly [k: string]: ThemeStateValue }

export type ThemeStateLifetime = 'browser' | 'session'
export type ThemeStateWithin = 'account' | 'workspace'
export type ThemeStateFailureReason = 'not-ready' | 'bad-key' | 'bad-value' | 'too-large' | 'unavailable'
export interface ThemeStateFailure { readonly ok: false; readonly reason: ThemeStateFailureReason; readonly message: string }

/** One blob's keys and values, deeply frozen. */
export type ThemeStateData = Readonly<Record<string, ThemeStateValue>>

/** What the store needs from a Storage: the five members both browser storages have. */
export type ThemeStateStorage = Pick<Storage, 'length' | 'key' | 'getItem' | 'setItem' | 'removeItem'>

/** Who the state belongs to: the user (`local` in the local edition) and, once known, the workspace. */
export interface ThemeStateScopeIds { readonly userKey: string; readonly workspaceKey: string | null }

/** A blob as read: its raw text (null when there is none), its write time (null when unreadable) and its data. */
export interface ThemeStateBlob { readonly raw: string | null; readonly at: number | null; readonly data: ThemeStateData }

/** What a read of one blob or one key gives back. */
export type ThemeStateBlobRead = ({ readonly ok: true } & ThemeStateBlob) | ThemeStateFailure
export type ThemeStateEntryRead = { readonly ok: true; readonly value: ThemeStateValue | null } | ThemeStateFailure
/** A write worked out but not made: the blob's new text and the other blobs it pushes out. */
export type ThemeStateWritePlan = { readonly ok: true; readonly raw: string; readonly evict: readonly string[] } | ThemeStateFailure
/** A write or remove made: the blob's text now (null when removed) and the blobs evicted. */
export type ThemeStateStoreResult = { readonly ok: true; readonly raw: string | null; readonly evicted: readonly string[] } | ThemeStateFailure

export const THEME_STATE_LIMITS = Object.freeze({
  keyPattern: '^[a-z][a-z0-9-]{0,47}$',
  keysPerBlob: 32,
  valueChars: 32768,
  blobChars: 65536,
  depth: 8,
  blobsPerLifetime: 8,
} as const)

/** Only keys with exactly this prefix are theme state; it overlaps no other host key. */
export const THEME_STATE_KEY_PREFIX = 'automatos-theme-state:v1:'
export const THEME_STATE_BLOB_VERSION = 1
export const DEFAULT_THEME_STATE_LIFETIME: ThemeStateLifetime = 'browser'
export const DEFAULT_THEME_STATE_WITHIN: ThemeStateWithin = 'account'

const LIFETIME_LETTERS: Readonly<Record<ThemeStateLifetime, string>> = Object.freeze({ browser: 'b', session: 's' })
const KEY_RE = new RegExp(THEME_STATE_LIMITS.keyPattern)
const SCOPE_SEPARATOR = ':'
/** Parsed blobs kept by raw text: both lifetimes' blobs, twice over. */
const PARSE_CACHE_SIZE = THEME_STATE_LIMITS.blobsPerLifetime * 4
const EMPTY_DATA: ThemeStateData = Object.freeze({})
const NONE: readonly string[] = Object.freeze([])
const parseCache = new Map<string, Omit<ThemeStateBlob, 'raw'>>()

function failure(reason: ThemeStateFailureReason, message: string): ThemeStateFailure {
  return Object.freeze({ ok: false as const, reason, message })
}

function isScopePart(part: unknown): part is string {
  return typeof part === 'string' && part.length > 0 && part.indexOf(SCOPE_SEPARATOR) === -1
}

/** The scope part of the key, or null when an id is missing (no workspace yet) or holds a ':'. */
export function themeStateScope(ids: ThemeStateScopeIds, within: ThemeStateWithin): string | null {
  if (!isScopePart(ids.userKey)) return null
  const user = `u:${ids.userKey}`
  if (within === 'account') return user
  return isScopePart(ids.workspaceKey) ? `w:${ids.workspaceKey}:${user}` : null
}

/** The storage key of one theme's blob, or null when the theme id is empty or holds a ':'. */
export function themeStateStorageKey(lifetime: ThemeStateLifetime, themeId: string, scope: string): string | null {
  if (!isScopePart(themeId) || scope.length === 0) return null
  return `${THEME_STATE_KEY_PREFIX}${LIFETIME_LETTERS[lifetime]}:${themeId}:${scope}`
}

/** A theme's own key: a lowercase letter, then up to 47 lowercase letters, digits or hyphens. */
export function isThemeStateKey(key: unknown): key is string {
  return typeof key === 'string' && KEY_RE.test(key)
}

function badKey(): ThemeStateFailure {
  return failure('bad-key', `A theme state key is a lowercase letter, then lowercase letters, digits or hyphens (${THEME_STATE_LIMITS.keyPattern}).`)
}

function isPlainObject(value: object): boolean {
  const proto = Object.getPrototypeOf(value)
  return proto === Object.prototype || proto === null
}

function arrayProblem(value: readonly unknown[], depth: number, ancestors: Set<object>): string | null {
  if (Object.keys(value).length !== value.length) return 'an array has holes or named properties'
  for (let index = 0; index < value.length; index += 1) {
    const problem = valueProblem(value[index], depth, ancestors)
    if (problem) return problem
  }
  return null
}

function objectProblem(value: object, depth: number, ancestors: Set<object>): string | null {
  if (!isPlainObject(value)) return 'only plain objects and arrays are kept (not a Date, Map, Set or class instance)'
  if (Object.getOwnPropertySymbols(value).length > 0) return 'an object has symbol keys'
  for (const name of Object.keys(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, name)
    if (!descriptor || !('value' in descriptor)) return 'an object has a getter'
    const problem = valueProblem(descriptor.value, depth, ancestors)
    if (problem) return problem
  }
  return null
}

/** Why `value` is not JSON within the depth cap, or null when it is. `depth` counts the containers above it. */
function valueProblem(value: unknown, depth: number, ancestors: Set<object>): string | null {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return null
  if (typeof value === 'number') return Number.isFinite(value) ? null : 'numbers must be finite'
  if (typeof value !== 'object') return `${typeof value} values are not JSON`
  if (ancestors.has(value)) return 'the value refers to itself'
  if (depth >= THEME_STATE_LIMITS.depth) return `values nest at most ${THEME_STATE_LIMITS.depth} deep`
  ancestors.add(value)
  const problem = Array.isArray(value) ? arrayProblem(value, depth + 1, ancestors) : objectProblem(value, depth + 1, ancestors)
  ancestors.delete(value)
  return problem
}

/** The value as JSON text when it may be kept; never throws, whatever the value. */
export function checkThemeStateValue(value: unknown): { readonly ok: true; readonly text: string } | ThemeStateFailure {
  let problem: string | null
  let text = ''
  try {
    problem = valueProblem(value, 0, new Set())
    if (!problem) text = JSON.stringify(value)
  } catch {
    problem = 'the value could not be read'
  }
  if (problem) return failure('bad-value', `Theme state keeps JSON values only: ${problem}.`)
  if (text.length > THEME_STATE_LIMITS.valueChars) {
    return failure('too-large', `A theme state value is at most ${THEME_STATE_LIMITS.valueChars} characters as JSON; this one is ${text.length}.`)
  }
  return Object.freeze({ ok: true as const, text })
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(deepFreeze)
    Object.freeze(value)
  }
  return value
}

function isValidData(data: unknown): data is Record<string, ThemeStateValue> {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) return false
  const keys = Object.keys(data)
  if (keys.length > THEME_STATE_LIMITS.keysPerBlob) return false
  const entries = data as Record<string, unknown>
  return keys.every((key) => isThemeStateKey(key) && checkThemeStateValue(entries[key]).ok)
}

/** The JSON text's value, or undefined when it is not JSON. */
function parseJson(raw: string): unknown {
  try {
    return JSON.parse(raw)
  } catch {
    return undefined
  }
}

function parseBlobText(raw: string): Omit<ThemeStateBlob, 'raw'> {
  const unreadable = { at: null, data: EMPTY_DATA }
  if (raw.length > THEME_STATE_LIMITS.blobChars) return unreadable
  const parsed = parseJson(raw)
  if (parsed === null || typeof parsed !== 'object') return unreadable
  const { v, at, data } = parsed as { v?: unknown; at?: unknown; data?: unknown }
  const readable = v === THEME_STATE_BLOB_VERSION && typeof at === 'number' && Number.isFinite(at) && isValidData(data)
  return readable ? { at: at as number, data: deepFreeze(data as ThemeStateData) } : unreadable
}

/** A blob's text as data: empty when null or unreadable; the same frozen objects for the same text. */
export function parseThemeStateBlob(raw: string | null): ThemeStateBlob {
  if (raw === null) return Object.freeze({ raw, at: null, data: EMPTY_DATA })
  const parsed = parseCache.get(raw) || Object.freeze(parseBlobText(raw))
  parseCache.delete(raw)
  parseCache.set(raw, parsed)
  if (parseCache.size > PARSE_CACHE_SIZE) parseCache.delete(parseCache.keys().next().value as string)
  return Object.freeze({ raw, ...parsed })
}

function unavailable(error: unknown): ThemeStateFailure {
  const name = error instanceof Error ? error.name : 'an error'
  return failure('unavailable', `This browser's storage could not be used (${name}).`)
}

/** One blob as stored now. */
export function readThemeStateBlob(storage: ThemeStateStorage, storageKey: string): ThemeStateBlobRead {
  try {
    return Object.freeze({ ok: true as const, ...parseThemeStateBlob(storage.getItem(storageKey)) })
  } catch (error) {
    return unavailable(error)
  }
}

/** One theme key's value, or null when it has none. */
export function readThemeStateEntry(storage: ThemeStateStorage, storageKey: string, key: string): ThemeStateEntryRead {
  if (!isThemeStateKey(key)) return badKey()
  const blob = readThemeStateBlob(storage, storageKey)
  if (!blob.ok) return blob
  const value = Object.prototype.hasOwnProperty.call(blob.data, key) ? blob.data[key] : null
  return Object.freeze({ ok: true as const, value })
}

function blobText(data: Readonly<Record<string, unknown>>, now: number): string {
  return JSON.stringify({ v: THEME_STATE_BLOB_VERSION, at: now, data })
}

/** Theme-state blobs in `storage` other than `storageKey`, oldest write first (unreadable ones before any). */
function otherBlobsOldestFirst(storage: ThemeStateStorage, storageKey: string): string[] {
  const others: { key: string; at: number }[] = []
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index)
    if (key === null || key === storageKey || !key.startsWith(THEME_STATE_KEY_PREFIX)) continue
    const at = parseThemeStateBlob(storage.getItem(key)).at
    others.push({ key, at: at === null ? Number.NEGATIVE_INFINITY : at })
  }
  others.sort((a, b) => a.at - b.at || (a.key < b.key ? -1 : 1))
  return others.map((other) => other.key)
}

function evictionFor(storage: ThemeStateStorage, storageKey: string): string[] {
  const others = otherBlobsOldestFirst(storage, storageKey)
  const over = others.length + 1 - THEME_STATE_LIMITS.blobsPerLifetime
  return over > 0 ? others.slice(0, over) : []
}

function nextBlob(data: ThemeStateData, key: string, value: unknown, now: number): string | ThemeStateFailure {
  const isNew = !Object.prototype.hasOwnProperty.call(data, key)
  if (isNew && Object.keys(data).length >= THEME_STATE_LIMITS.keysPerBlob) {
    return failure('too-large', `A theme keeps at most ${THEME_STATE_LIMITS.keysPerBlob} keys here; '${key}' would be one more.`)
  }
  const raw = blobText({ ...data, [key]: value }, now)
  if (raw.length > THEME_STATE_LIMITS.blobChars) {
    return failure('too-large', `A theme keeps at most ${THEME_STATE_LIMITS.blobChars} characters here; this write would make ${raw.length}.`)
  }
  return raw
}

/** Works out a write without making it: the blob's new text and which other blobs make room for it. */
export function planThemeStateWrite(storage: ThemeStateStorage, storageKey: string, key: string, value: unknown, now: number): ThemeStateWritePlan {
  if (!isThemeStateKey(key)) return badKey()
  const checked = checkThemeStateValue(value)
  if (!checked.ok) return checked
  try {
    const raw = nextBlob(parseThemeStateBlob(storage.getItem(storageKey)).data, key, value, now)
    if (typeof raw !== 'string') return raw
    return Object.freeze({ ok: true as const, raw, evict: Object.freeze(evictionFor(storage, storageKey)) })
  } catch (error) {
    return unavailable(error)
  }
}

/** Sets one key: writes the blob, then evicts the oldest other blobs over the per-lifetime cap. */
export function writeThemeStateEntry(storage: ThemeStateStorage, storageKey: string, key: string, value: unknown, now: number): ThemeStateStoreResult {
  const plan = planThemeStateWrite(storage, storageKey, key, value, now)
  if (!plan.ok) return plan
  try {
    storage.setItem(storageKey, plan.raw)
    plan.evict.forEach((evicted) => storage.removeItem(evicted))
  } catch (error) {
    return unavailable(error)
  }
  return Object.freeze({ ok: true as const, raw: plan.raw, evicted: plan.evict })
}

/** Removes one key; the blob goes with its last key. A key that is not there writes nothing. */
export function removeThemeStateEntry(storage: ThemeStateStorage, storageKey: string, key: string, now: number): ThemeStateStoreResult {
  if (!isThemeStateKey(key)) return badKey()
  const blob = readThemeStateBlob(storage, storageKey)
  if (!blob.ok) return blob
  if (!Object.prototype.hasOwnProperty.call(blob.data, key)) return Object.freeze({ ok: true as const, raw: blob.raw, evicted: NONE })
  const rest = Object.fromEntries(Object.entries(blob.data).filter(([name]) => name !== key))
  const raw = Object.keys(rest).length === 0 ? null : blobText(rest, now)
  try {
    if (raw === null) storage.removeItem(storageKey)
    else storage.setItem(storageKey, raw)
  } catch (error) {
    return unavailable(error)
  }
  return Object.freeze({ ok: true as const, raw, evicted: NONE })
}
