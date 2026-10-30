/**
 * PRD-268 §6.4, K7, K11 (US-C21b) — the browser's day, and where that day's brief is.
 *
 * `browserDay()` is what POST /api/proactive/brief/open asks for: the local date, the IANA time
 * zone and the browser's language (the server takes it before Accept-Language, else English).
 *
 * The server keeps one brief per person, workspace and local date, and an open answers where it
 * is. The tab remembers that answer per workspace and date in localStorage, so a reload reads the
 * day's brief without opening again. Storage can be blocked (a private window, cleared site data):
 * every read and write is wrapped, a blocked one reads as nothing, and the brief is opened again.
 */
import { isProactiveId } from './wire'
import type { BriefDay, BriefMessageRef } from './types'

const STORAGE_PREFIX = 'automatos:brief:'
/** When neither the browser's clock nor Intl names a zone. */
const FALLBACK_TIME_ZONE = 'UTC'

const pad = (value: number): string => String(value).padStart(2, '0')

/** The local date of `now`, YYYY-MM-DD. */
export function localDateOf(now: Date): string {
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
}

function browserTimeZone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || FALLBACK_TIME_ZONE
  } catch {
    return FALLBACK_TIME_ZONE
  }
}

/** The body of POST /api/proactive/brief/open for this browser, now. */
export function browserDay(now: Date = new Date()): BriefDay {
  const day = { local_date: localDateOf(now), timezone: browserTimeZone() }
  const language = typeof navigator !== 'undefined' ? navigator.language : ''
  return language ? { ...day, locale: language } : day
}

function storageKey(workspaceId: string, localDate: string): string {
  return `${STORAGE_PREFIX}${workspaceId}:${localDate}`
}

/** Where the workspace's brief for `localDate` is, as this browser last heard; null when unknown. */
export function rememberedBrief(workspaceId: string | null, localDate: string): BriefMessageRef | null {
  if (workspaceId === null) return null
  try {
    const stored: unknown = JSON.parse(window.localStorage.getItem(storageKey(workspaceId, localDate)) ?? 'null')
    const ref = stored !== null && typeof stored === 'object' ? (stored as Partial<BriefMessageRef>) : null
    return ref && isProactiveId(ref.chat_id) && isProactiveId(ref.message_id)
      ? { chat_id: ref.chat_id, message_id: ref.message_id }
      : null
  } catch {
    return null
  }
}

/** The workspace's keys for other days: a past day's brief is never read again. */
function otherDays(storage: Storage, workspaceId: string, localDate: string): string[] {
  const keep = storageKey(workspaceId, localDate)
  const prefix = storageKey(workspaceId, '')
  return Array.from({ length: storage.length }, (_, index) => storage.key(index))
    .filter((key): key is string => key !== null && key.startsWith(prefix) && key !== keep)
}

/** Forget where the workspace's brief for `localDate` is (its message is gone). A blocked write is dropped. */
export function forgetBrief(workspaceId: string | null, localDate: string): void {
  if (workspaceId === null) return
  try {
    window.localStorage.removeItem(storageKey(workspaceId, localDate))
  } catch {
    // Storage blocked: nothing was remembered.
  }
}

/** Remember where the workspace's brief for `localDate` is, forgetting its other days. A blocked write is dropped. */
export function rememberBrief(workspaceId: string | null, localDate: string, ref: BriefMessageRef): void {
  if (workspaceId === null) return
  try {
    const storage = window.localStorage
    otherDays(storage, workspaceId, localDate).forEach((key) => storage.removeItem(key))
    storage.setItem(storageKey(workspaceId, localDate), JSON.stringify(ref))
  } catch {
    // Storage blocked: the tab opens the brief again on its next load.
  }
}
