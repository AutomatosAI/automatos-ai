/**
 * Widget chat session persistence (PRD-220).
 *
 * The Auto widget remounts on every navigation (MainLayout is instantiated
 * per page), so which conversation it was in must live outside React. Only
 * identifiers and timestamps are stored — never message content.
 *
 * All updaters return new objects; nothing here mutates its inputs.
 */

/** Resume window — an active thread older than this starts fresh (S1). */
export const WIDGET_SESSION_TTL_MS = 24 * 60 * 60 * 1000

/** Thread rows shown in the switcher; older threads live in full chat (S2). */
export const MAX_WIDGET_THREADS = 5

/** Cap on tracked read/closed entries so the storage entry can't grow unbounded. */
const MAX_TRACKED_THREADS = 50

const STORAGE_PREFIX = 'automatos:widget:chat'

export interface WidgetChatSession {
  /** Conversation the widget reconnects to on mount; null = start fresh. */
  activeChatId: string | null
  /** Last interaction (ms epoch) — drives the 24h resume window. */
  lastActiveAt: number
  /** Per-thread last-viewed timestamps (ms epoch) — drives unread badges. */
  lastReadAt: Record<string, number>
  /** Threads hidden from the switcher (conversation itself is never deleted). */
  closedThreadIds: string[]
}

export const EMPTY_WIDGET_SESSION: WidgetChatSession = Object.freeze({
  activeChatId: null,
  lastActiveAt: 0,
  lastReadAt: {},
  closedThreadIds: [],
})

/** Storage key scoped to workspace + user so sessions never leak across either. */
export function widgetSessionKey(
  workspaceId: string | null | undefined,
  userId: string | null | undefined
): string {
  return `${STORAGE_PREFIX}:${workspaceId || 'default'}:${userId || 'anon'}`
}

function isValidSession(value: unknown): value is WidgetChatSession {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  if (v.activeChatId !== null && typeof v.activeChatId !== 'string') return false
  if (typeof v.lastActiveAt !== 'number' || !Number.isFinite(v.lastActiveAt)) return false
  if (typeof v.lastReadAt !== 'object' || v.lastReadAt === null) return false
  if (!Object.values(v.lastReadAt as Record<string, unknown>).every((t) => typeof t === 'number')) return false
  if (!Array.isArray(v.closedThreadIds)) return false
  if (!(v.closedThreadIds as unknown[]).every((id) => typeof id === 'string')) return false
  return true
}

export function loadWidgetSession(key: string, now: number = Date.now()): WidgetChatSession {
  if (typeof window === 'undefined') return EMPTY_WIDGET_SESSION
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return EMPTY_WIDGET_SESSION
    const parsed: unknown = JSON.parse(raw)
    if (!isValidSession(parsed)) return EMPTY_WIDGET_SESSION
    // Only the resume pointer expires; read markers are kept so unread
    // badges stay honest across the expiry.
    if (parsed.activeChatId && now - parsed.lastActiveAt > WIDGET_SESSION_TTL_MS) {
      return { ...parsed, activeChatId: null }
    }
    return parsed
  } catch {
    return EMPTY_WIDGET_SESSION
  }
}

export function saveWidgetSession(key: string, session: WidgetChatSession): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, JSON.stringify(session))
  } catch {
    // Quota exceeded / private mode — the widget degrades to ephemeral.
  }
}

/** Switch to (or land in) a thread — also reopens it if previously closed. */
export function withActiveChat(
  session: WidgetChatSession,
  chatId: string,
  now: number = Date.now()
): WidgetChatSession {
  return {
    ...session,
    activeChatId: chatId,
    lastActiveAt: now,
    lastReadAt: capRecord({ ...session.lastReadAt, [chatId]: now }),
    closedThreadIds: session.closedThreadIds.filter((id) => id !== chatId),
  }
}

export function withoutActiveChat(
  session: WidgetChatSession,
  now: number = Date.now()
): WidgetChatSession {
  return { ...session, activeChatId: null, lastActiveAt: now }
}

export function withThreadRead(
  session: WidgetChatSession,
  chatId: string | null,
  now: number = Date.now()
): WidgetChatSession {
  if (!chatId) return session
  return {
    ...session,
    lastActiveAt: now,
    lastReadAt: capRecord({ ...session.lastReadAt, [chatId]: now }),
  }
}

/** Hide a thread from the switcher; clears the active pointer if it was active. */
export function withThreadClosed(
  session: WidgetChatSession,
  chatId: string,
  now: number = Date.now()
): WidgetChatSession {
  const closedThreadIds = [
    chatId,
    ...session.closedThreadIds.filter((id) => id !== chatId),
  ].slice(0, MAX_TRACKED_THREADS)
  const next = { ...session, closedThreadIds }
  return session.activeChatId === chatId
    ? { ...next, activeChatId: null, lastActiveAt: now }
    : next
}

/**
 * Unread = the thread moved since the user last viewed it in the widget.
 * Threads never opened in the widget have no baseline and show no badge.
 */
export function isThreadUnread(
  session: WidgetChatSession,
  thread: { id: string; updatedAt: string }
): boolean {
  if (thread.id === session.activeChatId) return false
  const lastRead = session.lastReadAt[thread.id]
  if (!lastRead) return false
  const updated = Date.parse(thread.updatedAt)
  return Number.isFinite(updated) && updated > lastRead
}

/** Switcher rows: closed threads filtered out, capped at `max`. */
export function visibleThreads<T extends { id: string }>(
  threads: T[],
  session: WidgetChatSession,
  max: number = MAX_WIDGET_THREADS
): T[] {
  const closed = new Set(session.closedThreadIds)
  return threads.filter((t) => !closed.has(t.id)).slice(0, max)
}

/** Compact relative timestamp for thread rows ("now", "5m", "3h", "2d"). */
export function threadTimeAgo(iso: string, now: number = Date.now()): string {
  const then = Date.parse(iso)
  if (!Number.isFinite(then)) return ''
  const diffMinutes = Math.floor((now - then) / 60_000)
  if (diffMinutes < 1) return 'now'
  if (diffMinutes < 60) return `${diffMinutes}m`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h`
  return `${Math.floor(diffHours / 24)}d`
}

function capRecord(record: Record<string, number>): Record<string, number> {
  const keys = Object.keys(record)
  if (keys.length <= MAX_TRACKED_THREADS) return record
  const keep = keys.sort((a, b) => record[b] - record[a]).slice(0, MAX_TRACKED_THREADS)
  return Object.fromEntries(keep.map((k) => [k, record[k]]))
}
