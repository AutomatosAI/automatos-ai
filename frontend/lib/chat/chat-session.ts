/**
 * The conversation session (PRD-237 — generalises PRD-220's widget session).
 *
 * One record per workspace+user answers "which conversation am I in, and which
 * ones are open as tabs" for the chat page, the studio shell AND the floating
 * widget — so navigating, reloading or switching surface never starts a new
 * chat until the user asks for one (New Chat). Identifiers, timestamps and a
 * title cache only — never message content. No expiry.
 *
 * Every updater returns a new object; nothing here mutates its inputs. An
 * updater that changes nothing returns its input (reference-equal) so callers
 * can skip redundant saves.
 */

/** Tabs open at once; the oldest inactive tab is evicted beyond this. */
export const MAX_OPEN_CHATS = 8

/** Cap on tracked read/title entries so the storage entry can't grow unbounded. */
export const MAX_TRACKED_THREADS = 50

const STORAGE_PREFIX = 'automatos:chat:session'

export interface ChatSession {
  /** Conversation on screen; null = the "New chat" draft. */
  activeChatId: string | null
  /** A "New chat" draft tab exists (at most one). */
  draftOpen: boolean
  /** Ordered open tabs (conversation ids); the draft is never in here. */
  openChatIds: string[]
  /** Open tabs that moved while not on screen. Device-local. */
  unreadChatIds: string[]
  /** Per-thread last-viewed timestamps (ms epoch). */
  lastReadAt: Record<string, number>
  /** Tab labels; the server is the truth, this is a cache. Device-local. */
  titles: Record<string, string>
  /** Last local change (ms epoch) — decides who wins against the server copy. */
  lastActiveAt: number
}

/** The subset the hosted edition keeps server-side (PRD-237 S6). */
export interface ServerChatSession {
  activeChatId: string | null
  draftOpen: boolean
  openChatIds: string[]
  lastReadAt: Record<string, number>
  updatedAt?: string | null
}

export interface SessionTab {
  /** null = the draft tab. */
  id: string | null
  isDraft: boolean
}

export const EMPTY_CHAT_SESSION: ChatSession = Object.freeze({
  activeChatId: null,
  draftOpen: false,
  openChatIds: [],
  unreadChatIds: [],
  lastReadAt: {},
  titles: {},
  lastActiveAt: 0,
}) as ChatSession

/** Storage key scoped to workspace + user so sessions never leak across either. */
export function chatSessionKey(
  workspaceId: string | null | undefined,
  userId: string | null | undefined,
): string {
  return `${STORAGE_PREFIX}:${workspaceId || 'default'}:${userId || 'anon'}`
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === 'string')
}

function isRecordOf(value: unknown, type: 'number' | 'string'): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  return Object.values(value as Record<string, unknown>).every((v) => typeof v === type)
}

export function isValidChatSession(value: unknown): value is ChatSession {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Record<string, unknown>
  if (v.activeChatId !== null && typeof v.activeChatId !== 'string') return false
  if (typeof v.draftOpen !== 'boolean') return false
  if (!isStringArray(v.openChatIds) || !isStringArray(v.unreadChatIds)) return false
  if (!isRecordOf(v.lastReadAt, 'number') || !isRecordOf(v.titles, 'string')) return false
  if (typeof v.lastActiveAt !== 'number' || !Number.isFinite(v.lastActiveAt)) return false
  return true
}

export function loadChatSession(key: string): ChatSession {
  if (typeof window === 'undefined') return EMPTY_CHAT_SESSION
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return EMPTY_CHAT_SESSION
    const parsed: unknown = JSON.parse(raw)
    return isValidChatSession(parsed) ? parsed : EMPTY_CHAT_SESSION
  } catch {
    return EMPTY_CHAT_SESSION
  }
}

export function saveChatSession(key: string, session: ChatSession): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(key, JSON.stringify(session))
  } catch {
    // Quota exceeded / private mode — the session degrades to in-memory.
  }
}

/** The tab strip, in order: open conversations, then the draft (always at least one tab). */
export function sessionTabs(session: ChatSession): SessionTab[] {
  const tabs: SessionTab[] = session.openChatIds.map((id) => ({ id, isDraft: false }))
  if (session.draftOpen || tabs.length === 0) tabs.push({ id: null, isDraft: true })
  return tabs
}

export function isChatUnread(session: ChatSession, chatId: string): boolean {
  return session.unreadChatIds.includes(chatId)
}

function capRecord<T>(record: Record<string, T>, rank: (value: T) => number): Record<string, T> {
  const keys = Object.keys(record)
  if (keys.length <= MAX_TRACKED_THREADS) return record
  const keep = keys.sort((a, b) => rank(record[b]) - rank(record[a])).slice(0, MAX_TRACKED_THREADS)
  return Object.fromEntries(keep.map((k) => [k, record[k]]))
}

function without(list: string[], id: string): string[] {
  return list.includes(id) ? list.filter((x) => x !== id) : list
}

/** Drop the oldest tabs (front of the list) that are not protected until within the cap. */
function evictForCap(openChatIds: string[], protectedIds: ReadonlySet<string>): string[] {
  if (openChatIds.length <= MAX_OPEN_CHATS) return openChatIds
  const excess = openChatIds.length - MAX_OPEN_CHATS
  const evicted = new Set<string>()
  for (const id of openChatIds) {
    if (evicted.size === excess) break
    if (!protectedIds.has(id)) evicted.add(id)
  }
  return openChatIds.filter((id) => !evicted.has(id))
}

function stamp(record: Record<string, number>, chatId: string | null, now: number): Record<string, number> {
  if (!chatId) return record
  return capRecord({ ...record, [chatId]: now }, (t) => t)
}

/** Switch to (or land in) a conversation — opens its tab; an empty draft gives way. */
export function withChatOpened(session: ChatSession, chatId: string, now: number = Date.now()): ChatSession {
  const alreadyOpen = session.openChatIds.includes(chatId)
  if (alreadyOpen && session.activeChatId === chatId && !session.draftOpen) return session
  const previous = session.activeChatId
  // Never evict the tab being opened or the one the user was just on.
  const protectedIds = new Set([chatId, ...(previous ? [previous] : [])])
  const openChatIds = alreadyOpen
    ? session.openChatIds
    : evictForCap([...session.openChatIds, chatId], protectedIds)
  return {
    ...session,
    activeChatId: chatId,
    draftOpen: false,
    openChatIds,
    unreadChatIds: without(session.unreadChatIds, chatId),
    lastReadAt: stamp(stamp(session.lastReadAt, previous, now), chatId, now),
    lastActiveAt: now,
  }
}

/** New Chat: a single draft tab becomes active. Repeated presses are a no-op. */
export function withDraft(session: ChatSession, now: number = Date.now()): ChatSession {
  if (session.draftOpen && session.activeChatId === null) return session
  return {
    ...session,
    activeChatId: null,
    draftOpen: true,
    lastReadAt: stamp(session.lastReadAt, session.activeChatId, now),
    lastActiveAt: now,
  }
}

/** The backend named the conversation (first message of a draft, or a stale id healed). */
export function withChatIdAssigned(session: ChatSession, chatId: string, now: number = Date.now()): ChatSession {
  return withChatOpened(session, chatId, now)
}

/** Close a tab; the conversation itself is never deleted. Focus moves right, then left, then draft. */
export function withTabClosed(session: ChatSession, chatId: string, now: number = Date.now()): ChatSession {
  const index = session.openChatIds.indexOf(chatId)
  if (index === -1) return session
  const openChatIds = session.openChatIds.filter((id) => id !== chatId)
  const base: ChatSession = {
    ...session,
    openChatIds,
    unreadChatIds: without(session.unreadChatIds, chatId),
    lastActiveAt: now,
  }
  if (session.activeChatId !== chatId) return base
  const neighbour = openChatIds[index] ?? openChatIds[index - 1] ?? null
  if (neighbour === null) return { ...base, activeChatId: null, draftOpen: true }
  return withChatOpened({ ...base, activeChatId: null }, neighbour, now)
}

/** Close the draft tab (only when another tab remains — the strip is never empty). */
export function withDraftClosed(session: ChatSession, now: number = Date.now()): ChatSession {
  if (!session.draftOpen || session.openChatIds.length === 0) return session
  const next: ChatSession = { ...session, draftOpen: false, lastActiveAt: now }
  if (session.activeChatId !== null) return next
  return withChatOpened(next, session.openChatIds[session.openChatIds.length - 1], now)
}

export function withThreadRead(session: ChatSession, chatId: string | null, now: number = Date.now()): ChatSession {
  if (!chatId) return session
  return {
    ...session,
    unreadChatIds: without(session.unreadChatIds, chatId),
    lastReadAt: stamp(session.lastReadAt, chatId, now),
    lastActiveAt: now,
  }
}

/** Something landed in `chatId` (a finished turn, a background message) — flag it unless on screen. */
export function withChatTouched(session: ChatSession, chatId: string): ChatSession {
  if (chatId === session.activeChatId) return session
  if (!session.openChatIds.includes(chatId) || session.unreadChatIds.includes(chatId)) return session
  return { ...session, unreadChatIds: [...session.unreadChatIds, chatId] }
}

/** Merge tab labels; unchanged input returns the same session. */
export function withTitles(session: ChatSession, titles: Record<string, string>): ChatSession {
  const entries = Object.entries(titles).filter(([id, title]) => title && session.titles[id] !== title)
  if (entries.length === 0) return session
  const merged = { ...session.titles, ...Object.fromEntries(entries) }
  const keep = new Set(session.openChatIds)
  const bounded = Object.fromEntries(
    Object.entries(merged).filter(([id]) => keep.has(id) || id in titles).slice(-MAX_TRACKED_THREADS),
  )
  return { ...session, titles: bounded }
}

export function toServerSession(session: ChatSession): ServerChatSession {
  return {
    activeChatId: session.activeChatId,
    draftOpen: session.draftOpen,
    openChatIds: session.openChatIds,
    lastReadAt: session.lastReadAt,
  }
}

/**
 * Reconcile with the hosted copy: the server wins when it is strictly newer than
 * this device's last change (a fresh device has none, so it inherits the tabs).
 */
export function mergeServerSession(
  local: ChatSession,
  server: ServerChatSession | null | undefined,
  now: number = Date.now(),
): ChatSession {
  if (!server || !isStringArray(server.openChatIds)) return local
  // A server doc without a stamp was never written (the empty default) — nothing to inherit.
  if (!server.updatedAt) return local
  const serverAt = Date.parse(server.updatedAt)
  const serverKnown = Number.isFinite(serverAt)
  if (serverKnown && serverAt <= local.lastActiveAt) return local
  const openChatIds = server.openChatIds.slice(0, MAX_OPEN_CHATS)
  const active = server.activeChatId && openChatIds.includes(server.activeChatId) ? server.activeChatId : null
  return {
    ...local,
    activeChatId: active,
    draftOpen: Boolean(server.draftOpen) || active === null,
    openChatIds,
    unreadChatIds: local.unreadChatIds.filter((id) => openChatIds.includes(id)),
    lastReadAt: capRecord(
      { ...local.lastReadAt, ...(isRecordOf(server.lastReadAt, 'number') ? server.lastReadAt : {}) },
      (t) => t,
    ),
    lastActiveAt: serverKnown ? serverAt : now,
  }
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
