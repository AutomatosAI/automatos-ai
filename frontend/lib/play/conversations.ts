import type { HostChatSummary } from '@automatos/theme-sdk'

/**
 * PRD-260 §4 — Play's Conversations list as pure data, ported from Automatos
 * Play's lib/conversations/list.ts onto the core's Chat (getChatHistory): the
 * groups by calendar day (Today, This week, Earlier), the relative time, the
 * title, the last line and the title search. Dates are the user's local
 * calendar days, and "now" is always passed in, so tests pin it.
 *
 * The core's preview is the newest message of either side (orchestrator
 * api/chat.py _last_message_previews), so the line is shown without Play's
 * "Auto:" speaker: it may be the user's own words.
 *
 * PRD-261 A-14c: Play reads the core only through '@automatos/theme-sdk'. The
 * page's list is the SDK's chat.useConversationList, whose rows are
 * HostChatSummary; summaryOfHostChat turns one into the row read here.
 */

/**
 * The fields of a history row the list reads: the core's Chat fields, as Play's
 * own type (PRD-261 surface map: types are copied into Play). createdAt is only
 * the fallback when updatedAt is unreadable, and the SDK's summary carries none.
 */
export interface ConversationSummary {
  id: string
  title: string
  createdAt?: string | null
  updatedAt: string
  lastMessagePreview?: string | null
}

export type GroupLabel = 'Today' | 'This week' | 'Earlier'

export interface ConversationGroup<T extends ConversationSummary = ConversationSummary> {
  readonly label: GroupLabel
  readonly chats: readonly T[]
}

/** A conversation with no title yet. */
export const UNTITLED_CONVERSATION = 'New chat'

const GROUP_ORDER: readonly GroupLabel[] = ['Today', 'This week', 'Earlier']
/** "This week" is the six calendar days before today. */
const WEEK_DAYS = 7
const MINUTE_MS = 60_000
const HOUR_MINUTES = 60
const DAY_MS = 24 * HOUR_MINUTES * MINUTE_MS
const LOCALE = 'en-GB'

const weekday = new Intl.DateTimeFormat(LOCALE, { weekday: 'short' })
const dayMonth = new Intl.DateTimeFormat(LOCALE, { day: 'numeric', month: 'short' })
const dayMonthYear = new Intl.DateTimeFormat(LOCALE, { day: 'numeric', month: 'short', year: 'numeric' })

function parseDate(iso: string | null | undefined): Date | null {
  if (!iso) return null
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? null : date
}

/** When the conversation last moved (the history is ordered by it), else when it began. */
export function lastActive(chat: ConversationSummary): Date | null {
  return parseDate(chat.updatedAt) ?? parseDate(chat.createdAt)
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
}

/** Whole calendar days from date to now: 0 today, 1 yesterday; negative in the future. Rounded, so a DST day still counts as one. */
function daysAgo(date: Date, now: Date): number {
  return Math.round((startOfDay(now) - startOfDay(date)) / DAY_MS)
}

/** Today from local midnight; This week for the six days before; Earlier after that or with no date. */
export function groupOf(date: Date | null, now: Date): GroupLabel {
  if (!date) return 'Earlier'
  const days = daysAgo(date, now)
  if (days <= 0) return 'Today'
  return days < WEEK_DAYS ? 'This week' : 'Earlier'
}

/** The groups in the mock's order, empty ones left out; the history's own order is kept inside each. */
export function groupConversations<T extends ConversationSummary>(
  chats: readonly T[],
  now: Date,
): readonly ConversationGroup<T>[] {
  return GROUP_ORDER.map((label) => ({
    label,
    chats: chats.filter((chat) => groupOf(lastActive(chat), now) === label),
  })).filter((group) => group.chats.length > 0)
}

function sameDayTime(date: Date, now: Date): string {
  const minutes = Math.max(0, Math.floor((now.getTime() - date.getTime()) / MINUTE_MS))
  if (minutes < 1) return 'Just now'
  if (minutes < HOUR_MINUTES) return `${minutes} min`
  return `${Math.floor(minutes / HOUR_MINUTES)} h`
}

/** "2 min", "3 h", "Yesterday", "Mon", "9 Oct" (with the year when it isn't this one); "" when unknown. */
export function relativeTime(date: Date | null, now: Date): string {
  if (!date) return ''
  const days = daysAgo(date, now)
  if (days <= 0) return sameDayTime(date, now)
  if (days === 1) return 'Yesterday'
  if (days < WEEK_DAYS) return weekday.format(date)
  return date.getFullYear() === now.getFullYear() ? dayMonth.format(date) : dayMonthYear.format(date)
}

/** The title on one line, or UNTITLED_CONVERSATION. */
export function conversationTitle(chat: ConversationSummary): string {
  return (chat.title ?? '').replace(/\s+/g, ' ').trim() || UNTITLED_CONVERSATION
}

/** The newest message on one line, or null when the conversation has none yet. */
export function previewLine(chat: ConversationSummary): string | null {
  const preview = chat.lastMessagePreview?.replace(/\s+/g, ' ').trim()
  return preview || null
}

/**
 * One of the SDK's conversations as the list reads it. The SDK hands no start
 * time; the history's updatedAt is always set (chats.updated_at is NOT NULL),
 * so lastActive reads the same date it read from the core's Chat.
 */
export function summaryOfHostChat(chat: HostChatSummary): ConversationSummary {
  return { id: chat.id, title: chat.title, updatedAt: chat.updatedAt, lastMessagePreview: chat.preview }
}

/** A case-insensitive match on the title; an empty query keeps everything. */
export function filterByTitle<T extends ConversationSummary>(chats: readonly T[], query: string): readonly T[] {
  const needle = query.trim().toLocaleLowerCase()
  if (!needle) return chats
  return chats.filter((chat) => conversationTitle(chat).toLocaleLowerCase().includes(needle))
}
