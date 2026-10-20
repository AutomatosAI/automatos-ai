/**
 * PRD-261 A-12h — SDK area `pages/command-center`: the data the /command-center page reads, for a
 * theme's own (or cloned) Command Center page (PRD §4.3 item 7). The names are A-10c's
 * (surface/pages-command-center.json, docs/theme-host/sdk-surface-map.md §3).
 *
 * Each hook wraps the core hook command-center-shell.tsx reads today (hooks/use-activity-api.ts,
 * use-approval-grants.ts, use-board-event-stream.ts, use-needs-you.ts, use-watches-api.ts), so a
 * theme's page asks the same endpoints, through the same backend client, with the same cache keys,
 * polling and invalidation as the core page; no core hook is added or changed. A read comes back
 * as a small, frozen, theme-neutral view, never react-query's own result:
 * `{ data, isLoading, isFetching, error, refetch }`, its error as a message and its data a deep,
 * frozen copy, never the objects react-query caches for the core page (internal/query-view.ts,
 * the views every page area uses). The hooks read
 * react-query, so they need the host's QueryClientProvider above them, as every page and theme the
 * host draws has (components/providers.tsx).
 *
 * A theme is outside code, so its inputs are checked before they reach a URL: the stats and the
 * schedule put their period and range into the path as they are (use-activity-api.ts), so only a
 * value the activity API knows passes (api/activity.py), and anything else asks with the core
 * hook's default; the feed keeps only known types and statuses and whole-number paging inside the
 * API's bounds. useBoardEventStream opens the board's event stream for the theme's page (the core
 * shell is not mounted there); its pushes reach the theme as `events.useHostEvent('board-changed')`
 * and refresh useNeedsYou, as on the core page.
 */
import {
  useActivityFeed as useCoreActivityFeed,
  useActivitySchedule as useCoreActivitySchedule,
  useActivityStats as useCoreActivityStats,
  type ActivityFeedFilters as CoreActivityFeedFilters,
  type ActivityFeedItem as CoreActivityFeedItem,
  type ActivityFeedResponse,
  type ActivityStats as CoreActivityStats,
  type ScheduleItem as CoreScheduleItem,
  type ScheduleResponse,
} from '@/hooks/use-activity-api'
import { useQuestions as useCoreQuestions } from '@/hooks/use-approval-grants'
import { useBoardEventStream as useCoreBoardEventStream } from '@/hooks/use-board-event-stream'
import {
  useNeedsYou as useCoreNeedsYou,
  type NeedsYou as CoreNeedsYou,
  type NeedsYouAskRow as CoreNeedsYouAskRow,
  type NeedsYouCounts as CoreNeedsYouCounts,
  type NeedsYouTicketRow as CoreNeedsYouTicketRow,
} from '@/hooks/use-needs-you'
import { useWatches as useCoreWatches } from '@/hooks/use-watches-api'
import type { ApprovalGrant, ApprovalGrantsResponse, WatchRow, WatchesResponse } from '@/lib/api-client'

import { useQueryView } from '../internal/query-view'

export type ActivityFeed = ActivityFeedResponse
export type ActivityFeedItem = CoreActivityFeedItem
export type ActivityStats = CoreActivityStats
export type ActivitySchedule = ScheduleResponse
export type ScheduleItem = CoreScheduleItem
export type NeedsYou = CoreNeedsYou
export type NeedsYouCounts = CoreNeedsYouCounts
export type NeedsYouTicketRow = CoreNeedsYouTicketRow
export type NeedsYouAskRow = CoreNeedsYouAskRow
export type QuestionGrants = ApprovalGrantsResponse
export type QuestionGrant = ApprovalGrant
export type Watches = WatchesResponse
export type Watch = WatchRow

/** A window of the stats and the feed (api/activity.py: 1d, 7d, 30d, 90d). */
export type ActivityPeriod = '1d' | '7d' | '30d' | '90d'
/** How far ahead the schedule reads (api/activity.py: 7d, 14d, 30d). */
export type ScheduleRange = '7d' | '14d' | '30d'
/** The kinds of run the feed merges (ActivityFeedItem.type). */
export type ActivityItemType = 'chat' | 'routine' | 'recipe' | 'mission' | 'task'
/** A feed status: one of the API's groups, or a run's own status (activity_service.py _map_status_filter). */
export type ActivityStatusFilter =
  | 'working' | 'done' | 'attention' | 'upcoming'
  | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'

export interface ActivityFeedFilters {
  readonly types?: readonly ActivityItemType[]
  readonly status?: ActivityStatusFilter | null
  readonly period?: ActivityPeriod
  /** 1 to 100 (the API's cap); the API's own default when unset. */
  readonly limit?: number
  readonly offset?: number
}

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface CommandCenterQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

const ACTIVITY_PERIODS: readonly ActivityPeriod[] = ['1d', '7d', '30d', '90d']
const SCHEDULE_RANGES: readonly ScheduleRange[] = ['7d', '14d', '30d']
const ITEM_TYPES: readonly ActivityItemType[] = ['chat', 'routine', 'recipe', 'mission', 'task']
const STATUS_FILTERS: readonly ActivityStatusFilter[] = [
  'working', 'done', 'attention', 'upcoming', 'pending', 'running', 'completed', 'failed', 'cancelled', 'paused',
]
/** The core hooks' own defaults (use-activity-api.ts, command-center-shell.tsx). */
const DEFAULT_STATS_PERIOD: ActivityPeriod = '1d'
const DEFAULT_SCHEDULE_RANGE: ScheduleRange = '7d'
/** The feed's page size bounds (api/activity.py: limit ge=1 le=100, offset ge=0). */
const MIN_FEED_LIMIT = 1
const MAX_FEED_LIMIT = 100

function known<T extends string>(allowed: readonly T[], value: unknown): T | undefined {
  return allowed.find((entry) => entry === value)
}

function wholeNumberIn(value: unknown, min: number, max: number): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max ? value : undefined
}

/** The known item types of a theme's list, or undefined (every type) when none is left. */
function itemTypes(types: unknown): string[] | undefined {
  if (!Array.isArray(types)) return undefined
  const kept = ITEM_TYPES.filter((type) => types.includes(type))
  return kept.length > 0 ? kept : undefined
}

/** The filters the core hook may put in the feed's query string. */
function feedFilters(filters: ActivityFeedFilters | undefined): CoreActivityFeedFilters {
  return {
    types: itemTypes(filters?.types),
    status: known(STATUS_FILTERS, filters?.status),
    period: known(ACTIVITY_PERIODS, filters?.period),
    limit: wholeNumberIn(filters?.limit, MIN_FEED_LIMIT, MAX_FEED_LIMIT),
    offset: wholeNumberIn(filters?.offset, 0, Number.MAX_SAFE_INTEGER),
  }
}

/** The workspace's runs (chats, routines, Playbooks, Missions, tasks), newest first, polled as the core feed is. */
export function useActivityFeed(filters?: ActivityFeedFilters): CommandCenterQuery<ActivityFeed> {
  return useQueryView<ActivityFeed>(useCoreActivityFeed(feedFilters(filters)))
}

/** The stats strip's numbers for a period (1d when unset or unknown). */
export function useActivityStats(period?: ActivityPeriod): CommandCenterQuery<ActivityStats> {
  return useQueryView<ActivityStats>(useCoreActivityStats(known(ACTIVITY_PERIODS, period) ?? DEFAULT_STATS_PERIOD))
}

/** What is scheduled to run next (7d when unset or unknown); a refetch keeps the last answer on screen. */
export function useActivitySchedule(range?: ScheduleRange): CommandCenterQuery<ActivitySchedule> {
  return useQueryView<ActivitySchedule>(useCoreActivitySchedule(known(SCHEDULE_RANGES, range) ?? DEFAULT_SCHEDULE_RANGE))
}

/** The open questions waiting for an answer, newest first. */
export function useQuestions(): CommandCenterQuery<QuestionGrants> {
  return useQueryView<QuestionGrants>(useCoreQuestions())
}

/** The one Needs-you number and the rows behind it; a board change refreshes it at once. */
export function useNeedsYou(): CommandCenterQuery<NeedsYou> {
  return useQueryView<NeedsYou>(useCoreNeedsYou())
}

/** The live watches (every watch, closed ones too, with `includeClosed`). */
export function useWatches(includeClosed?: boolean): CommandCenterQuery<Watches> {
  return useQueryView<Watches>(useCoreWatches(includeClosed === true))
}

/** Keeps the board's event stream open while `enabled` (on by default) and the page is mounted. */
export function useBoardEventStream(enabled?: boolean): void {
  useCoreBoardEventStream(enabled !== false)
}
