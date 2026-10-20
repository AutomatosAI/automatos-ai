/**
 * PRD-261 A-12h — SDK area `pages/command-center` (frontend/lib/theme-sdk/pages/command-center.ts):
 * the data the /command-center page reads, for a theme's own Command Center page.
 *
 * Each hook reads what the core page reads, through the host's backend client, polled and refreshed
 * as there. A read is a frozen view, never react-query's result: `data` is null until it answers,
 * `isLoading` is the first load only, and `error` is a message. A period, range, type or status the
 * activity API does not know is dropped (the stats and the schedule then use 1d and 7d).
 * useBoardEventStream opens the board's event stream while the page is mounted; its pushes reach a
 * theme as `useHostEvent('board-changed')` (events area) and refresh useNeedsYou.
 *
 * The response shapes are the backend's, read only; the helper shapes below them are not exported.
 */

/** A window of the stats and the feed. */
export type ActivityPeriod = '1d' | '7d' | '30d' | '90d'
/** How far ahead the schedule reads. */
export type ScheduleRange = '7d' | '14d' | '30d'
/** The kinds of run the feed merges. */
export type ActivityItemType = 'chat' | 'routine' | 'recipe' | 'mission' | 'task'
/** A feed status: one of the API's groups, or a run's own status. */
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

interface ActivityAgent {
  readonly id: number | null
  readonly name: string
  readonly avatar_url: string | null
}

interface ActivityStepProgress {
  readonly current: number
  readonly total: number
  readonly steps: readonly { readonly name: string; readonly status: 'pending' | 'running' | 'completed' | 'failed' }[]
}

export interface ActivityFeedItem {
  readonly id: string
  readonly type: ActivityItemType
  readonly name: string
  readonly status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  readonly started_at: string | null
  readonly completed_at: string | null
  readonly duration_seconds: number | null
  readonly agent: ActivityAgent | null
  readonly agents: readonly ActivityAgent[]
  readonly summary: string
  readonly source_id: string | null
  readonly source_url: string | null
  readonly trigger: string | null
  readonly channel?: { readonly type: string; readonly name: string } | null
  readonly step_progress?: ActivityStepProgress | null
  readonly error_message: string | null
  readonly tokens_used?: number | null
  readonly total_tokens?: number | null
  readonly total_duration_ms?: number | null
  /** The latest orchestration event as a plain-English progress line. */
  readonly last_progress?: { readonly summary: string; readonly at: string | null; readonly requires_attention: boolean } | null
  readonly orchestration_run_id?: string | null
  /** A ticket's board status ("blocked"). */
  readonly board_status?: string | null
  /** A ticket's number, #0042. */
  readonly number?: string | null
}

export interface ActivityFeed {
  readonly items: readonly ActivityFeedItem[]
  readonly total: number
  readonly limit: number
  readonly offset: number
}

export interface ActivityStats {
  readonly working_now: number
  readonly channels_live: number
  readonly completed_today: number
  /** The Needs-you number for this viewer. */
  readonly needs_attention: number
  readonly agents_active: number
  readonly tasks_in_queue: number
  readonly period: string
}

interface ScheduleRecurrence {
  readonly cron_expression: string | null
  readonly interval_minutes: number | null
  readonly timezone: string
  readonly active_hours: Readonly<Record<string, unknown>> | null
}

export interface ScheduleItem {
  readonly id: string
  readonly name: string
  /** routine, recipe (a scheduled Playbook), task, mission (a deadline), task_due (a board deadline) or social. */
  readonly type: 'routine' | 'recipe' | 'task' | 'mission' | 'task_due' | 'social'
  readonly next_run_at: string | null
  readonly frequency: string
  readonly agent_name: string | null
  readonly agent_id: number | null
  readonly recurrence?: ScheduleRecurrence
  readonly scheduled_task_id?: number
  readonly task_type?: 'one_shot' | 'recurring'
  readonly deliver_as?: 'chat' | 'board_task'
  readonly mission_id?: number
  readonly playbook_id?: number
  readonly board_task_id?: number
  readonly status?: string
  readonly priority?: string
  readonly post_id?: string
  readonly timezone?: string
}

export interface ActivitySchedule {
  readonly scheduled: readonly ScheduleItem[]
  readonly scheduler_active?: boolean
}

export interface NeedsYouTicketRow {
  readonly ticket_id: number
  readonly number?: string | null
  readonly title: string | null
  readonly agent_name: string | null
  readonly mission_id: string | null
  readonly mission_number?: string | null
  readonly mission_title?: string | null
  /** What the row opens: the ticket, or its Mission. */
  readonly opens?: 'ticket' | 'mission'
  readonly at: string | null
  /** On a stuck ticket: why nothing will move it. */
  readonly why?: 'no_host' | 'no_agent' | 'not_picked_up' | 'mission_ended' | 'step_failed' | 'over_budget' | 'out_of_credit'
}

export interface NeedsYouAskRow {
  readonly source: 'grant' | 'mission'
  readonly id: string
  readonly title: string | null
  readonly ticket_id: number | null
  readonly number?: string | null
  readonly agent_name: string | null
  readonly mission_id?: string | null
  readonly mission_number?: string | null
  readonly mission_title?: string | null
  readonly at: string | null
}

export interface NeedsYouCounts {
  readonly review: number
  readonly question: number
  readonly approval: number
  readonly stuck: number
  readonly failed: number
}

export interface NeedsYou {
  readonly total: number
  readonly counts: NeedsYouCounts
  readonly rows: {
    readonly review: readonly NeedsYouTicketRow[]
    readonly question: readonly NeedsYouAskRow[]
    readonly approval: readonly NeedsYouAskRow[]
    readonly stuck: readonly NeedsYouTicketRow[]
    readonly failed: readonly NeedsYouTicketRow[]
  }
}

interface GrantOversight {
  readonly risk_class: string
  readonly tier: string
  readonly rationale: string
  readonly requires_approval: boolean
}

interface GrantOwnerRef {
  readonly agent?: { readonly id: number; readonly name?: string | null } | null
  readonly ticket?: { readonly id: number; readonly title?: string | null; readonly number?: string | null } | null
}

export interface QuestionGrant {
  readonly id: number
  readonly workspace_id: string | null
  readonly subject_type: string
  readonly subject_id: string
  readonly tool_name: string | null
  readonly risk_tier: string | null
  readonly agent_id: number | null
  readonly status: string
  readonly reason: string | null
  readonly estimated_cost_usd: string | null
  readonly requested_at: string | null
  readonly expires_at: string | null
  readonly granted_at: string | null
  readonly granted_by: string | null
  readonly revoked_at: string | null
  readonly revoked_by: string | null
  readonly oversight?: GrantOversight
  readonly details?: Readonly<Record<string, unknown>>
  /** 'question' for every row useQuestions reads. */
  readonly kind?: string
  /** The question, in Markdown. */
  readonly question_md?: string | null
  readonly options?: readonly string[] | null
  readonly answer_text?: string | null
  readonly answered_by?: string | null
  readonly answered_at?: string | null
  readonly asked_by_agent_id?: number | null
  /** The agent and the ticket the question belongs to. */
  readonly owner?: GrantOwnerRef | null
  readonly channel_refs?: Readonly<Record<string, unknown>>
  /** The downstream tasks parked behind the question. */
  readonly cascade?: {
    readonly total: number
    readonly tasks: readonly { readonly id: number; readonly title: string; readonly status: string }[]
  }
}

export interface QuestionGrants {
  readonly grants: readonly QuestionGrant[]
}

export interface Watch {
  readonly id: string
  readonly title: string
  readonly watch_type: string
  readonly target_type: string
  readonly target_id: string
  readonly status: string
  readonly policy: string
  readonly success_criteria: string | null
  readonly quality_threshold: number
  readonly final_score: number | null
  /** "8.3/10" or "unscored". */
  readonly final_score_display: string
  readonly final_verdict: string | null
  readonly actions_taken: number
  readonly action_budget: number
  readonly last_checked_at: string | null
  readonly next_check_at: string | null
  readonly deadline_at: string | null
  readonly created_at: string | null
  readonly closed_at: string | null
  readonly lineage?: readonly Readonly<Record<string, string>>[]
}

export interface Watches {
  readonly watches: readonly Watch[]
  readonly total: number
}

/** The workspace's runs (chats, routines, Playbooks, Missions, tasks), newest first. */
export declare function useActivityFeed(filters?: ActivityFeedFilters): CommandCenterQuery<ActivityFeed>
/** The stats strip's numbers for a period (1d when unset or unknown). */
export declare function useActivityStats(period?: ActivityPeriod): CommandCenterQuery<ActivityStats>
/** What is scheduled to run next (7d when unset or unknown). */
export declare function useActivitySchedule(range?: ScheduleRange): CommandCenterQuery<ActivitySchedule>
/** The open questions waiting for an answer, newest first. */
export declare function useQuestions(): CommandCenterQuery<QuestionGrants>
/** The one Needs-you number and the rows behind it; a board change refreshes it at once. */
export declare function useNeedsYou(): CommandCenterQuery<NeedsYou>
/** The live watches (closed ones too, with `includeClosed`). */
export declare function useWatches(includeClosed?: boolean): CommandCenterQuery<Watches>
/** Keeps the board's event stream open while `enabled` (on by default) and the page is mounted. */
export declare function useBoardEventStream(enabled?: boolean): void

// Only the declarations marked `export` above leave this file (the helper shapes stay private).
export {}
