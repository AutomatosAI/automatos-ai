/**
 * PRD-261 A-12e — SDK area `pages/assignments` (frontend/lib/theme-sdk/pages/assignments.ts): the data the
 * /assignments Missions tab and /missions/[id] read, for a theme's own missions pages. Each hook wraps the core
 * hook the page reads, so a theme's page asks the same endpoints with the same polling and cache as the core page.
 *
 * A query hands back `MissionQuery` (null data until it answers), an action `MissionAction`; an error is its
 * message. A mission id is one path segment of letters, digits, '-' and '_': a query given a malformed one asks for
 * nothing and says so in `error`, and an action given one throws a TypeError before anything is sent. The hooks
 * need the host's providers above them, as every theme page the host draws has.
 */

export type MissionRunState =
  | 'pending'
  | 'planning'
  | 'awaiting_approval'
  | 'running'
  | 'paused'
  | 'verifying'
  | 'awaiting_human'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type MissionTaskState =
  | 'pending'
  | 'queued'
  | 'assigned'
  | 'running'
  | 'completed'
  | 'verifying'
  | 'verified'
  | 'failed'
  | 'skipped'
  | 'stalled'
  | 'retrying'

export type MissionStateType = 'initial' | 'active' | 'blocked' | 'terminal'

/** One Mission, as the list and the actions return it. */
export interface Mission {
  readonly id: string
  readonly workspace_id: string
  readonly goal: string
  readonly state: MissionRunState
  readonly state_type: MissionStateType
  readonly plan: Readonly<Record<string, unknown>> | null
  readonly config: Readonly<Record<string, unknown>> | null
  /** "Check each step with me". */
  readonly check_each_step: boolean
  readonly output_summary: Readonly<Record<string, unknown>> | null
  readonly token_budget_estimate: number | null
  readonly tokens_used: number
  readonly max_retries: number
  readonly max_concurrent: number
  readonly complexity_tier: string | null
  readonly parallel_groups: readonly string[]
  readonly has_synthesis_tasks: boolean
  readonly created_by: string
  /** Why the run stopped or paused. */
  readonly stop_reason?: string | null
  readonly stop_detail?: string | null
  readonly started_at: string | null
  readonly completed_at: string | null
  readonly created_at: string | null
  readonly updated_at: string | null
}

/** One of a Mission's tasks. */
export interface MissionTask {
  readonly id: string
  readonly title: string
  readonly description: string | null
  readonly task_type: string | null
  readonly sequence_number: number
  readonly agent_role: string | null
  readonly state: MissionTaskState
  readonly state_type: MissionStateType
  readonly assigned_agent_id: number | null
  readonly attempt_number: number
  readonly tokens_used: number
  readonly estimated_tokens: number
  readonly complexity: string | null
  readonly parallel_group: string | null
  readonly failure_reason_code: string | null
  readonly failure_detail: string | null
  readonly output_excerpt: string | null
  readonly output: string | null
  readonly started_at: string | null
  readonly completed_at: string | null
  readonly created_at: string | null
}

/** One of a Mission's recent state changes. */
export interface MissionEvent {
  readonly id: string
  readonly event_type: string
  readonly actor_type: string
  readonly actor_id: string | null
  readonly old_state: string | null
  readonly new_state: string | null
  readonly task_id: string | null
  readonly stop_detail?: string | null
  readonly created_at: string | null
}

/** The server's counts of a Mission's tasks. */
export interface MissionTaskProgress {
  readonly total: number
  readonly done: number
  readonly active: number
  readonly verified: number
  readonly failed: number
  readonly skipped: number
  readonly all_terminal: boolean
}

/** One Mission with its tasks and recent events. */
export interface MissionDetail extends Mission {
  readonly tasks: readonly MissionTask[]
  readonly recent_events: readonly MissionEvent[]
  readonly task_progress?: MissionTaskProgress
}

/** A page of the workspace's Missions. */
export interface MissionList {
  readonly missions: readonly Mission[]
  readonly total: number
  readonly limit: number
  readonly offset: number
}

/** The list's filters: a state, and a page (whole numbers). */
export interface MissionFilters {
  readonly state?: MissionRunState
  readonly limit?: number
  readonly offset?: number
}

export interface MissionApproveBody {
  readonly max_concurrent_override?: number
  readonly token_budget_override?: number
  /** The owner's note on the plan; every step's prompt carries it. */
  readonly note?: string
}

export interface MissionRejectBody {
  readonly reason: string
}

/** A new Mission (a re-run): the goal, and the finished one's config. */
export interface MissionCreateBody {
  readonly goal: string
  readonly config?: Readonly<Record<string, unknown>>
  readonly template_id?: string
}

export interface MissionSettingsBody {
  readonly check_each_step: boolean
}

export interface MissionSettings {
  readonly id: string
  readonly check_each_step: boolean
}

export interface SaveAsRoutineBody {
  readonly name: string
  readonly description?: string
  readonly tags?: string[]
}

export interface SavedRoutine {
  readonly template_id: string
  readonly name: string
  readonly task_count: number
}

export type MissionFieldStatus = 'not_created' | 'missing' | 'empty' | 'active' | 'unavailable'
export type MissionFieldScope = 'mission' | 'workspace'

/** One pattern in a Mission's field (its shared memory). */
export interface MissionFieldPattern {
  readonly id: string
  readonly key: string
  readonly value: string
  readonly agent_id: number
  readonly strength: number
  readonly decayed_strength: number
  readonly access_count: number
  readonly created_at: string
  readonly last_accessed: string
  readonly is_archived: boolean
  readonly mission_id?: string | null
  readonly expired_at?: string | null
}

export interface MissionFieldStability {
  readonly stability: number
  readonly pattern_count: number
  readonly avg_strength: number
  readonly organization?: number
  readonly active_patterns?: number
  readonly decayed_patterns?: number
}

export interface MissionFieldMetrics {
  readonly total_injections: number
  readonly total_queries: number
  readonly avg_query_latency_ms: number
  readonly avg_results_per_query: number
  readonly injections_by_agent: Readonly<Record<number, number>>
  readonly queries_by_agent: Readonly<Record<number, number>>
}

export interface MissionField {
  readonly status?: MissionFieldStatus
  readonly field_id: string | null
  readonly backend: string | null
  readonly patterns: readonly MissionFieldPattern[]
  readonly stability: MissionFieldStability
  readonly metrics: MissionFieldMetrics | null
}

/** A pattern that fired for a field query, and its score. */
export interface MissionFieldTraceHit {
  readonly id: string
  readonly key: string
  readonly value: string
  readonly score: number
  readonly cosine_similarity: number
  readonly decayed_strength: number
  readonly agent_id: number
  readonly mission_id?: string | null
}

export interface MissionFieldTrace {
  readonly field_id: string | null
  readonly query: string
  readonly results: readonly MissionFieldTraceHit[]
}

/** A recommended Playbook or Mission, from the workspace or the marketplace. */
export interface RecommendedAssignment {
  readonly id: number | string
  readonly type: 'playbook' | 'mission'
  readonly name: string
  readonly description: string
  readonly icon: string | null
  readonly category: string
  readonly use_count?: number
  readonly install_count?: number
  readonly is_featured?: boolean
  readonly created_at?: string | null
  readonly source: 'workspace' | 'marketplace'
}

export interface RecommendedAssignments {
  readonly items: readonly RecommendedAssignment[]
  readonly workspace_count: number
  readonly marketplace_count: number
}

export interface MissionApproveInput { readonly id: string; readonly body?: MissionApproveBody }
export interface MissionRejectInput { readonly id: string; readonly body: MissionRejectBody }
export interface MissionReplanInput { readonly id: string; readonly notes?: string }
export interface MissionSettingsInput { readonly id: string; readonly body: MissionSettingsBody }
export interface SaveAsRoutineInput { readonly id: string; readonly body: SaveAsRoutineBody }
export interface MissionFieldQueryInput { readonly query: string; readonly topK?: number }
export interface MissionFieldOptions { readonly enabled?: boolean; readonly scope?: MissionFieldScope }

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface MissionQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface MissionAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

/** The workspace's Missions, polled as the core list is. */
export declare function useMissions(filters?: MissionFilters): MissionQuery<MissionList>

/** One Mission with its tasks and recent events; a null id asks for nothing. */
export declare function useMission(missionId: string | null): MissionQuery<MissionDetail>

/** A Mission's field, or the workspace's with `scope: 'workspace'`. */
export declare function useMissionField(missionId: string | null, options?: MissionFieldOptions): MissionQuery<MissionField>

/** Which of a Mission's field patterns fire for a query; a null or malformed id refuses every run. */
export declare function useFieldQuery(missionId: string | null): MissionAction<MissionFieldQueryInput, MissionFieldTrace>

/** The recommended Playbooks and Missions. */
export declare function useRecommendedAssignments(limit?: number): MissionQuery<RecommendedAssignments>

/** Approve a Mission's plan. */
export declare function useApproveMission(): MissionAction<MissionApproveInput, Mission>

/** Reject a Mission's plan, with the reason. */
export declare function useRejectMission(): MissionAction<MissionRejectInput, Mission>

/** Pause a running Mission: `run(missionId)`. */
export declare function usePauseMission(): MissionAction<string, Mission>

/** Resume a paused Mission: `run(missionId)`. */
export declare function useResumeMission(): MissionAction<string, Mission>

/** Cancel a Mission: `run(missionId)`. */
export declare function useCancelMission(): MissionAction<string, Mission>

/** Plan a failed Mission again, with optional notes. */
export declare function useReplanMission(): MissionAction<MissionReplanInput, Mission>

/** Start a new Mission from a finished one's goal and config. */
export declare function useRerunMission(): MissionAction<MissionCreateBody, Mission>

/** Save a Mission as a reusable routine. */
export declare function useSaveAsRoutine(): MissionAction<SaveAsRoutineInput, SavedRoutine>

/** Change a Mission's settings ("Check each step with me"). */
export declare function useUpdateMissionSettings(): MissionAction<MissionSettingsInput, MissionSettings>
