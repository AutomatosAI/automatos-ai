/**
 * PRD-261 A-12e — SDK area `pages/assignments`: the data the /assignments Missions tab and
 * /missions/[id] read, for a theme's own (or cloned) missions pages (PRD §4.3 item 7). The names
 * are A-10c's (surface/pages-assignments.json, docs/theme-host/sdk-surface-map.md §3).
 *
 * Each hook wraps the core hook the page reads today (hooks/use-missions-api.ts,
 * hooks/use-assignments-api.ts), so a theme's page asks the same endpoints, through the same
 * backend client, with the same cache keys, polling and invalidation as the core page; no core
 * hook is added or changed. What comes back is a small, frozen, theme-neutral view, never
 * react-query's own result: a query is `{ data, isLoading, isFetching, error, refetch }`, an action
 * is `{ run, runAsync, isRunning, error, result, reset }`, and an error is its message. The hooks
 * read react-query, so they need the host's QueryClientProvider above them, as every page and
 * theme the host draws has (components/providers.tsx).
 *
 * A theme is outside code, so its inputs are checked before they reach a URL: a mission id is one
 * path segment of letters, digits, '-' and '_' (a query with a malformed id asks for nothing and
 * says why; an action given one throws a TypeError, as the navigation area does for a bad href),
 * list filters keep only a known state and whole-number limits, and the field scope is one of two.
 */
import { useCallback, useMemo } from 'react'

import {
  useRecommendedAssignments as useCoreRecommendedAssignments,
  type RecommendedItem,
  type RecommendedResponse,
} from '@/hooks/use-assignments-api'
import {
  useApproveMission as useCoreApproveMission,
  useCancelMission as useCoreCancelMission,
  useFieldQuery as useCoreFieldQuery,
  useMission as useCoreMission,
  useMissionField as useCoreMissionField,
  useMissions as useCoreMissions,
  usePauseMission as useCorePauseMission,
  useRejectMission as useCoreRejectMission,
  useReplanMission as useCoreReplanMission,
  useRerunMission as useCoreRerunMission,
  useResumeMission as useCoreResumeMission,
  useSaveAsRoutine as useCoreSaveAsRoutine,
  useUpdateMissionSettings as useCoreUpdateMissionSettings,
  type FieldPattern,
  type FieldMetrics,
  type FieldScope,
  type FieldStability,
  type FieldTraceHit,
  type FieldTraceResponse,
  type MissionFieldResponse,
  type MissionFieldStatus as CoreMissionFieldStatus,
  type MissionFilters as CoreMissionFilters,
} from '@/hooks/use-missions-api'
import {
  RUN_STATE_CONFIG,
  type EventResponse,
  type MissionApproveRequest,
  type MissionCreateRequest,
  type MissionDetailResponse,
  type MissionListResponse,
  type MissionRejectRequest,
  type MissionResponse,
  type MissionSettingsRequest,
  type MissionSettingsResponse,
  type RunState,
  type SaveAsRoutineRequest,
  type SaveAsRoutineResponse,
  type StateType,
  type TaskProgress,
  type TaskResponse,
  type TaskState,
} from '@/types/missions'

export type MissionRunState = RunState
export type MissionTaskState = TaskState
export type MissionStateType = StateType
export type Mission = MissionResponse
export type MissionDetail = MissionDetailResponse
export type MissionTask = TaskResponse
export type MissionEvent = EventResponse
export type MissionTaskProgress = TaskProgress
export type MissionList = MissionListResponse
export type MissionFilters = CoreMissionFilters
export type MissionApproveBody = MissionApproveRequest
export type MissionRejectBody = MissionRejectRequest
export type MissionCreateBody = MissionCreateRequest
export type MissionSettingsBody = MissionSettingsRequest
export type MissionSettings = MissionSettingsResponse
export type SaveAsRoutineBody = SaveAsRoutineRequest
export type SavedRoutine = SaveAsRoutineResponse
export type MissionField = MissionFieldResponse
export type MissionFieldStatus = CoreMissionFieldStatus
export type MissionFieldScope = FieldScope
export type MissionFieldPattern = FieldPattern
export type MissionFieldStability = FieldStability
export type MissionFieldMetrics = FieldMetrics
export type MissionFieldTrace = FieldTraceResponse
export type MissionFieldTraceHit = FieldTraceHit
export type RecommendedAssignment = RecommendedItem
export type RecommendedAssignments = RecommendedResponse

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

/** The parts of react-query's query result the view reads. */
interface CoreQuery<T> {
  readonly data: T | undefined
  readonly error: unknown
  readonly isInitialLoading: boolean
  readonly isFetching: boolean
  readonly refetch: () => unknown
}

/** The parts of react-query's mutation result the view reads. */
interface CoreMutation<V, R> {
  readonly mutate: (variables: V) => void
  readonly mutateAsync: (variables: V) => Promise<R>
  readonly isLoading: boolean
  readonly error: unknown
  readonly data: R | undefined
  readonly reset: () => void
}

/** Why an action's variables are refused, or null when they may go. */
type Refusal<V> = (variables: V) => string | null

const MISSION_ID = /^[A-Za-z0-9_-]{1,128}$/
const NOT_A_MISSION_ID = 'Not a mission id.'
const FIELD_SCOPES: readonly MissionFieldScope[] = ['mission', 'workspace']
const DEFAULT_FIELD_SCOPE: MissionFieldScope = 'mission'

function isMissionId(value: unknown): value is string {
  return typeof value === 'string' && MISSION_ID.test(value)
}

function messageOf(error: unknown): string | null {
  if (error === null || error === undefined) return null
  return error instanceof Error ? error.message : String(error)
}

function wholeNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined
}

function knownRunState(value: unknown): MissionRunState | undefined {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(RUN_STATE_CONFIG, value)
    ? (value as MissionRunState)
    : undefined
}

/** The field scope the core hook may put in its query string: one of two, the Mission's by default. */
function fieldScope(value: unknown): MissionFieldScope {
  return FIELD_SCOPES.find((scope) => scope === value) ?? DEFAULT_FIELD_SCOPE
}

/** The filters the core hook may put in the list's query string: a known state, whole numbers. */
function listFilters(filters: MissionFilters | undefined): MissionFilters {
  return { state: knownRunState(filters?.state), limit: wholeNumber(filters?.limit), offset: wholeNumber(filters?.offset) }
}

/** The mission id a query may ask for: the id, null for none or a malformed one. */
function queryId(missionId: unknown): { readonly id: string | null; readonly refusal: string | null } {
  if (missionId === null || missionId === undefined) return { id: null, refusal: null }
  return isMissionId(missionId) ? { id: missionId, refusal: null } : { id: null, refusal: NOT_A_MISSION_ID }
}

function refuseBadId(id: unknown): string | null {
  return isMissionId(id) ? null : NOT_A_MISSION_ID
}

const refuseBadInputId: Refusal<{ readonly id: string }> = (variables) => refuseBadId(variables?.id)
const refuseBadIdArgument: Refusal<string> = (id) => refuseBadId(id)
const refuseNothing: Refusal<unknown> = () => null

/** The theme's view of a core query; `refusal` stands in for the error when the SDK asked for nothing. */
function useQueryView<T>(query: CoreQuery<T>, refusal: string | null = null): MissionQuery<T> {
  const { data, error, isInitialLoading, isFetching, refetch } = query
  return useMemo(
    () =>
      Object.freeze({
        data: data ?? null,
        isLoading: isInitialLoading,
        isFetching,
        error: refusal ?? messageOf(error),
        refetch: () => {
          void refetch()
        },
      }),
    [data, error, isInitialLoading, isFetching, refetch, refusal],
  )
}

/** The theme's view of a core mutation; variables `refuse` names are thrown as a TypeError, never sent. */
function useActionView<V, R>(mutation: CoreMutation<V, R>, refuse: Refusal<V>): MissionAction<V, R> {
  const { mutate, mutateAsync, isLoading, error, data, reset } = mutation
  return useMemo(() => {
    const checked = (variables: V): V => {
      const reason = refuse(variables)
      if (reason !== null) throw new TypeError(`Automatos theme SDK: ${reason}`)
      return variables
    }
    return Object.freeze({
      run: (variables: V) => mutate(checked(variables)),
      runAsync: (variables: V) => new Promise<R>((resolve, reject) => {
        mutateAsync(checked(variables)).then(resolve, reject)
      }),
      isRunning: isLoading,
      error: messageOf(error),
      result: data ?? null,
      reset,
    })
  }, [mutate, mutateAsync, isLoading, error, data, reset, refuse])
}

/** The workspace's Missions, newest first, polled as the core list is. */
export function useMissions(filters?: MissionFilters): MissionQuery<MissionList> {
  return useQueryView<MissionList>(useCoreMissions(listFilters(filters)))
}

/** One Mission with its tasks and recent events; null id asks for nothing. */
export function useMission(missionId: string | null): MissionQuery<MissionDetail> {
  const { id, refusal } = queryId(missionId)
  return useQueryView<MissionDetail>(useCoreMission(id), refusal)
}

/** A Mission's field (its shared memory), or the workspace's with `scope: 'workspace'`. */
export function useMissionField(missionId: string | null, options?: MissionFieldOptions): MissionQuery<MissionField> {
  const { id, refusal } = queryId(missionId)
  const field = useCoreMissionField(id, options?.enabled !== false, fieldScope(options?.scope))
  return useQueryView<MissionField>(field, refusal)
}

/** Which of a Mission's field patterns fire for a query, and why; a null or malformed id refuses every run. */
export function useFieldQuery(missionId: string | null): MissionAction<MissionFieldQueryInput, MissionFieldTrace> {
  const refuse = useCallback<Refusal<MissionFieldQueryInput>>(() => refuseBadId(missionId), [missionId])
  const trace = useCoreFieldQuery(isMissionId(missionId) ? missionId : null)
  return useActionView<MissionFieldQueryInput, MissionFieldTrace>(trace, refuse)
}

/** The recommended Playbooks and Missions, from the workspace and the marketplace. */
export function useRecommendedAssignments(limit?: number): MissionQuery<RecommendedAssignments> {
  const count = wholeNumber(limit)
  const recommended = useCoreRecommendedAssignments(count !== undefined && count > 0 ? count : undefined)
  return useQueryView<RecommendedAssignments>(recommended)
}

/** Approve a Mission's plan (an optional note and overrides). */
export function useApproveMission(): MissionAction<MissionApproveInput, Mission> {
  return useActionView<MissionApproveInput, Mission>(useCoreApproveMission(), refuseBadInputId)
}

/** Reject a Mission's plan, with the reason. */
export function useRejectMission(): MissionAction<MissionRejectInput, Mission> {
  return useActionView<MissionRejectInput, Mission>(useCoreRejectMission(), refuseBadInputId)
}

/** Pause a running Mission; `run(missionId)`. */
export function usePauseMission(): MissionAction<string, Mission> {
  return useActionView<string, Mission>(useCorePauseMission(), refuseBadIdArgument)
}

/** Resume a paused Mission; `run(missionId)`. */
export function useResumeMission(): MissionAction<string, Mission> {
  return useActionView<string, Mission>(useCoreResumeMission(), refuseBadIdArgument)
}

/** Cancel a Mission; `run(missionId)`. */
export function useCancelMission(): MissionAction<string, Mission> {
  return useActionView<string, Mission>(useCoreCancelMission(), refuseBadIdArgument)
}

/** Plan a failed Mission again, with optional notes. */
export function useReplanMission(): MissionAction<MissionReplanInput, Mission> {
  return useActionView<MissionReplanInput, Mission>(useCoreReplanMission(), refuseBadInputId)
}

/** Start a new Mission from a finished one's goal and config. */
export function useRerunMission(): MissionAction<MissionCreateBody, Mission> {
  return useActionView<MissionCreateBody, Mission>(useCoreRerunMission(), refuseNothing)
}

/** Save a Mission as a reusable routine. */
export function useSaveAsRoutine(): MissionAction<SaveAsRoutineInput, SavedRoutine> {
  return useActionView<SaveAsRoutineInput, SavedRoutine>(useCoreSaveAsRoutine(), refuseBadInputId)
}

/** Change a Mission's settings ("Check each step with me"). */
export function useUpdateMissionSettings(): MissionAction<MissionSettingsInput, MissionSettings> {
  return useActionView<MissionSettingsInput, MissionSettings>(useCoreUpdateMissionSettings(), refuseBadInputId)
}
