/**
 * PRD-261 A-12g — SDK area `pages/playbooks`: the data the Playbooks pages read, for a theme's own
 * (or cloned) /playbooks page (PRD §4.3 item 7). The names are A-10c's (surface/pages-playbooks.json,
 * docs/theme-host/sdk-surface-map.md §3).
 *
 * Each hook wraps the core hook of hooks/use-playbook-api.ts that the Playbooks views read today
 * (the Assignments hub's Playbooks tab and its run dialog, where /playbooks forwards under Studio),
 * so a theme's page asks the same endpoints, through the same backend client, with the same cache
 * keys and invalidation as the core page; no core hook is added or changed. Classic's
 * PlaybooksPanel calls `apiClient.listWorkflowRecipes` itself, the same endpoint `useWorkflowPlaybooks`
 * reads. What comes back is a small, frozen, theme-neutral view, never react-query's own result: a
 * query is `{ data, isLoading, isFetching, error, refetch }`, an action is
 * `{ run, runAsync, isRunning, error, result, reset }`, and an error is its message; the data and
 * results are deep, frozen copies, never the objects react-query caches for the core page (the same
 * views as the other page areas, internal/query-view.ts). The hooks read react-query, so they need the host's QueryClientProvider
 * above them, as every page and theme the host draws has (components/providers.tsx).
 *
 * The backend client types these answers as unknown; the shapes below are the backend's own
 * (WorkflowTemplate.to_dict and RecipeExecution.to_dict, orchestrator/core/models/core.py; the
 * routes in orchestrator/api/workflow_recipes.py), as the core views read them.
 *
 * A theme is outside code, so its inputs are checked before they reach a URL or a body: a playbook
 * is addressed by its template id (letters, digits, '.', '-' and '_', starting with a letter or
 * digit) or by its number; a query given a malformed address asks for nothing and says why, and an
 * action given one throws a TypeError, as the navigation area does for a bad href. List filters
 * keep only booleans, a bounded search, whole-number paging within the backend's bounds and a
 * known sort or run status; a playbook body and a run's inputs must be plain objects.
 */
import {
  useCreatePlaybook as useCoreCreatePlaybook,
  useDeletePlaybook as useCoreDeletePlaybook,
  useExecutePlaybook as useCoreExecutePlaybook,
  usePlaybookExecutions as useCorePlaybookExecutions,
  useUpdatePlaybook as useCoreUpdatePlaybook,
  useWorkflowPlaybook as useCoreWorkflowPlaybook,
  useWorkflowPlaybooks as useCoreWorkflowPlaybooks,
} from '@/hooks/use-playbook-api'

import { refuse, useActionView, useQueryView, type Prepare } from '../internal/query-view'

/** The orders the backend lists playbooks in (its `sort_by` pattern). */
export type PlaybookSort = 'popularity' | 'created_at' | 'use_count' | 'average_rating' | 'name'
export type PlaybookRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
/** A playbook's template id, or the number the list shows. */
export type PlaybookId = string | number
/** A playbook as the create and update routes take it (the backend checks every field). */
export type PlaybookDraft = Readonly<Record<string, unknown>>

/** One playbook, as the list and the detail return it. */
export interface Playbook {
  readonly id: number
  readonly template_id: string
  readonly name: string
  readonly description: string | null
  readonly tags: readonly string[]
  readonly steps: readonly Readonly<Record<string, unknown>>[]
  readonly inputs: unknown
  readonly schedule_config: Readonly<Record<string, unknown>> | null
  readonly use_count: number | null
  readonly success_rate: number | null
  readonly quality_score: number | null
  readonly is_public: boolean | null
  readonly is_featured: boolean | null
  readonly version: string | null
  readonly created_by: string | null
  readonly created_at: string | null
  readonly updated_at: string | null
  readonly last_used_at: string | null
}

export interface PlaybookList {
  readonly items: readonly Playbook[]
  readonly total: number
  readonly skip: number
  readonly limit: number
}

export interface PlaybookFilters {
  readonly is_featured?: boolean
  readonly is_public?: boolean
  readonly search?: string
  readonly skip?: number
  readonly limit?: number
  readonly sort_by?: PlaybookSort
}

/** One run of a playbook. */
export interface PlaybookRun {
  readonly id: number
  readonly execution_id: string
  readonly recipe_id: number
  readonly status: string
  readonly current_step: number | null
  readonly error_message: string | null
  readonly started_at: string | null
  readonly completed_at: string | null
  readonly triggered_by: string | null
  readonly attempt_count: number | null
  readonly retry_of: string | null
}

export interface PlaybookRunList {
  readonly items: readonly PlaybookRun[]
  readonly total: number
  readonly skip: number
  readonly limit: number
  readonly recipe_id: string
  readonly recipe_quality_score: number | null
}

export interface PlaybookRunFilters {
  readonly status?: PlaybookRunStatus
  readonly skip?: number
  readonly limit?: number
}

export interface PlaybookUpdateInput { readonly id: PlaybookId; readonly body: PlaybookDraft }
/** `waitForMe` is this run's own "wait for me" (F242); left out, the playbook's own setting decides. */
export interface PlaybookRunInput {
  readonly id: PlaybookId
  readonly inputs?: Readonly<Record<string, unknown>>
  readonly waitForMe?: boolean
}

/** The create and update routes' answer. */
export interface PlaybookSaved { readonly message: string; readonly recipe: Playbook }
export interface PlaybookDeleted { readonly message: string; readonly recipe_id: string }
/** The Run button's answer: the run's id, and the card it works on by number (F294). */
export interface PlaybookRunStarted {
  readonly recipe_execution_id: string
  readonly recipe_id: string
  readonly status: string
  readonly total_steps: number
  readonly message: string
  readonly task_id: number | null
  readonly number: string | null
}

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface PlaybookQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface PlaybookAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

/** The core hook's variables for the theme's, or a TypeError naming what is wrong. */
type ToCore<V, C> = Prepare<V, C>

interface CoreUpdateVariables {
  readonly playbookId: string
  readonly playbookData: PlaybookDraft
}

interface CoreRunVariables {
  readonly playbookId: string
  readonly inputData?: Record<string, unknown>
  readonly waitForMe?: boolean
}

const PLAYBOOK_TEMPLATE_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$/
const NOT_A_PLAYBOOK_ID = 'Not a playbook id.'
const NOT_A_BODY = 'A playbook body is a plain object.'
const NOT_INPUTS = "A run's inputs are a plain object."
const SEARCH_MAX_LENGTH = 200
/** The backend's page bounds: `limit` 1..100 on both lists (workflow_recipes.py). */
const PAGE_LIMIT_MAX = 100
const SORTS: readonly PlaybookSort[] = ['popularity', 'created_at', 'use_count', 'average_rating', 'name']
const RUN_STATUSES: readonly PlaybookRunStatus[] = ['pending', 'running', 'completed', 'failed', 'cancelled']

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/** The address a URL may carry: a well-formed template id, or a whole number; else null. */
function playbookIdOf(value: unknown): string | null {
  if (typeof value === 'number') return Number.isSafeInteger(value) && value >= 0 ? String(value) : null
  return typeof value === 'string' && PLAYBOOK_TEMPLATE_ID.test(value) ? value : null
}

function checkedId(value: unknown): string {
  return playbookIdOf(value) ?? refuse(NOT_A_PLAYBOOK_ID)
}

function checkedObject<T>(value: T, reason: string): T {
  return isPlainObject(value) ? value : refuse(reason)
}

function wholeNumber(value: unknown, min: number, max: number): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= min && value <= max ? value : undefined
}

function flag(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function search(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const text = value.trim().slice(0, SEARCH_MAX_LENGTH)
  return text.length > 0 ? text : undefined
}

function oneOf<T extends string>(value: unknown, known: readonly T[]): T | undefined {
  return known.find((entry) => entry === value)
}

/** The list filters the core hook may put in its query string. */
function listFilters(filters: PlaybookFilters | undefined): PlaybookFilters {
  return {
    is_featured: flag(filters?.is_featured),
    is_public: flag(filters?.is_public),
    search: search(filters?.search),
    skip: wholeNumber(filters?.skip, 0, Number.MAX_SAFE_INTEGER),
    limit: wholeNumber(filters?.limit, 1, PAGE_LIMIT_MAX),
    sort_by: oneOf(filters?.sort_by, SORTS),
  }
}

/** The run filters the core hook may put in its query string. */
function runFilters(filters: PlaybookRunFilters | undefined): PlaybookRunFilters {
  return {
    status: oneOf(filters?.status, RUN_STATUSES),
    skip: wholeNumber(filters?.skip, 0, Number.MAX_SAFE_INTEGER),
    limit: wholeNumber(filters?.limit, 1, PAGE_LIMIT_MAX),
  }
}

/** The address a query may ask for (undefined for none), and why it asks for nothing. */
function queryId(value: PlaybookId | null | undefined): { readonly id: string | undefined; readonly refusal: string | null } {
  if (value === null || value === undefined) return { id: undefined, refusal: null }
  const id = playbookIdOf(value)
  return id === null ? { id: undefined, refusal: NOT_A_PLAYBOOK_ID } : { id, refusal: null }
}

const createToCore: ToCore<PlaybookDraft, PlaybookDraft> = (body) => checkedObject(body, NOT_A_BODY)

const updateToCore: ToCore<PlaybookUpdateInput, CoreUpdateVariables> = (input) => ({
  playbookId: checkedId(input?.id),
  playbookData: checkedObject(input?.body, NOT_A_BODY),
})

const deleteToCore: ToCore<PlaybookId, string> = (id) => checkedId(id)

const runToCore: ToCore<PlaybookRunInput, CoreRunVariables> = (input) => {
  const playbookId = checkedId(input?.id)
  const inputs = input?.inputs
  const inputData = inputs === undefined ? undefined : { ...checkedObject(inputs, NOT_INPUTS) }
  return { playbookId, inputData, waitForMe: flag(input?.waitForMe) }
}

/** The workspace's playbooks, as the Playbooks tab lists them. */
export function useWorkflowPlaybooks(filters?: PlaybookFilters): PlaybookQuery<PlaybookList> {
  return useQueryView<PlaybookList>(useCoreWorkflowPlaybooks(listFilters(filters)))
}

/** One playbook with its steps; a null id asks for nothing. */
export function useWorkflowPlaybook(playbookId: PlaybookId | null): PlaybookQuery<Playbook> {
  const { id, refusal } = queryId(playbookId)
  return useQueryView<Playbook>(useCoreWorkflowPlaybook(id), refusal)
}

/** A playbook's runs, newest first; a null id asks for nothing. */
export function usePlaybookExecutions(playbookId: PlaybookId | null, filters?: PlaybookRunFilters): PlaybookQuery<PlaybookRunList> {
  const { id, refusal } = queryId(playbookId)
  return useQueryView<PlaybookRunList>(useCorePlaybookExecutions(id, runFilters(filters)), refusal)
}

/** Create a playbook; `run(body)`. */
export function useCreatePlaybook(): PlaybookAction<PlaybookDraft, PlaybookSaved> {
  return useActionView<PlaybookDraft, PlaybookDraft, PlaybookSaved>(useCoreCreatePlaybook(), createToCore)
}

/** Change a playbook; `run({ id, body })`. */
export function useUpdatePlaybook(): PlaybookAction<PlaybookUpdateInput, PlaybookSaved> {
  return useActionView<PlaybookUpdateInput, CoreUpdateVariables, PlaybookSaved>(useCoreUpdatePlaybook(), updateToCore)
}

/** Delete a playbook; `run(id)`. */
export function useDeletePlaybook(): PlaybookAction<PlaybookId, PlaybookDeleted> {
  return useActionView<PlaybookId, string, PlaybookDeleted>(useCoreDeletePlaybook(), deleteToCore)
}

/** Run a playbook now; `run({ id, inputs?, waitForMe? })`. */
export function useExecutePlaybook(): PlaybookAction<PlaybookRunInput, PlaybookRunStarted> {
  return useActionView<PlaybookRunInput, CoreRunVariables, PlaybookRunStarted>(useCoreExecutePlaybook(), runToCore)
}
