/**
 * PRD-261 A-12g — SDK area `pages/playbooks` (frontend/lib/theme-sdk/pages/playbooks.ts): the data the Playbooks
 * pages read, for a theme's own (or cloned) /playbooks page. Each hook wraps the core hook the Playbooks views read
 * (hooks/use-playbook-api.ts), so a theme's page asks the same endpoints with the same cache and invalidation as
 * the core page.
 *
 * A query hands back `PlaybookQuery` (null data until it answers), an action `PlaybookAction`; an error is its
 * message. A playbook is addressed by its template id (letters, digits, '.', '-' and '_', starting with a letter or
 * digit) or its number: a query given a malformed one asks for nothing and says so in `error`, and an action given
 * one throws a TypeError before anything is sent. A playbook body and a run's inputs must be plain objects. The
 * hooks need the host's providers above them, as every theme page the host draws has.
 */

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

/** The workspace's playbooks, as the Playbooks tab lists them. */
export declare function useWorkflowPlaybooks(filters?: PlaybookFilters): PlaybookQuery<PlaybookList>

/** One playbook with its steps; a null id asks for nothing. */
export declare function useWorkflowPlaybook(playbookId: PlaybookId | null): PlaybookQuery<Playbook>

/** A playbook's runs, newest first; a null id asks for nothing. */
export declare function usePlaybookExecutions(playbookId: PlaybookId | null, filters?: PlaybookRunFilters): PlaybookQuery<PlaybookRunList>

/** Create a playbook; `run(body)`. */
export declare function useCreatePlaybook(): PlaybookAction<PlaybookDraft, PlaybookSaved>

/** Change a playbook; `run({ id, body })`. */
export declare function useUpdatePlaybook(): PlaybookAction<PlaybookUpdateInput, PlaybookSaved>

/** Delete a playbook; `run(id)`. */
export declare function useDeletePlaybook(): PlaybookAction<PlaybookId, PlaybookDeleted>

/** Run a playbook now; `run({ id, inputs?, waitForMe? })`. */
export declare function useExecutePlaybook(): PlaybookAction<PlaybookRunInput, PlaybookRunStarted>
