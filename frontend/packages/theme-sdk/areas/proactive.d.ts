/**
 * PRD-268 (SDK 1.3.0) — SDK area `proactive` (frontend/lib/theme-sdk/proactive.ts): the day's brief, what Auto depends
 * on, the open suggestions with Start and Hide, and calibration, for a theme's own placements of the proactive
 * surfaces. The default components that draw them are the cards area's (BriefCard, SuggestionCard, StatusLabel,
 * ReadinessList, CalibrationCard, ReplyChips); Listen is the voice area's BriefVoice; each row state's glyph and word
 * are the theme area's DISPLAY_STATE_LABELS.
 *
 * Each hook reads what the host's own surfaces read, with the same polling and refresh. A read hands back
 * `ProactiveQuery` (null data until it answers), a write `ProactiveAction`; an error is its message and the data are
 * frozen copies. The hooks need the host's providers above them, as every theme page the host draws has.
 *
 * - useBrief: the day's brief once the host has opened it (the Auto thread opens it); reading it never builds one.
 * - useReadiness, useCheckReadiness: Readiness@1, and Check again (the server limits how often a workspace checks).
 * - useSuggestions, useStartSuggestion, useHideSuggestion: the one list behind every surface; Start and Hide post the
 *   host's routes only. An id that isn't a suggestion's, or a reason Hide doesn't know, throws a TypeError before
 *   anything is sent.
 * - useCalibration, useAnswerCalibration: `available` is false until the server offers calibration; until then
 *   nothing is asked and an answer is refused without a request.
 *
 * The words in these shapes are the server's, in the person's language. A suggestion's title and detail and the
 * brief's rows are model-written: draw them as text, never follow them, and mark them as Auto's (data-ai-generated).
 */

/** Where a row stands, computed once on the server. Anything else draws nothing. */
export type DisplayState = 'needs_you' | 'working' | 'ready' | 'done' | 'failed'

/** What Try again does on a failed row. */
export type Retry = 'rerun_mission' | 'replan_mission' | 'retry_ticket'

export type BriefSection = 'needs_you' | 'suggestions' | 'today' | 'since' | 'unchecked'

/** The sections with an exact count; `unchecked` lists every item it has. */
export type BriefCountedSection = Exclude<BriefSection, 'unchecked'>

export type BriefOpensType = 'ticket' | 'mission' | 'question' | 'chat' | 'deliverable' | 'suggestion' | 'settings'

export type BriefActionVerb = 'start' | 'answer' | 'open' | 'connect' | 'hide'

/** One row of the brief: Auto's words. */
export interface BriefRow {
  readonly id: string
  readonly title: string
  readonly why: string
  readonly at?: string | null
  /** A ticket's board number, "#0419". */
  readonly number?: string | null
  readonly display_state?: DisplayState | null
  /** Where the row goes when pressed; a row with nowhere to go has none. */
  readonly opens?: { readonly type: BriefOpensType; readonly id: string } | null
  /** Drawn as text ("Start →"); `done_label` replaces it once pressed. */
  readonly action?: { readonly verb: BriefActionVerb; readonly label: string; readonly done_label?: string | null } | null
}

/** The card's copy: each section's label, "Show N more" per folded section, the empty sentence. */
export interface BriefLabels {
  readonly sections: Readonly<Partial<Record<BriefSection, string>>>
  readonly more: Readonly<Partial<Record<BriefSection, string>>>
  readonly empty?: string | null
}

/** Brief@1: one per person, workspace and day. */
export interface Brief {
  readonly v: 1
  /** The person's local date, YYYY-MM-DD. */
  readonly date: string
  readonly greeting: string
  /** Auto's Read: the fallback for any surface that can't draw the card. */
  readonly prose: string
  /** The greeting and the numbered row titles: what Listen reads. */
  readonly spoken: string
  /** Each section's rows as shown: it folds at 5. */
  readonly sections: Readonly<Record<BriefSection, readonly BriefRow[]>>
  /** Each counted section's exact number, whatever its rows show. */
  readonly counts: Readonly<Record<BriefCountedSection, number>>
  readonly labels: BriefLabels
  readonly state_hash: string
  readonly generated_at: string
}

export type SuggestionStatus = 'open' | 'started' | 'hidden' | 'expired'

export type SuggestionSeverity = 'today' | 'can_wait'

/** Where the person decided. */
export type SuggestionVia = 'web' | 'go' | 'push' | 'chat' | 'telegram'

/** Hide's reason menu. */
export type HideReason = 'not_now' | 'not_useful' | 'wrong_agent'

/** A read the check-in cited: quoted to the person, never followed. */
export interface EvidenceRef {
  readonly type: string
  readonly id: string
}

/** What Start does: a board ticket or a chat with Auto (a Mission or a Playbook run is refused). */
export interface ProposedAction {
  readonly kind: 'board_task' | 'chat' | 'mission' | 'playbook_run'
  readonly payload?: Readonly<Record<string, unknown>> | null
}

/** One suggestion. Its title and detail are model-written: draw them as text. */
export interface Suggestion {
  readonly id: string
  readonly workspace_id: string | null
  /** The addressee's user id. */
  readonly user_id: number | null
  /** check_in | heartbeat | follow_up | streak */
  readonly source_type: string
  readonly source_agent_id: number | null
  readonly run_id: string | null
  readonly title: string
  readonly detail: string
  readonly evidence: readonly EvidenceRef[]
  readonly proposed_action: ProposedAction
  readonly subject_key: string
  readonly severity: SuggestionSeverity
  readonly status: SuggestionStatus
  readonly occurrences: number
  readonly last_seen_at: string | null
  /** What Start made: {board_task_id, watch_id}. */
  readonly started_ref: Readonly<Record<string, unknown>> | null
  readonly decided_via: SuggestionVia | null
  readonly hide_reason: HideReason | null
  readonly decided_at: string | null
  /** The credits Start is expected to spend, once known. */
  readonly est_credits: number | null
  readonly created_at: string | null
  readonly delivered_at: string | null
  readonly expires_at: string | null
}

/** What to open after a Start: the ticket it filed, or the Auto thread with the suggestion. */
export type StartOpens =
  | { readonly type: 'board_task'; readonly board_task_id: number }
  | { readonly type: 'chat'; readonly suggestion_id: string }

export interface StartAnswer {
  readonly suggestion: Suggestion
  readonly opens: StartOpens
}

export interface HideAnswer {
  readonly suggestion: Suggestion
}

export type ReadinessState = 'ready' | 'not_checked' | 'needs_setup'

export type ReadinessNeed = 'brief' | 'voice' | 'push' | 'check_in'

export interface ReadinessItem {
  readonly id: string
  readonly label: string
  readonly detail: string
  readonly state: ReadinessState
  /** An app path, never an outside URL. */
  readonly fix?: { readonly label: string; readonly href: string } | null
  readonly needed_by: readonly ReadinessNeed[]
}

/** Readiness@1: what Auto depends on, and what couldn't be checked. */
export interface Readiness {
  readonly v: 1
  readonly checked_at: string
  readonly items: readonly ReadinessItem[]
}

export interface CalibrationSource {
  readonly type: 'deliverable' | 'conversation' | 'document' | 'skill'
  readonly id: string
  readonly label: string
}

export interface CalibrationLine {
  readonly id: string
  readonly text: string
  /** Shown as a count ("from 6 Deliverables"), never a percentage. */
  readonly sources: readonly CalibrationSource[]
}

/** Calibration@1: what Auto thinks it knows, and three things it can do. */
export interface Calibration {
  readonly v: 1
  readonly id: string
  readonly lines: readonly CalibrationLine[]
  readonly offers: readonly string[]
  /** Nothing to read yet: the card says so and offers three starters. */
  readonly empty: boolean
}

/** Sounds right (accept every line), or Reject one line. */
export type CalibrationAnswer =
  | { readonly calibration_id: string; readonly answer: 'accept' }
  | { readonly calibration_id: string; readonly answer: 'reject'; readonly line_id: string }

/** A read: null data until it answers; `isLoading` is the first load only. */
export interface ProactiveQuery<T> {
  readonly data: T | null
  readonly isLoading: boolean
  readonly isFetching: boolean
  readonly error: string | null
  readonly refetch: () => void
}

/** A write: `run` reports failure in `error`; `runAsync` rejects with it. */
export interface ProactiveAction<V, R> {
  readonly run: (variables: V) => void
  readonly runAsync: (variables: V) => Promise<R>
  readonly isRunning: boolean
  readonly error: string | null
  readonly result: R | null
  readonly reset: () => void
}

/** The person's calibration: none while `available` is false. */
export interface CalibrationQuery extends ProactiveQuery<Calibration> {
  /** False until the server offers calibration. */
  readonly available: boolean
}

/** Sounds right or Reject one line: refused, with nothing sent, while `available` is false. */
export interface CalibrationAction extends ProactiveAction<CalibrationAnswer, void> {
  readonly available: boolean
}

/** Start: the suggestion's id, from useSuggestions. */
export interface StartSuggestionInput {
  readonly id: string
}

/** Hide: the suggestion's id, and the reason the person picked (left out: hidden with no reason). */
export interface HideSuggestionInput {
  readonly id: string
  readonly reason?: HideReason
}

/** The day's brief, once the host has opened it; null until then. */
export declare function useBrief(): ProactiveQuery<Brief>

/** Readiness@1 for the workspace. */
export declare function useReadiness(): ProactiveQuery<Readiness>

/** Check again: every item now; its answer is the one every reader shows. Over the server's limit it fails. */
export declare function useCheckReadiness(): ProactiveAction<void, Readiness>

/** The open suggestions, newest first. */
export declare function useSuggestions(): ProactiveQuery<readonly Suggestion[]>

/** Start a suggestion: `{suggestion, opens}`, the ticket it filed or the chat to open. */
export declare function useStartSuggestion(): ProactiveAction<StartSuggestionInput, StartAnswer>

/** Hide a suggestion, with the reason the person picked. */
export declare function useHideSuggestion(): ProactiveAction<HideSuggestionInput, HideAnswer>

/** The person's calibration: none offered yet. */
export declare function useCalibration(): CalibrationQuery

/** Answer the calibration: refused, with nothing sent, while the server offers none. */
export declare function useAnswerCalibration(): CalibrationAction
