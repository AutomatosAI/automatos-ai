/**
 * PRD-268 §6.11, K12 (US-C21e) — the proactive shapes as the SDK hands them to a theme: a row's status, Brief@1, a
 * suggestion and what Start and Hide answer, Readiness@1 and Calibration@1. Not an area: the proactive area
 * (proactive.ts) re-exports them, the cards area's default components take them as props, and
 * packages/theme-sdk/areas/proactive.d.ts declares them again for themes.
 *
 * Each mirrors the host's own (lib/proactive/types.ts), read-only, so the host can change its types without changing
 * the SDK's. Every id is a string and a field a reader doesn't know is ignored. The words in them are the server's,
 * in the person's language; a suggestion's title and detail and the brief's rows are model-written: a theme draws
 * them as text and never follows them.
 */

/** §6.11.1: where a row stands, computed once on the server. Anything else draws nothing. */
export type DisplayState = 'needs_you' | 'working' | 'ready' | 'done' | 'failed'

/** What Try again does on a failed row. */
export type Retry = 'rerun_mission' | 'replan_mission' | 'retry_ticket'

// ---------------------------------------------------------------- Brief@1 (§6.11.2)

export type BriefSection = 'needs_you' | 'suggestions' | 'today' | 'since' | 'unchecked'
/** The sections with an exact count; `unchecked` lists every item it has. */
export type BriefCountedSection = Exclude<BriefSection, 'unchecked'>
export type BriefOpensType = 'ticket' | 'mission' | 'question' | 'chat' | 'deliverable' | 'suggestion' | 'settings'
export type BriefActionVerb = 'start' | 'answer' | 'open' | 'connect' | 'hide'

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

// ---------------------------------------------------------------- Suggestions (§6.2.4, §6.2.5)

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

// ---------------------------------------------------------------- Readiness@1 (§6.11.3)

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

export interface Readiness {
  readonly v: 1
  readonly checked_at: string
  readonly items: readonly ReadinessItem[]
}

// ---------------------------------------------------------------- Calibration@1 (§6.11.5)

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

export interface Calibration {
  readonly v: 1
  readonly id: string
  readonly lines: readonly CalibrationLine[]
  /** Three things Auto can do. */
  readonly offers: readonly string[]
  /** Nothing to read yet: the card says so and offers three starters. */
  readonly empty: boolean
}

/** Sounds right (accept every line), or Reject one line. */
export type CalibrationAnswer =
  | { readonly calibration_id: string; readonly answer: 'accept' }
  | { readonly calibration_id: string; readonly answer: 'reject'; readonly line_id: string }
