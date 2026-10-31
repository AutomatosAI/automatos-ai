/**
 * PRD-268 §6.11 (US-C21b) — the proactive shapes: one typed source for the web app and the theme SDK.
 *
 * Each mirrors the server's own model:
 * - `Brief` (Brief@1) and `BriefRow`: services/brief/model.py, posted as a `brief@1` message part;
 * - `Suggestion`: W1's ProactiveSuggestion.to_dict (core/models/proactive.py), the rows of
 *   GET /api/proactive/suggestions and the answers of Start and Hide (api/proactive.py);
 * - `Readiness` (Readiness@1): services/readiness/contract.py, GET /api/readiness;
 * - `ApprovalView` (ApprovalView@1): modules/tools/formatting/approval_view.py, on the approval cards;
 * - `Calibration` (Calibration@1): §6.11.5, US-C22's routes (GET and POST /api/auto/calibration), off until CALIBRATION_ENABLED;
 * - `DisplayState`: services/proactive/display_state.py, on needs-you, Mission and board rows.
 *
 * Every id is a string and a field a reader doesn't know is ignored. The words in these shapes are
 * the server's, in the person's language (K16): a client draws them as text and composes none of its
 * own, beyond DISPLAY_STATE_LABELS (display-state.ts) and its control labels.
 */

/** §6.11.1: where a row stands, computed once on the server. Anything else draws nothing. */
export const DISPLAY_STATES = ['needs_you', 'working', 'ready', 'done', 'failed'] as const
export type DisplayState = (typeof DISPLAY_STATES)[number]

/** What Try again does on a failed row (US-C17). */
export type Retry = 'rerun_mission' | 'replan_mission' | 'retry_ticket'

/** The status fields W1 adds to needs-you, Mission and board rows (US-C17a/b); null when unknown. */
export interface RowStatus {
  display_state?: DisplayState | null
  /** At most 90 characters, in the person's language. */
  next_step?: string | null
  retry?: Retry | null
}

// ---------------------------------------------------------------- Brief@1 (§6.11.2, K11)

export const BRIEF_SECTIONS = ['needs_you', 'suggestions', 'today', 'since', 'unchecked'] as const
export type BriefSection = (typeof BRIEF_SECTIONS)[number]
/** The sections with an exact count; `unchecked` lists every item it has. */
export type BriefCountedSection = Exclude<BriefSection, 'unchecked'>

export type BriefOpensType = 'ticket' | 'mission' | 'question' | 'chat' | 'deliverable' | 'suggestion' | 'settings'
export type BriefActionVerb = 'start' | 'answer' | 'open' | 'connect' | 'hide'

export interface BriefRow {
  id: string
  title: string
  why: string
  at?: string | null
  /** A ticket's board number, "#0419". */
  number?: string | null
  display_state?: DisplayState | null
  /** Where the row goes when pressed; a row with nowhere to go has none and draws no chevron. */
  opens?: { type: BriefOpensType; id: string } | null
  /** Drawn as text ("Start →"); `done_label` replaces it once pressed. */
  action?: { verb: BriefActionVerb; label: string; done_label?: string | null } | null
}

/** The card's copy (K16): each section's label, "Show N more" per folded section, the empty sentence. */
export interface BriefLabels {
  sections: Partial<Record<BriefSection, string>>
  more: Partial<Record<BriefSection, string>>
  empty?: string | null
}

/**
 * The card's words whatever its rows (US-C23): each section's label, the empty sentence, and "Show {n} more"
 * unfilled, for a client that folds the rows its own way. The default card draws the server's fold, so its
 * "Show N more" stays `labels.more`. A brief built before W3, or one a theme hands the card, may have no copy;
 * the card then reads `labels`.
 */
export interface BriefCopy {
  section_labels: Partial<Record<BriefSection, string>>
  empty_sentence?: string | null
  show_more_template?: string | null
}

export interface Brief {
  v: 1
  /** The person's local date, YYYY-MM-DD. */
  date: string
  greeting: string
  /** Auto's Read, unchanged: the fallback for any surface that can't draw the card. */
  prose: string
  /** The greeting and the numbered row titles, the words Listen speaks (K11). */
  spoken: string
  /** Each section's rows as shown: it folds at 5, never at 6. */
  sections: Record<BriefSection, BriefRow[]>
  /** Each counted section's exact number, whatever its rows show. */
  counts: Record<BriefCountedSection, number>
  labels: BriefLabels
  copy?: BriefCopy | null
  state_hash: string
  generated_at: string
}

/** The message parts a background post carries after its text part (services/chat_parts.py). */
export const BRIEF_PART = 'brief@1'
export const SUGGESTION_PART = 'suggestion@1'
export interface BriefPart {
  type: typeof BRIEF_PART
  brief: Brief
}
export interface SuggestionPart {
  type: typeof SUGGESTION_PART
  suggestion_id: string
}

/** POST /api/proactive/brief/open's body: the browser's local date, its IANA zone and its language. */
export interface BriefDay {
  local_date: string
  timezone: string
  locale?: string
}

/** Where the day's brief message is. */
export interface BriefMessageRef {
  chat_id: string
  message_id: string
}

/** POST /api/proactive/brief/open's answer: the day's message, or pending while another open builds it. */
export type BriefOpened = (BriefMessageRef & { posted: boolean; pending?: false }) | { posted: false; pending: true }

// ---------------------------------------------------------------- Suggestions (§6.2.4, §6.2.5)

export type SuggestionStatus = 'open' | 'started' | 'hidden' | 'expired'
export type SuggestionSeverity = 'today' | 'can_wait'
/** Where the person decided; the server says web or go when it is left out. */
export type SuggestionVia = 'web' | 'go' | 'push' | 'chat' | 'telegram'

/** The reason menu's three reasons (POST …/hide). */
export const HIDE_REASONS = ['not_now', 'not_useful', 'wrong_agent'] as const
export type HideReason = (typeof HIDE_REASONS)[number]

/** A read the check-in cited: quoted to the person, never followed. */
export interface EvidenceRef {
  type: string
  id: string
}

/** What Start does: a board ticket or a chat with Auto in W2 (a Mission or a playbook run answers 409). */
export interface ProposedAction {
  kind: 'board_task' | 'chat' | 'mission' | 'playbook_run'
  payload?: Record<string, unknown> | null
}

/** One suggestion, as W1's to_dict writes it. Its title and detail are model-written: draw them as text. */
export interface Suggestion {
  id: string
  workspace_id: string | null
  /** The addressee's users.id. */
  user_id: number | null
  /** check_in | heartbeat | follow_up | streak */
  source_type: string
  source_agent_id: number | null
  run_id: string | null
  title: string
  detail: string
  evidence: EvidenceRef[]
  proposed_action: ProposedAction
  subject_key: string
  severity: SuggestionSeverity
  status: SuggestionStatus
  occurrences: number
  last_seen_at: string | null
  /** What Start made: {board_task_id, watch_id}. */
  started_ref: Record<string, unknown> | null
  decided_via: SuggestionVia | null
  hide_reason: HideReason | null
  decided_at: string | null
  /** Known only once US-C15 computes it; the web shows "about N credits" then. */
  est_credits: number | null
  created_at: string | null
  delivered_at: string | null
  expires_at: string | null
}

/** What the client opens after a Start: the ticket it filed, or the Auto thread with the suggestion. */
export type StartOpens = { type: 'board_task'; board_task_id: number } | { type: 'chat'; suggestion_id: string }

export interface StartAnswer {
  suggestion: Suggestion
  opens: StartOpens
}

export interface HideAnswer {
  suggestion: Suggestion
}

/** A refusal the proactive routes answer: `{code, detail}`, the detail a sentence in the person's language. */
export interface ProactiveRefusal {
  code: string | null
  detail: string
  status: number | null
}

// ---------------------------------------------------------------- Readiness@1 (§6.11.3)

export type ReadinessState = 'ready' | 'not_checked' | 'needs_setup'
export type ReadinessNeed = 'brief' | 'voice' | 'push' | 'check_in'

export interface ReadinessItem {
  id: string
  label: string
  detail: string
  state: ReadinessState
  /** An app path, never an outside URL. */
  fix?: { label: string; href: string } | null
  needed_by: ReadinessNeed[]
}

export interface Readiness {
  v: 1
  checked_at: string
  items: ReadinessItem[]
}

// ---------------------------------------------------------------- ApprovalView@1 (§6.11.4, K10)

/** What a person may answer. Always allow is not built (§12-Q17). */
export type ApprovalAnswer = 'refuse' | 'allow_once'

/**
 * The exact action, its target, its preview and the rule that asked, built by the server and
 * checked off the wire by lib/chat/approvals.ts (approvalViewOf). The preview is the call's own
 * payload, shown as text.
 */
export interface ApprovalView {
  grant_id: string
  asked_by: { kind: 'auto' | 'agent'; name: string }
  action: { verb: string; summary: string }
  target?: { label: string; detail?: string }
  preview?: { kind: 'text' | 'markdown' | 'diff'; body: string }
  rule: { id: string; sentence: string }
  answers: ApprovalAnswer[]
  expires_at?: string
}

// ---------------------------------------------------------------- Calibration@1 (§6.11.5)

export interface CalibrationSource {
  type: 'deliverable' | 'conversation' | 'document' | 'skill'
  id: string
  label: string
}

export interface CalibrationLine {
  id: string
  text: string
  /** Shown as a count ("from 6 Deliverables"), never a percentage. */
  sources: CalibrationSource[]
}

export interface Calibration {
  v: 1
  id: string
  lines: CalibrationLine[]
  /** Three things Auto can do, from the installed skills. */
  offers: string[]
  /** Nothing to read yet: the card says so and offers three starters. */
  empty: boolean
}

/** Sounds right (accept every line), or Reject one line. */
export type CalibrationAnswer =
  | { calibration_id: string; answer: 'accept' }
  | { calibration_id: string; answer: 'reject'; line_id: string }
