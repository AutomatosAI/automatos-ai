/**
 * PRD-261 A-12a — SDK area `cards` (frontend/lib/theme-sdk/cards.ts): the host cards a theme places.
 *
 * A theme places approval, task, withdraw and Mission cards and never draws its own versions of them, and its
 * frame renders HeaderUtilities (PRD §4.4). Each card is the host's own, so its buttons make the host's calls.
 * ToolApproval and MissionApproval are protected surfaces: no theme stylesheet rule can hide them. Each throws
 * when its payload's id is not an id (a grant id is a whole number; a Mission id is letters, digits, - and _).
 *
 * SDK 1.3.0 (PRD-268) adds the proactive defaults, the host's own components for the proactive area's data: BriefCard,
 * SuggestionCard, StatusLabel, ReadinessList, CalibrationCard and ReplyChips. They draw the host's words, glyphs, AI
 * marks and controls in the look's tokens, and their calls are the host's (Start and Hide, Check again, the brief's
 * thumbs). Each throws, naming the card and the prop, when a prop is of the wrong kind; a row, an item or a line that
 * isn't one is left out. What a card hands back (a row, Start's answer, a calibration answer) is a frozen copy.
 */
import type { ReactElement, ReactNode } from 'react'

import type {
  Brief,
  BriefRow,
  BriefSection,
  Calibration,
  CalibrationAnswer,
  DisplayState,
  Readiness,
  ReadinessItem,
  Retry,
  StartAnswer,
  Suggestion,
} from './proactive'

/** What the owner decided on a tool approval card. */
export type ApprovalDecision = 'approved' | 'denied'

/** The AI-Act oversight tier a card shows. */
export type OversightTier = 'monitor' | 'human_on_the_loop' | 'human_in_the_loop'

/** PRD-268 §6.11.4 (K10): what a person may answer on a card. Always allow is not built. */
export type ApprovalAnswer = 'refuse' | 'allow_once'

/**
 * PRD-268 §6.11.4: `ApprovalView@1`, the exact action, its target, its preview and the rule that asked. A
 * card that carries one shows it before its buttons; one whose grant_id is not the card's id is dropped.
 */
export interface ApprovalViewData {
  readonly grant_id: string
  readonly asked_by: { readonly kind: 'auto' | 'agent'; readonly name: string }
  readonly action: { readonly verb: string; readonly summary: string }
  readonly target?: { readonly label: string; readonly detail?: string }
  readonly preview?: { readonly kind: 'text' | 'markdown' | 'diff'; readonly body: string }
  readonly rule: { readonly id: string; readonly sentence: string }
  readonly answers: readonly ApprovalAnswer[]
  readonly expires_at?: string
}

/** A confirmation-gated tool call waiting for the owner (the chat's `tool_approval` payload). */
export interface ToolApprovalData {
  readonly grant_id: number
  readonly action: string
  readonly message?: string
  readonly question_md?: string | null
  readonly permission_level?: string
  readonly params?: Readonly<Record<string, unknown>>
  readonly risk_tier?: OversightTier
  readonly risk_class?: string
  readonly oversight_rationale?: string
  readonly requires_approval?: boolean
  readonly approval_view?: ApprovalViewData
}

/** One task of a Mission plan. */
export interface MissionApprovalTask {
  readonly title: string
  readonly agent_role?: string
  readonly sequence?: number
  readonly temp_id?: string
  readonly match_agent?: string
  readonly match_reason?: string
  readonly match_is_override?: boolean
}

/** A Mission plan waiting for the owner (the chat's `mission_approval` payload). */
export interface MissionApprovalData {
  readonly mission_id: string
  readonly goal: string
  readonly state?: string
  readonly task_count: number
  readonly tasks: readonly MissionApprovalTask[]
  readonly cost_estimate_usd?: number
  readonly cost_ceiling_usd?: number
  readonly approval_deadline_at?: string
  readonly risk_tier?: OversightTier
  readonly risk_class?: string
  readonly oversight_rationale?: string
  readonly requires_approval?: boolean
  /** Its grant_id is the mission_id. */
  readonly approval_view?: ApprovalViewData
}

/** Where a card came from: its badge, its conversation and when it was raised. */
export interface CardOrigin {
  /** The badge: the tool that raised it (default: the action, or the Mission tool). */
  readonly source?: string
  /** The conversation the card came from. */
  readonly conversationId?: string
  /** When it was raised (default: when the card first rendered). */
  readonly createdAt?: Date
}

export interface ToolApprovalProps extends CardOrigin {
  readonly data: ToolApprovalData
  readonly title?: string
  /** Told once an approve or a deny went through (never after a failed call). */
  readonly onDecided?: (decision: ApprovalDecision) => void
  readonly onClose?: () => void
}

export interface MissionApprovalProps extends CardOrigin {
  readonly data: MissionApprovalData
  readonly title?: string
  /** The card closes itself after a decision. */
  readonly onClose?: () => void
}

/** A board ticket as the reply that filed or checked it carries it. */
export interface TaskCardData {
  readonly id: number
  readonly number?: string | null
  readonly title: string
  readonly status: string
  readonly assigned_agent: string
  readonly runtime?: string | null
  readonly last_tool?: string | null
  readonly files_touched?: number
  readonly exit_reason?: string | null
  readonly denials?: number
  readonly started_at?: string | null
  readonly completed_at?: string | null
}

export interface TaskCardProps {
  readonly card: TaskCardData
}

/** One request of Auto's still waiting on a yes. */
export interface WithdrawRequest {
  readonly grant_id: number
  readonly action?: string | null
  readonly reason?: string | null
}

/** The owner said no while those requests still wait: Withdraw denies each grant. */
export interface WithdrawOffer {
  readonly message: string
  readonly requests: readonly WithdrawRequest[]
}

export interface WithdrawOfferCardProps {
  readonly offer: WithdrawOffer
}

export interface MissionSuggestionMessage {
  readonly role: string
  readonly content: string
}

export interface MissionSuggestionCardProps {
  readonly goal: string
  readonly complexity: string
  readonly agentId?: number | string
  readonly chatId: string
  readonly recentMessages?: readonly MissionSuggestionMessage[]
}

export interface MissionCreatedCardProps {
  readonly missionId: string
}

export type FilePreviewType =
  | 'text'
  | 'image'
  | 'pdf'
  | 'code'
  | 'binary'
  | 'html'
  | 'markdown'
  | 'csv'
  | 'video'
  | 'audio'
  | 'docx'
  | 'xlsx'
  | 'json'

export interface FilePreviewProps {
  /** Text content (HTML source, Markdown, code, CSV, JSON, plain text). */
  readonly content?: string
  /** The API path of a binary file (PDF, image, video, audio, docx, xlsx), fetched with the caller's token. */
  readonly url?: string
  /** The kind of preview; inferred from the file name when omitted. */
  readonly previewType?: FilePreviewType
  readonly filename?: string
  readonly className?: string
}

export interface MarkdownViewProps {
  readonly children: string | null | undefined
  /** `compact` for side panels and cards; `default` for full-page reads. */
  readonly density?: 'default' | 'compact'
  /** Element overrides (the theme's own `a`, `img`, `pre`…), merged over the host's. */
  readonly components?: Readonly<Record<string, unknown>>
  /** Elements to drop, keeping their children (`['img']`). */
  readonly disallowedElements?: readonly string[]
  /** Drop raw HTML in the text instead of printing it as literal tags. Set it for model output. */
  readonly skipHtml?: boolean
  readonly className?: string
}

export interface MadeByProps {
  /** The agent that made it, when one did. */
  readonly agentName?: string | null
  /** Where it came from (`upload`, `mission`, `social_post`…), named when no agent made it. */
  readonly sourceType?: string | null
  readonly className?: string
}

export interface TrialExhaustedBannerProps {
  /** The typed error a refused request returned (`credits_exhausted`), before the workspace's balance catches up. */
  readonly errorCode?: string | null
}

/** The host's two AI labels: the page's, beside every in-app chat, and Auto's floating widget's. */
export type AiLabelId = 'page' | 'widget'

export interface AiLabelFenceProps {
  readonly children: ReactNode
  /** The label beside this chat, the page's or Auto's floating widget's: taken, changes nothing. */
  readonly label?: AiLabelId
  /** Taken, changes nothing: no composer waits on the host's AI label. */
  readonly held?: boolean
}

export interface CorePageProps {
  /** What to render where there is no core page (a theme route, or outside the host's page seam). */
  readonly children?: ReactNode
}

export interface BriefCardProps {
  readonly brief: Brief
  /** Listen (voice.BriefVoice: the brief the host holds) in the card's footer; default true. False when the theme places it itself. */
  readonly listen?: boolean
  /** A row's action was pressed: resolve to show "✓ done_label" in place, reject to leave the row as it was. */
  readonly onAction?: (row: BriefRow, section: BriefSection) => unknown
  /** A row with somewhere to go was opened; without it no row is a link. */
  readonly onOpen?: (row: BriefRow, section: BriefSection) => void
  /** "Show N more" was pressed for a folded section; without it the line is text. */
  readonly onShowMore?: (section: BriefSection) => void
  readonly className?: string
}

export type SuggestionCardVariant = 'card' | 'pinned'

export interface SuggestionCardProps {
  readonly suggestion: Suggestion
  /** The card (the default), or the one-line variant pinned above a composer. */
  readonly variant?: SuggestionCardVariant
  /** Start was accepted: what it made, for the theme to open. */
  readonly onStarted?: (answer: StartAnswer) => void
  /** The pinned line's title opens the full card. */
  readonly onOpen?: () => void
  /** What the theme draws under the suggestion's text, such as its evidence. */
  readonly children?: ReactNode
  readonly className?: string
}

export interface StatusLabelProps {
  /** The row's `display_state`; anything else draws nothing. */
  readonly state?: DisplayState | null
  /** The row's `next_step`, drawn after ▸. */
  readonly nextStep?: string | null
  readonly retry?: Retry | null
  /** What Try again runs; without it Try again is not drawn. */
  readonly onRetry?: (retry: Retry) => void
  /** The row's title, which names Try again for a screen reader. */
  readonly title?: string
  readonly className?: string
}

export interface ReadinessListProps {
  /** Readiness@1 as useReadiness read it; null while it loads. */
  readonly readiness: Readiness | null | undefined
  /** Why the read failed (useReadiness's error): one line in place of "Checking…". */
  readonly error?: string | null
  /** Rows only the browser knows (its microphone, its speech), drawn after the server's. */
  readonly clientItems?: readonly ReadinessItem[]
  readonly className?: string
}

export interface CalibrationCardProps {
  readonly calibration: Calibration
  /** Sends an answer (useAnswerCalibration's runAsync); without it Reject and Sounds right are disabled. */
  readonly onAnswer?: (answer: CalibrationAnswer) => Promise<void> | void
  /** Something's off: the theme asks the person what is wrong; without it the button is disabled. */
  readonly onSomethingOff?: () => void
  readonly className?: string
}

export interface ReplyChipsProps {
  /** The answers Auto offered with its message; the host draws two or three of at most 24 characters. */
  readonly chips: readonly string[]
  /** True while the message is Auto's last one in the thread. */
  readonly latest: boolean
  /** Sends the chip's text as the person's message; a rejected promise gives the chips back. */
  readonly onReply: (text: string) => void | Promise<unknown>
  readonly className?: string
}

/** The host's tool approval card for `data` (a protected surface). */
export declare function ToolApproval(props: ToolApprovalProps): ReactElement

/** The host's Mission plan approval card for `data` (a protected surface). */
export declare function MissionApproval(props: MissionApprovalProps): ReactElement

/** The host's live card for a board ticket. */
export declare function TaskCard(props: TaskCardProps): ReactElement

/** The host's withdraw-offer card. */
export declare function WithdrawOfferCard(props: WithdrawOfferCardProps): ReactElement

/** The host's card offering to turn the conversation into a Mission. */
export declare function MissionSuggestionCard(props: MissionSuggestionCardProps): ReactElement

/** The host's card for the Mission the chat created. */
export declare function MissionCreatedCard(props: MissionCreatedCardProps): ReactElement

/** The host's preview of a file. */
export declare function FilePreview(props: FilePreviewProps): ReactElement

/** The host's sanitising Markdown renderer; a theme never renders untrusted Markdown itself. */
export declare function MarkdownView(props: MarkdownViewProps): ReactElement

/** Who made a Deliverable, as the host's cards say it. */
export declare function MadeBy(props: MadeByProps): ReactElement

/**
 * The host's banner for used-up credits (work on Automatos's AI keys paused), with its way out (add a key);
 * it draws nothing while work runs. The name predates billing in credits.
 */
export declare function TrialExhaustedBanner(props: TrialExhaustedBannerProps): ReactElement

/** The local edition's first-run nudge; it draws nothing in SaaS, once dismissed, or once a key exists. */
export declare function FirstRunNudge(): ReactElement

/**
 * The host's header utilities: the credit balance (hosted edition), Help, notifications, the theme menu and the
 * account. Every frame renders it.
 */
export declare function HeaderUtilities(): ReactElement

/**
 * Once the host's AI label fence (PRD-264 EUAI-11a) around a theme's chat composer; now its children and
 * nothing else. No theme can hide the host's AI label (a hidden one is drawn again in the host's own layer),
 * so no composer waits on it: this disables nothing and draws no element. Kept so themes that use it load.
 */
export declare function AiLabelFence(props: AiLabelFenceProps): ReactElement

/** Core's own page for this route inside the theme's page, or `children` where there is none. */
export declare function CorePage(props: CorePageProps): ReactElement

/**
 * The host's brief card: the date, the greeting (the only serif), Auto's Read, the five sections with the server's
 * labels and counts, "Show N more", the empty sentence, the "Auto is an AI" caption, Listen and the thumbs.
 */
export declare function BriefCard(props: BriefCardProps): ReactElement

/**
 * The host's suggestion card: "Auto suggests", the text as Auto's words, what Start does, Start (with "about N
 * credits" once known) and Hide with its reasons, and the "Auto is an AI" caption; or its line pinned above a composer.
 */
export declare function SuggestionCard(props: SuggestionCardProps): ReactElement

/** A row's state as a glyph and a word (never colour alone), ▸ its next step, and Try again; nothing for an unknown state. */
export declare function StatusLabel(props: StatusLabelProps): ReactElement

/** What Auto depends on: each item's state as a glyph and a word, its fix as an in-app link, and Check again. */
export declare function ReadinessList(props: ReadinessListProps): ReactElement

/** What Auto knows so far, from its props: up to three lines with their source count and Reject, the offers, Sounds right. */
export declare function CalibrationCard(props: CalibrationCardProps): ReactElement

/** Two or three answers under Auto's last message, gone once a newer one exists; a tap hands its text to `onReply` once. */
export declare function ReplyChips(props: ReplyChipsProps): ReactElement
