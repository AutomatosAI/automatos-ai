/**
 * PRD-261 A-12a — SDK area `cards` (frontend/lib/theme-sdk/cards.ts): the host cards a theme places.
 *
 * A theme places approval, task, withdraw and Mission cards and never draws its own versions of them, and its
 * frame renders HeaderUtilities (PRD §4.4). Each card is the host's own, so its buttons make the host's calls.
 * ToolApproval and MissionApproval are protected surfaces: no theme stylesheet rule can hide them. Each throws
 * when its payload's id is not an id (a grant id is a whole number; a Mission id is letters, digits, - and _).
 */
import type { ReactElement, ReactNode } from 'react'

/** What the owner decided on a tool approval card. */
export type ApprovalDecision = 'approved' | 'denied'

/** The AI-Act oversight tier a card shows. */
export type OversightTier = 'monitor' | 'human_on_the_loop' | 'human_in_the_loop'

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
  /** The label beside this chat: the page's (the default) or Auto's floating widget's. */
  readonly label?: AiLabelId
  /** Keep the composer disabled until that label has been checked once. */
  readonly held?: boolean
}

export interface CorePageProps {
  /** What to render where there is no core page (a theme route, or outside the host's page seam). */
  readonly children?: ReactNode
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
 * The host's AI label fence (PRD-264 EUAI-11a) around a theme's chat composer: every control inside it is
 * disabled while a theme keeps the host's AI label beside that chat from being seen. It draws no box. Wrap the
 * composer only, never a whole chat screen (the approval cards in a reply stay clickable).
 */
export declare function AiLabelFence(props: AiLabelFenceProps): ReactElement

/** Core's own page for this route inside the theme's page, or `children` where there is none. */
export declare function CorePage(props: CorePageProps): ReactElement
