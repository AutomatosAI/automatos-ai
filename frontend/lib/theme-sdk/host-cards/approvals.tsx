/**
 * PRD-261 A-12a — the SDK's two approval cards, `cards.ToolApproval` and `cards.MissionApproval`.
 *
 * A theme places these and never draws its own (PRD §4.4). Each one is the host's own widget
 * (ToolApprovalWidget, MissionApprovalWidget), so Approve & run, Deny and Reject are the host's
 * calls, and each is already a protected surface (components/theme-host/protected-surface.tsx,
 * A-12c). The theme hands over the payload and where it came from; the host builds the widget's
 * metadata, and checks the payload the way the host's chat checks one off the wire
 * (lib/chat/approvals.ts approvalsInPart): the grant id and the Mission id reach a request path, so
 * a payload whose id is not an id throws here, before any request, and the theme's error boundary
 * reports it.
 */
import { useMemo, useState, type ReactElement } from 'react'

import { MissionApprovalWidget } from '@/components/widgets/MissionApprovalWidget'
import { ToolApprovalWidget } from '@/components/widgets/ToolApprovalWidget'
import type { WidgetMetadata } from '@/components/widgets/types'
import {
  approvalsInPart,
  MISSION_APPROVAL_TITLE,
  MISSION_SOURCE,
  TOOL_APPROVAL_TITLE,
  UNNAMED_ACTION,
  type ApprovalCard,
} from '@/lib/chat/approvals'

/** What the owner decided on a tool approval card. */
export type ApprovalDecision = 'approved' | 'denied'

/** The AI-Act oversight tier a card shows (PRD-181 S5). */
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
  /** The card closes itself after a decision, as it leaves Studio's canvas. */
  readonly onClose?: () => void
}

type ToolCard = Extract<ApprovalCard, { kind: 'tool' }>
type MissionCard = Extract<ApprovalCard, { kind: 'mission' }>

/** The stream part the host's chat reads the payloads from (lib/chat/approvals.ts). */
const TOOL_DATA_PART = 'tool-data'
const SOURCE_TYPE = 'tool'

function rejected(card: string, what: string): TypeError {
  return new TypeError(`Automatos theme SDK: ${card} needs ${what}.`)
}

function isToolCard(card: ApprovalCard): card is ToolCard {
  return card.kind === 'tool'
}

function isMissionCard(card: ApprovalCard): card is MissionCard {
  return card.kind === 'mission'
}

/** The tool card the host's chat would file for this payload; throws when its grant id is not an id. */
export function checkedToolCard(data: unknown): ToolCard {
  const card = approvalsInPart({ type: TOOL_DATA_PART, data: { tool_approval: data } }).find(isToolCard)
  if (card === undefined) throw rejected('ToolApproval', 'a whole-number grant_id')
  return card
}

/** The Mission card the host's chat would file for this payload; throws when its Mission id is not an id. */
export function checkedMissionCard(data: unknown): MissionCard {
  const card = approvalsInPart({ type: TOOL_DATA_PART, data: { mission_approval: data } }).find(isMissionCard)
  if (card === undefined) throw rejected('MissionApproval', 'a mission_id of letters, digits, - and _')
  return card
}

/** The widget's metadata: the badge, the conversation and when the card was raised. */
function useMetadata(origin: CardOrigin, fallbackSource: string): WidgetMetadata {
  const [raised] = useState(() => new Date())
  const { source, conversationId, createdAt } = origin
  return useMemo<WidgetMetadata>(
    () => ({ source: { type: SOURCE_TYPE, name: source || fallbackSource }, createdAt: createdAt ?? raised, conversationId }),
    [source, fallbackSource, createdAt, raised, conversationId],
  )
}

/** The host's tool approval card for `data`. */
export function ToolApproval({ data, title = TOOL_APPROVAL_TITLE, onDecided, onClose, ...origin }: ToolApprovalProps): ReactElement {
  const card = useMemo(() => checkedToolCard(data), [data])
  const metadata = useMetadata(origin, card.data.action || UNNAMED_ACTION)
  return (
    <ToolApprovalWidget id={card.key} title={title} data={card.data} metadata={metadata} onDecided={onDecided} onClose={onClose} />
  )
}

/** The host's Mission plan approval card for `data`. */
export function MissionApproval({ data, title = MISSION_APPROVAL_TITLE, onClose, ...origin }: MissionApprovalProps): ReactElement {
  const card = useMemo(() => checkedMissionCard(data), [data])
  const metadata = useMetadata(origin, MISSION_SOURCE)
  return <MissionApprovalWidget id={card.key} title={title} data={card.data} metadata={metadata} onClose={onClose} />
}
