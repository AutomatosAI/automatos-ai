import type { MissionApprovalTask, MissionApprovalWidgetData, ToolApprovalWidgetData } from '@/components/widgets/types'
import type { ApprovalAnswer, ApprovalView } from '@/lib/proactive/types'

/**
 * PRD-260 T-005c, PRD-261 A-11a (M28) — the approvals a turn raises, for any
 * theme's chat. The core chat (components/chatbot/chat.tsx) turns a
 * 'tool-data' part's `tool_approval` (a confirmation-gated action holding a
 * grant, PRD-193) and `mission_approval` (a Mission plan awaiting approval,
 * PRD-163) into the 'Action approval needed' and 'Mission plan — approval
 * needed' widgets; a theme's chat reads the same part the same way and shows
 * the same widgets inside Auto's reply. The payloads come off the wire
 * (result_formatter.py builds them), so every field is checked here, and the
 * ids are checked before they reach a request path. A filed card remembers the
 * reply that raised it and whether the owner has answered it or closed it; a
 * card whose reply left the conversation is loose, still open. Pure
 * (lib/chat-controller/approvals.ts holds the cards).
 *
 * PRD-268 US-C20c: either card may carry `approval_view` (ApprovalView@1, §6.11.4), the exact
 * action and the rule that asked. It is checked here too: a view whose fields aren't what they
 * say, or whose grant_id isn't this card's id, is dropped and the card stays exactly as it was.
 */

/** chat.tsx's widget titles. */
export const TOOL_APPROVAL_TITLE = 'Action approval needed'
export const MISSION_APPROVAL_TITLE = 'Mission plan — approval needed'
/** chat.tsx's widget source names (the card's badge). */
export const UNNAMED_ACTION = 'platform_action'
export const MISSION_SOURCE = 'platform_create_mission'

/** A grant id is a whole number; the backend sends the row id. */
const GRANT_ID = /^\d{1,15}$/
/** A Mission id goes into /api/missions/<id>/approve: nothing but an id's own characters. */
const MISSION_ID = /^[A-Za-z0-9_-]{1,128}$/
const RISK_TIERS: ReadonlySet<string> = new Set(['monitor', 'human_on_the_loop', 'human_in_the_loop'])
const ASKERS: ReadonlySet<string> = new Set(['auto', 'agent'])
const PREVIEW_KINDS: ReadonlySet<string> = new Set(['text', 'markdown', 'diff'])
/** K10: what a person may answer; any other answer (Always allow included) is dropped. */
const ANSWERS: ReadonlySet<string> = new Set<ApprovalAnswer>(['refuse', 'allow_once'])

export type ApprovalCard =
  | { readonly kind: 'tool'; readonly key: string; readonly data: ToolApprovalWidgetData }
  | { readonly kind: 'mission'; readonly key: string; readonly data: MissionApprovalWidgetData }

/** waiting: the owner hasn't answered; answered: approved or denied (the card shows the outcome); closed: gone. */
export type ApprovalStatus = 'waiting' | 'answered' | 'closed'

export type FiledApproval = ApprovalCard & {
  /** The reply that raised it: the card shows inside that reply. */
  readonly replyId: string
  readonly status: ApprovalStatus
  /** When it arrived (the widget's metadata). */
  readonly filedAt: Date
}

type Fields = Readonly<Record<string, unknown>>

function recordOf(value: unknown): Fields | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Fields) : null
}

function textOf(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function numberOf(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function grantIdOf(value: unknown): number | null {
  const text = typeof value === 'number' ? String(value) : textOf(value)
  return text !== undefined && GRANT_ID.test(text) ? Number(text) : null
}

type Oversight = Pick<ToolApprovalWidgetData, 'risk_tier' | 'risk_class' | 'oversight_rationale' | 'requires_approval'>

/** The AI-Act oversight fields both cards show (PRD-181 S5), each only when it is what it says. */
function oversightOf(fields: Fields): Oversight {
  const tier = textOf(fields.risk_tier)
  return {
    risk_tier: tier !== undefined && RISK_TIERS.has(tier) ? (tier as Oversight['risk_tier']) : undefined,
    risk_class: textOf(fields.risk_class),
    oversight_rationale: textOf(fields.oversight_rationale),
    requires_approval: typeof fields.requires_approval === 'boolean' ? fields.requires_approval : undefined,
  }
}

function askedByOf(value: unknown): ApprovalView['asked_by'] | null {
  const fields = recordOf(value)
  const kind = textOf(fields?.kind)
  const name = textOf(fields?.name)
  if (kind === undefined || !ASKERS.has(kind) || name === undefined) return null
  return { kind: kind as ApprovalView['asked_by']['kind'], name }
}

function actionOf(value: unknown): ApprovalView['action'] | null {
  const fields = recordOf(value)
  const verb = textOf(fields?.verb)
  const summary = textOf(fields?.summary)
  return verb !== undefined && summary !== undefined ? { verb, summary } : null
}

function ruleOf(value: unknown): ApprovalView['rule'] | null {
  const fields = recordOf(value)
  const id = textOf(fields?.id)
  const sentence = textOf(fields?.sentence)
  return id !== undefined && sentence !== undefined ? { id, sentence } : null
}

/** The target and the preview are optional: one that isn't what it says is left out, the view kept. */
function targetOf(value: unknown): Pick<ApprovalView, 'target'> {
  const fields = recordOf(value)
  const label = textOf(fields?.label)
  return label === undefined ? {} : { target: { label, detail: textOf(fields?.detail) } }
}

function previewOf(value: unknown): Pick<ApprovalView, 'preview'> {
  const fields = recordOf(value)
  const kind = textOf(fields?.kind)
  const body = textOf(fields?.body)
  if (kind === undefined || !PREVIEW_KINDS.has(kind) || body === undefined) return {}
  return { preview: { kind: kind as NonNullable<ApprovalView['preview']>['kind'], body } }
}

function answersOf(value: unknown): ApprovalAnswer[] {
  const answers = Array.isArray(value) ? value : []
  return answers.filter((answer): answer is ApprovalAnswer => typeof answer === 'string' && ANSWERS.has(answer))
}

/**
 * A card's ApprovalView@1, checked off the wire, or undefined: when it isn't an object, when a part
 * it needs (who asked, the action, the rule) isn't what it says, or when its grant_id isn't an id or
 * isn't `cardId` (the card's grant id, or its Mission's id). Never mutates.
 */
export function approvalViewOf(value: unknown, cardId: string): ApprovalView | undefined {
  const fields = recordOf(value)
  const id = typeof fields?.grant_id === 'number' ? String(fields.grant_id) : textOf(fields?.grant_id)
  // MISSION_ID admits an id's own characters, a grant id's digits among them.
  if (fields === null || id === undefined || !MISSION_ID.test(id) || id !== cardId) return undefined
  const askedBy = askedByOf(fields.asked_by)
  const action = actionOf(fields.action)
  const rule = ruleOf(fields.rule)
  if (askedBy === null || action === null || rule === null) return undefined
  const expiresAt = textOf(fields.expires_at)
  return {
    grant_id: id,
    asked_by: askedBy,
    action,
    ...targetOf(fields.target),
    ...previewOf(fields.preview),
    rule,
    answers: answersOf(fields.answers),
    ...(expiresAt === undefined ? {} : { expires_at: expiresAt }),
  }
}

/** `{ approval_view }` when the card carries a view that checks out, else nothing: today's card. */
function viewField(value: unknown, cardId: string): { approval_view?: ApprovalView } {
  const view = approvalViewOf(value, cardId)
  return view === undefined ? {} : { approval_view: view }
}

function toolCard(value: unknown): ApprovalCard | null {
  const fields = recordOf(value)
  const grantId = grantIdOf(fields?.grant_id)
  if (fields === null || grantId === null) return null
  const question = fields.question_md
  const data: ToolApprovalWidgetData = {
    grant_id: grantId,
    action: textOf(fields.action) ?? '',
    message: textOf(fields.message),
    question_md: typeof question === 'string' || question === null ? question : undefined,
    permission_level: textOf(fields.permission_level),
    params: recordOf(fields.params) ?? undefined,
    ...oversightOf(fields),
    ...viewField(fields.approval_view, String(grantId)),
  }
  return { kind: 'tool', key: `tool-${grantId}`, data }
}

function taskOf(value: unknown): MissionApprovalTask | null {
  const fields = recordOf(value)
  const title = textOf(fields?.title)
  if (fields === null || title === undefined) return null
  return {
    title,
    agent_role: textOf(fields.agent_role),
    sequence: numberOf(fields.sequence),
    temp_id: textOf(fields.temp_id),
    match_agent: textOf(fields.match_agent),
    match_reason: textOf(fields.match_reason),
    match_is_override: typeof fields.match_is_override === 'boolean' ? fields.match_is_override : undefined,
  }
}

/** chat.tsx's mission widget data, with the oversight fields the backend sends beside it. */
function missionCard(value: unknown): ApprovalCard | null {
  const fields = recordOf(value)
  const missionId = textOf(fields?.mission_id)
  if (fields === null || missionId === undefined || !MISSION_ID.test(missionId)) return null
  const tasks = (Array.isArray(fields.tasks) ? fields.tasks : []).flatMap((task) => taskOf(task) ?? [])
  const data: MissionApprovalWidgetData = {
    mission_id: missionId,
    goal: textOf(fields.goal) ?? '',
    state: textOf(fields.state),
    task_count: numberOf(fields.task_count) ?? tasks.length,
    tasks,
    cost_estimate_usd: numberOf(fields.cost_estimate_usd),
    cost_ceiling_usd: numberOf(fields.cost_ceiling_usd),
    approval_deadline_at: textOf(fields.approval_deadline_at),
    ...oversightOf(fields),
    ...viewField(fields.approval_view, missionId),
  }
  return { kind: 'mission', key: `mission-${missionId}`, data }
}

/** The approvals a forwarded onData part raises: none for any other part, or for one that isn't what it says. */
export function approvalsInPart(part: unknown): ApprovalCard[] {
  const forwarded = recordOf(part)
  const data = recordOf(forwarded?.data)
  if (forwarded?.type !== 'tool-data' || data === null) return []
  return [missionCard(data.mission_approval), toolCard(data.tool_approval)].flatMap((card) => card ?? [])
}

/** The card's source badge, as chat.tsx names it. */
export function sourceName(card: ApprovalCard): string {
  return card.kind === 'tool' ? card.data.action || UNNAMED_ACTION : MISSION_SOURCE
}

/** `filed` with the cards it doesn't hold yet, each under `replyId`, waiting. Never mutates. */
export function fileApprovals(
  filed: readonly FiledApproval[],
  raised: readonly ApprovalCard[],
  replyId: string,
  filedAt: Date,
): readonly FiledApproval[] {
  const known = new Set(filed.map((card) => card.key))
  const fresh = raised.filter((card) => !known.has(card.key))
  if (fresh.length === 0) return filed
  return [...filed, ...fresh.map((card): FiledApproval => ({ ...card, replyId, status: 'waiting', filedAt }))]
}

/** `filed` with one card's status changed (a closed card stays closed). Never mutates. */
export function withStatus(filed: readonly FiledApproval[], key: string, status: ApprovalStatus): readonly FiledApproval[] {
  const changes = (card: FiledApproval) => card.key === key && card.status !== status && card.status !== 'closed'
  return filed.some(changes) ? filed.map((card) => (changes(card) ? { ...card, status } : card)) : filed
}

/** This card still waits for the owner (not answered, not closed). */
export function isWaitingCard(card: FiledApproval): boolean {
  return card.status === 'waiting'
}

/** Whether a card still waits for the owner. */
export function isWaiting(filed: readonly FiledApproval[]): boolean {
  return filed.some(isWaitingCard)
}

/**
 * The open cards whose reply is no longer in the conversation: useChat drops
 * the reply when the request fails mid-turn (a reply you stopped is kept,
 * hooks/use-stopped-reply.ts), and Try again replaces it; a conversation
 * shown again (the chat screen mounted afresh) holds the server's copy of a
 * reply, under its own id. The grant or Mission
 * still waits on the server, so the card stays on screen (as Studio's widget
 * stays on its canvas) until the owner answers or closes it. Never mutates.
 */
export function looseApprovals(filed: readonly FiledApproval[], replyIds: ReadonlySet<string>): FiledApproval[] {
  return filed.filter((card) => card.status !== 'closed' && !replyIds.has(card.replyId))
}
