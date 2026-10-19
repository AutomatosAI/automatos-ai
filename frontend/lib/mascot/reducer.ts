import { toolActivity, type ToolActivity } from './tools'
import {
  type ApprovalEvent,
  type ClientEvent,
  type MascotEvent,
  type MascotName,
  type MascotState,
  ONE_SHOTS,
  type Pose,
  RANK,
  type ReceiptsEvent,
  type StreamEvent,
  type ToolEndEvent,
  type ToolStartEvent,
  URGENT,
} from './types'

/**
 * PRD-260 D6 — Auto's state machine, ported from Automatos Play's
 * lib/mascot/reducer.ts: a pure (state, event) => state with no timers and no
 * DOM. The rules (the state table, the priority order, the one-shots) are
 * Play's mascot spec. hooks/use-mascot-player.ts owns the clip boundary, the
 * hold time and reduced motion, and reports clip-ended when a one-shot's clip
 * ends. Unchanged input returns the same object. While an approval card waits
 * for the owner (Play's chat, T-005c), Auto rests on listening where he would
 * otherwise go back to idle. A Deliverable made mid-turn is celebrated when it
 * lands, so the turn's done receipt adds no second celebration at finish, and
 * Stop drops the celebration a done receipt promised (P260-FX-04).
 */

export const INITIAL_MASCOT: MascotState = {
  mascot: 'idle',
  rank: RANK.idle,
  queued: null,
  openTools: [],
  turnActive: false,
  celebrateOnFinish: false,
  celebratedThisTurn: false,
  backendDown: false,
  approvalWaiting: false,
  pastThinking: false,
}

const pose = (mascot: MascotName, rank: number): Pose => ({ mascot, rank })

const IDLE = pose('idle', RANK.idle)
const THINKING = pose('thinking', RANK.thinking)
const WORKING = pose('working', RANK.working)
const READING = pose('reading', RANK.tool)
const OOPS = pose('oops', RANK.oops)
const CELEBRATE = pose('celebrate', RANK.oneShot)
const HELLO = pose('hello', RANK.oneShot)
const WAITING = pose('listening', RANK.waiting)
const TYPING = pose('listening', RANK.typing)
const SLEEPING = pose('sleeping', RANK.rest)
const LOST = pose('lost', RANK.rest)
const LOADING = pose('loading', RANK.idle)

const ACTIVITY_POSE: Readonly<Record<ToolActivity, Pose>> = {
  searching: pose('searching', RANK.tool),
  reading: READING,
  working: WORKING,
}

const playingOneShot = (state: MascotState): boolean => ONE_SHOTS.has(state.mascot)

/** Where Auto goes when nothing else is going on: listening while an approval waits for the owner, else idle. */
const rest = (state: MascotState): Pose => (state.approvalWaiting ? WAITING : IDLE)

/** The pose that is on screen, or that will be once the current one-shot ends. */
export function pendingPose(state: MascotState): Pose {
  if (playingOneShot(state)) return state.queued ?? rest(state)
  return pose(state.mascot, state.rank)
}

function show(state: MascotState, next: Pose): MascotState {
  if (state.mascot === next.mascot && state.rank === next.rank && state.queued === null) return state
  return { ...state, mascot: next.mascot, rank: next.rank, queued: null }
}

/**
 * What plays when the current one-shot ends. The one-shot on screen never
 * queues behind itself: its finished clip would never end again, and Auto
 * would freeze on it (lib/mascot/player.ts retarget).
 */
function enqueue(state: MascotState, next: Pose): MascotState {
  if (ONE_SHOTS.has(next.mascot) && next.mascot === state.mascot) return state
  const queued = state.queued
  if (queued && queued.mascot === next.mascot && queued.rank === next.rank) return state
  return { ...state, queued: next }
}

/** Raise the state when the new reason outranks the pending one (rule 1). */
function propose(state: MascotState, next: Pose): MascotState {
  if (URGENT.has(next.mascot) && next.rank >= state.rank) return show(state, next)
  return next.rank >= pendingPose(state).rank ? settle(state, next) : state
}

/** Set the state whatever its rank (a downgrade). A playing one-shot finishes first (rule 5). */
function settle(state: MascotState, next: Pose): MascotState {
  return playingOneShot(state) ? enqueue(state, next) : show(state, next)
}

const isToolPose = (p: Pose): boolean => p.rank === RANK.tool || p.rank === RANK.working

/** The highest-ranked tool still running, or where Auto rests when none is. */
function afterTools(state: MascotState): Pose {
  const busiest = state.openTools
    .map((tool) => ACTIVITY_POSE[tool.activity])
    .reduce((best, next) => (next.rank > best.rank ? next : best), IDLE)
  return busiest === IDLE ? rest(state) : busiest
}

/** A turn's words started or a tool ran: the turn is past thinking (only a turn's own events mark it). */
function markPastThinking(state: MascotState): MascotState {
  return state.turnActive && !state.pastThinking ? { ...state, pastThinking: true } : state
}

function onToolStart(state: MascotState, event: ToolStartEvent): MascotState {
  const activity = toolActivity(event.toolName)
  const marked = markPastThinking(state)
  const opened = { ...marked, openTools: [...marked.openTools, { toolCallId: event.toolCallId, activity }] }
  return propose(opened, ACTIVITY_POSE[activity])
}

function onToolEnd(state: MascotState, event: ToolEndEvent): MascotState {
  const openTools = state.openTools.filter((tool) => tool.toolCallId !== event.toolCallId)
  const closed = openTools.length === state.openTools.length ? state : { ...state, openTools }
  if (event.waiting) return propose(closed, WAITING)
  if (!event.success && !event.skipped) return propose(closed, OOPS)
  return isToolPose(pendingPose(closed)) ? settle(closed, afterTools(closed)) : closed
}

function onReceipts(state: MascotState, event: ReceiptsEvent): MascotState {
  const done = event.receipts.some((r) => r.status === 'done')
  const marked = done && !state.celebrateOnFinish ? { ...state, celebrateOnFinish: true } : state
  return event.receipts.some((r) => r.status === 'refused') ? propose(marked, OOPS) : marked
}

/** What a turn's start and its end both clear. */
const TURN_RESET: Pick<MascotState, 'openTools' | 'celebrateOnFinish' | 'celebratedThisTurn' | 'pastThinking'> = {
  openTools: [],
  celebrateOnFinish: false,
  celebratedThisTurn: false,
  pastThinking: false,
}

function endTurn(state: MascotState): MascotState {
  return { ...state, ...TURN_RESET, turnActive: false }
}

function onFinish(state: MascotState): MascotState {
  const pending = pendingPose(state)
  const quiet = pending.rank >= RANK.thinking && pending.rank <= RANK.tool ? settle(state, rest(state)) : state
  const ended = endTurn(quiet)
  return state.celebrateOnFinish && !state.celebratedThisTurn ? propose(ended, CELEBRATE) : ended
}

/** A text delta ends thinking (the words show Auto answering), and marks the turn past it while a card waits too. */
function onText(state: MascotState): MascotState {
  const marked = markPastThinking(state)
  return pendingPose(marked).mascot === 'thinking' ? settle(marked, rest(marked)) : marked
}

function reduceStream(state: MascotState, event: StreamEvent): MascotState {
  switch (event.type) {
    case 'text':
      return onText(state)
    case 'reasoning':
      return propose(state, THINKING)
    case 'tool-start':
      return onToolStart(state, event)
    case 'tool-end':
      return onToolEnd(state, event)
    case 'progress':
    case 'workflow-update':
    case 'agent-assigned':
      return propose(state, WORKING)
    case 'memory-injected':
      return propose(state, READING)
    case 'receipts':
      return onReceipts(state, event)
    case 'finish':
      return onFinish(state)
    case 'error':
      return propose(endTurn(state), OOPS)
    case 'limit_reached':
    case 'tool-permission-denied':
    case 'attachment_unavailable':
      return propose(state, OOPS)
    default:
      return state
  }
}

function onClipEnded(state: MascotState, ended: MascotName): MascotState {
  if (!playingOneShot(state) || ended !== state.mascot) return state
  return show(state, state.queued ?? rest(state))
}

function onSend(state: MascotState): MascotState {
  return settle({ ...state, ...TURN_RESET, turnActive: true }, THINKING)
}

/** The owner stopped the turn: its done receipt no longer celebrates at finish. */
function onStop(state: MascotState): MascotState {
  return state.celebrateOnFinish ? { ...state, celebrateOnFinish: false } : state
}

/** A new Deliverable: celebrate now. Made mid-turn, it is the turn's celebration (the shelf's, outside a turn, marks nothing). */
function onDeliverableAdded(state: MascotState): MascotState {
  const marked = state.turnActive && !state.celebratedThisTurn ? { ...state, celebratedThisTurn: true } : state
  return propose(marked, CELEBRATE)
}

/** hello on the first open. A pose above idle (an awaited reply's thinking, a running tool) waits behind it and plays when it ends. */
function onAppOpen(state: MascotState): MascotState {
  const pending = pendingPose(state)
  const greeted = propose(state, HELLO)
  return greeted !== state && pending.rank > RANK.idle ? enqueue(greeted, pending) : greeted
}

/** Only these get through while the backend is unreachable. */
const WHILE_DOWN: ReadonlySet<MascotEvent['type']> = new Set<MascotEvent['type']>([
  'backend-up',
  'not-found',
  'clip-ended',
])

/** The page events that only look at what Auto shows now or next. */
function reducePresence(state: MascotState, event: ClientEvent): MascotState {
  const pending = pendingPose(state)
  switch (event.type) {
    case 'input-blur':
      return pending.mascot === 'listening' && pending.rank === RANK.typing ? settle(state, rest(state)) : state
    case 'idle-timeout':
      return !state.turnActive && pending.rank <= RANK.typing ? settle(state, SLEEPING) : state
    case 'wake':
      return pending.mascot === 'sleeping' ? settle(state, rest(state)) : state
    case 'loading-start':
      return pending.rank <= RANK.idle ? settle(state, LOADING) : state
    case 'loading-end':
      return pending.mascot === 'loading' ? settle(state, rest(state)) : state
    case 'request-ok':
      return pending.mascot === 'oops' && !state.turnActive ? settle(state, rest(state)) : state
    default:
      return state
  }
}

function reduceClient(state: MascotState, event: ClientEvent): MascotState {
  switch (event.type) {
    case 'app-open':
      return onAppOpen(state)
    case 'input-focus':
      return propose(state, TYPING)
    case 'send':
      return onSend(state)
    case 'stop':
      return onStop(state)
    case 'backend-down':
      return settle(endTurn({ ...state, backendDown: true }), SLEEPING)
    case 'backend-up':
      return state.backendDown ? settle({ ...state, backendDown: false }, rest(state)) : state
    case 'not-found':
      return settle(state, LOST)
    case 'deliverable-added':
      return onDeliverableAdded(state)
    case 'request-failed':
      return propose(state, OOPS)
    case 'clip-ended':
      return onClipEnded(state, event.state)
    default:
      return reducePresence(state, event)
  }
}

const CLIENT_TYPES: ReadonlySet<MascotEvent['type']> = new Set<MascotEvent['type']>([
  'app-open',
  'input-focus',
  'input-blur',
  'send',
  'stop',
  'idle-timeout',
  'wake',
  'backend-down',
  'backend-up',
  'not-found',
  'loading-start',
  'loading-end',
  'deliverable-added',
  'request-failed',
  'request-ok',
  'clip-ended',
])

const isClientEvent = (event: MascotEvent): event is ClientEvent => CLIENT_TYPES.has(event.type)

const isWaiting = (p: Pose): boolean => p.mascot === WAITING.mascot && p.rank === WAITING.rank

/**
 * Where Auto goes when the last card settles while he listens for it: what the
 * turn shows without the card. A tool still running (its start was outranked by
 * the card); thinking while the turn runs and nothing moved him past it; idle
 * once its words started, a tool ran or the turn is over.
 */
function afterCard(cleared: MascotState): Pose {
  if (cleared.openTools.length > 0) return afterTools(cleared)
  return cleared.turnActive && !cleared.pastThinking ? THINKING : rest(cleared)
}

/**
 * An approval card waits (Auto listens, above the turn's tools and thinking) or
 * every card is settled (listening for it ends). The flag is kept while the
 * backend is down, so Auto wakes to the right rest; only the pose waits.
 */
function reduceApproval(state: MascotState, event: ApprovalEvent): MascotState {
  if (event.type === 'approval-waiting') {
    const flagged = state.approvalWaiting ? state : { ...state, approvalWaiting: true }
    return flagged.backendDown ? flagged : propose(flagged, WAITING)
  }
  if (!state.approvalWaiting) return state
  const cleared = { ...state, approvalWaiting: false }
  return !cleared.backendDown && isWaiting(pendingPose(cleared)) ? settle(cleared, afterCard(cleared)) : cleared
}

const isApprovalEvent = (event: MascotEvent): event is ApprovalEvent =>
  event.type === 'approval-waiting' || event.type === 'approval-settled'

export function mascotReducer(state: MascotState, event: MascotEvent): MascotState {
  if (isApprovalEvent(event)) return reduceApproval(state, event)
  if (state.backendDown && !WHILE_DOWN.has(event.type)) return state
  return isClientEvent(event) ? reduceClient(state, event) : reduceStream(state, event)
}
