/**
 * Diff approval model
 * PRD-170 S4
 *
 * Maps SDK permission prompts to UI approval cards and holds the approve/deny
 * decision flow. Pure + immutable (no React, no I/O) so it is exhaustively
 * unit-testable; the DiffCard component renders a card and calls `decide`, and
 * the panel forwards the resulting decision to the session (approve → applies,
 * deny → reverts + informs).
 *
 * Auto-accept: SESSION-SCOPED, default OFF, and visibly indicated by the panel.
 * When ON, an incoming edit card is pre-resolved as approved WITHOUT a manual
 * click — this is the "per-turn auto-accept" affordance. It never applies to a
 * bash/exec permission (those always require an explicit decision), so a
 * destructive shell command can never be silently auto-run.
 */

export type ApprovalDecision = 'approve' | 'deny'
export type ApprovalStatus = 'pending' | 'approved' | 'denied'

export interface DiffModel {
  /** Path of the file the edit targets. */
  path: string
  /** Content before the proposed edit (empty string for a new file). */
  oldContent: string
  /** Content after the proposed edit. */
  newContent: string
  /** Monaco language id, derived from the path. */
  language: string
}

export interface ApprovalCard {
  requestId: string
  toolName: string
  /** File edits carry a diff; a bare permission (e.g. Bash) may not. */
  diff: DiffModel | null
  status: ApprovalStatus
  /** True when the card was resolved by the auto-accept toggle, not a click. */
  autoAccepted: boolean
}

export interface DiffApprovalState {
  cards: ApprovalCard[]
  /** Session-scoped auto-accept for file edits. Default OFF. */
  autoAcceptEdits: boolean
}

export const initialDiffApprovalState: DiffApprovalState = {
  cards: [],
  autoAcceptEdits: false,
}

// Monaco language ids by extension — a small mirror of the worker's map, enough
// for the diff viewer to syntax-highlight common files.
const _LANG_BY_EXT: Record<string, string> = {
  py: 'python',
  js: 'javascript',
  jsx: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  json: 'json',
  yaml: 'yaml',
  yml: 'yaml',
  md: 'markdown',
  html: 'html',
  css: 'css',
  scss: 'scss',
  sql: 'sql',
  sh: 'shell',
  go: 'go',
  rs: 'rust',
  java: 'java',
  rb: 'ruby',
  toml: 'toml',
}

export function languageForPath(path: string): string {
  const dot = path.lastIndexOf('.')
  if (dot < 0 || dot === path.length - 1) return 'plaintext'
  const ext = path.slice(dot + 1).toLowerCase()
  return _LANG_BY_EXT[ext] ?? 'plaintext'
}

export interface IncomingEdit {
  requestId: string
  toolName: string
  path: string
  oldContent: string
  newContent: string
}

/**
 * Add an incoming file-edit permission as a card. When session auto-accept is
 * ON, the card lands already `approved` (autoAccepted=true) so the UI shows what
 * was applied while the flow stays non-blocking.
 */
export function addEditCard(
  state: DiffApprovalState,
  edit: IncomingEdit
): DiffApprovalState {
  const auto = state.autoAcceptEdits
  const card: ApprovalCard = {
    requestId: edit.requestId,
    toolName: edit.toolName,
    diff: {
      path: edit.path,
      oldContent: edit.oldContent,
      newContent: edit.newContent,
      language: languageForPath(edit.path),
    },
    status: auto ? 'approved' : 'pending',
    autoAccepted: auto,
  }
  return { ...state, cards: [...state.cards, card] }
}

/**
 * Add a non-edit permission request (e.g. Bash). These ALWAYS require an
 * explicit decision — auto-accept never applies to them.
 */
export function addPermissionCard(
  state: DiffApprovalState,
  requestId: string,
  toolName: string
): DiffApprovalState {
  const card: ApprovalCard = {
    requestId,
    toolName,
    diff: null,
    status: 'pending',
    autoAccepted: false,
  }
  return { ...state, cards: [...state.cards, card] }
}

/** Resolve a card by an explicit user decision. */
export function decide(
  state: DiffApprovalState,
  requestId: string,
  decision: ApprovalDecision
): DiffApprovalState {
  return {
    ...state,
    cards: state.cards.map((c) =>
      c.requestId === requestId && c.status === 'pending'
        ? { ...c, status: decision === 'approve' ? 'approved' : 'denied' }
        : c
    ),
  }
}

/** Toggle session-scoped auto-accept for edits. Does not retroactively resolve. */
export function setAutoAccept(
  state: DiffApprovalState,
  on: boolean
): DiffApprovalState {
  return { ...state, autoAcceptEdits: on }
}

export function pendingCards(state: DiffApprovalState): ApprovalCard[] {
  return state.cards.filter((c) => c.status === 'pending')
}
