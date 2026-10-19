/**
 * PRD-261 A-10a — SDK area `notices`: a theme tells the user something through the host's own
 * toaster (sonner, mounted once by components/providers.tsx). sonner keeps its toasts in a
 * module-level store that only the host's copy of it renders, so a theme never bundles or
 * imports sonner (the vite-preset makes it external and review refuses a package that ships it);
 * it calls `notify` and the host's Toaster draws the notice in the host's style.
 *
 * Notices are text only: a title, an optional description and an optional action button. Every
 * field is checked; a malformed notice throws a TypeError instead of showing something half-made.
 */
import { toast, type ExternalToast } from 'sonner'

export type NoticeKind = 'message' | 'success' | 'info' | 'warning' | 'error'
export type NoticeId = string | number

export interface NoticeAction {
  readonly label: string
  readonly onClick: () => void
}

export interface Notice {
  /** Default 'message'. */
  readonly kind?: NoticeKind
  readonly title: string
  readonly description?: string
  /** How long it stays, in milliseconds; the toaster's default when absent. */
  readonly durationMs?: number
  readonly action?: NoticeAction
}

type Sender = (title: string, options?: ExternalToast) => NoticeId

const SENDERS: Readonly<Record<NoticeKind, Sender>> = {
  message: (title, options) => toast.message(title, options),
  success: (title, options) => toast.success(title, options),
  info: (title, options) => toast.info(title, options),
  warning: (title, options) => toast.warning(title, options),
  error: (title, options) => toast.error(title, options),
}

const DEFAULT_KIND: NoticeKind = 'message'

function refuse(problem: string): never {
  throw new TypeError(`Automatos theme SDK: notify() ${problem}.`)
}

function isText(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function checkedKind(kind: unknown): NoticeKind {
  if (kind === undefined) return DEFAULT_KIND
  if (typeof kind !== 'string' || !Object.prototype.hasOwnProperty.call(SENDERS, kind)) {
    return refuse(`has no notice kind "${String(kind)}"`)
  }
  return kind as NoticeKind
}

function checkedAction(action: unknown): ExternalToast['action'] {
  if (action === undefined) return undefined
  const candidate = action as Partial<NoticeAction> | null
  if (!candidate || !isText(candidate.label) || typeof candidate.onClick !== 'function') {
    return refuse('needs an action with a label and an onClick function')
  }
  const onClick = candidate.onClick
  return { label: candidate.label, onClick: () => onClick() }
}

function checkedOptions(notice: Notice): ExternalToast {
  const { description, durationMs } = notice
  if (description !== undefined && typeof description !== 'string') refuse('needs a text description')
  if (durationMs !== undefined && !(typeof durationMs === 'number' && Number.isFinite(durationMs) && durationMs > 0)) {
    refuse('needs a positive durationMs')
  }
  const action = checkedAction(notice.action)
  return {
    ...(description !== undefined ? { description } : {}),
    ...(durationMs !== undefined ? { duration: durationMs } : {}),
    ...(action !== undefined ? { action } : {}),
  }
}

/** Shows `notice` in the host's toaster; returns its id, for dismissNotice. */
export function notify(notice: Notice): NoticeId {
  if (typeof notice !== 'object' || notice === null) return refuse('needs a notice object')
  if (!isText(notice.title)) return refuse('needs a title')
  const send = SENDERS[checkedKind(notice.kind)]
  return send(notice.title, checkedOptions(notice))
}

/** Dismisses one notice by its id, or every notice when no id is given. */
export function dismissNotice(id?: NoticeId): void {
  toast.dismiss(id)
}
