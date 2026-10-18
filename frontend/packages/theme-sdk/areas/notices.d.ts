/**
 * PRD-261 A-10b — SDK area `notices` (frontend/lib/theme-sdk/notices.ts): a theme tells the user something
 * through the host's own toaster. A theme never bundles or imports a toaster of its own.
 */

export type NoticeKind = 'message' | 'success' | 'info' | 'warning' | 'error'
export type NoticeId = string | number

export interface NoticeAction {
  readonly label: string
  readonly onClick: () => void
}

/** Text only. Every field is checked; a malformed notice throws a TypeError instead of showing half of it. */
export interface Notice {
  /** Default 'message'. */
  readonly kind?: NoticeKind
  readonly title: string
  readonly description?: string
  /** How long it stays, in milliseconds; the toaster's default when absent. */
  readonly durationMs?: number
  readonly action?: NoticeAction
}

/** Shows `notice` in the host's toaster; returns its id, for dismissNotice. */
export declare function notify(notice: Notice): NoticeId

/** Dismisses one notice by its id, or every notice when no id is given. */
export declare function dismissNotice(id?: NoticeId): void
