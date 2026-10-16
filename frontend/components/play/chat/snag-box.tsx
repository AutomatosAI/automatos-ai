'use client'

import styles from './turn.module.css'

/**
 * PRD-260 §4 — the "I hit a snag" box (mock 08), ported from Automatos Play's
 * components/chat/SnagBox.tsx: what went wrong, in words, and Try again, which
 * asks the core chat to answer the last message again (the hook's reload()).
 * Only the newest failed turn gets the button.
 */

export const SNAG_TITLE = 'I hit a snag'
export const TRY_AGAIN = 'Try again'
/** A turn that failed before Auto's reply began (the request itself failed): the toast has the detail. */
export const TURN_FAILED = "Auto couldn't answer that just now."

export interface SnagBoxProps {
  readonly lines: readonly string[]
  /** Answers the last message again; none for an older reply. */
  readonly onRetry?: () => void
}

export function SnagBox({ lines, onRetry }: SnagBoxProps) {
  return (
    <section className={styles.snag} aria-label={SNAG_TITLE} data-tone="failed">
      <h3 className={styles.snagTitle}>{SNAG_TITLE}</h3>
      {lines.map((line, index) => (
        <p key={`${index}-${line}`} className={styles.snagLine}>
          <span className={styles.mark} aria-hidden="true">
            ✕
          </span>
          {line}
        </p>
      ))}
      {onRetry && (
        <div className={styles.snagActions}>
          <button type="button" className={styles.snagRetry} onClick={onRetry}>
            {TRY_AGAIN}
          </button>
        </div>
      )}
    </section>
  )
}
