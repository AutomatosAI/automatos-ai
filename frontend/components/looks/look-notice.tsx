'use client'

import styles from './manage.module.css'
import type { Told } from './use-notice-seq'
import { usePlayLook } from './use-play-look'

/**
 * PRD-260 — what the last manage or import action on the Looks page did, in
 * words, ported from Automatos Play's components/looks/LookNotice.tsx.
 * Successes go to a polite live region that is always present (so a screen
 * reader hears the change); failures are an alert listing the reasons.
 * Everything here is React text: an imported name is never read as HTML.
 * A success that changed the looks (`stored`) adds LOOKS_NOT_KEPT while
 * storage isn't keeping them (usePlayLook().persisted), including when the write
 * after the action is the one refused (M98).
 * Every notice carries a `seq` (./use-notice-seq.ts): the words sit in a
 * <span key={seq}> inside the polite region and the alert is keyed by it, so
 * the same outcome twice (the same import, the same failure) is a new node a
 * screen reader announces again, never an unchanged text it skips (M22).
 */

export const LOOKS_NOT_KEPT = "This browser isn't keeping looks, so it lasts until you close this tab."

export type Notice =
  | {
      readonly kind: 'done'
      readonly text: string
      /** It changed the looks, which storage may not keep. */
      readonly stored?: boolean
    }
  | { readonly kind: 'failed'; readonly title: string; readonly errors: readonly string[] }

/** A success that changed the looks: Save, Duplicate, Rename, Import. */
export function storedNotice(text: string): Notice {
  return { kind: 'done', text, stored: true }
}

export function LookNotice({ notice }: { readonly notice: Told<Notice> | null }) {
  const { persisted } = usePlayLook()
  const notKept = notice?.kind === 'done' && notice.stored === true && !persisted
  return (
    <>
      <p className={styles.done} aria-live="polite">
        {notice?.kind === 'done' && (
          <span key={notice.seq}>
            {notice.text}
            {notKept ? ` ${LOOKS_NOT_KEPT}` : ''}
          </span>
        )}
      </p>
      {notice?.kind === 'failed' && (
        <div key={notice.seq} className={styles.errors} role="alert">
          <p className={styles.errorTitle}>{notice.title}</p>
          <ul className={styles.errorList}>
            {notice.errors.map((error, index) => (
              <li key={`${index}-${error}`}>{error}</li>
            ))}
          </ul>
        </div>
      )}
    </>
  )
}

/** A failure: shown as an alert, and logged with the same reasons. */
export function failed(title: string, errors: readonly string[]): Notice {
  console.warn(`[looks] ${title}`, errors)
  return { kind: 'failed', title, errors }
}
