'use client'

import { useLook } from './look-provider'
import styles from './manage.module.css'

/**
 * PRD-260 — what the last manage or import action on the Looks page did, in
 * words, ported from Automatos Play's components/looks/LookNotice.tsx.
 * Successes go to a polite live region that is always present (so a screen
 * reader hears the change); failures are an alert listing the reasons.
 * Everything here is React text: an imported name is never read as HTML.
 * A success that changed the looks (`stored`) adds LOOKS_NOT_KEPT while
 * storage isn't keeping them (useLook().persisted), including when the write
 * after the action is the one refused (M98).
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

export function LookNotice({ notice }: { readonly notice: Notice | null }) {
  const { persisted } = useLook()
  const notKept = notice?.kind === 'done' && notice.stored === true && !persisted
  return (
    <>
      <p className={styles.done} aria-live="polite">
        {notice?.kind === 'done' ? notice.text : ''}
        {notKept ? ` ${LOOKS_NOT_KEPT}` : ''}
      </p>
      {notice?.kind === 'failed' && (
        <div className={styles.errors} role="alert">
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
