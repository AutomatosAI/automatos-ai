'use client'

import Link from 'next/link'

import { type DeliverableRef, deliverableHref, isDeliverableId, KIND_WORDS } from '@/lib/chat/deliverables'

import styles from './turn.module.css'

/**
 * PRD-260 §4, K8 — the Deliverable Auto just made, inline in his reply
 * (mock 02), ported from Automatos Play's components/chat/DeliverableCard.tsx:
 * a printed-sheet tile, the title, the type, and Open, which goes to the
 * core's /deliverables?deliverable=<id>. The id is checked once more here
 * before it goes into the link: a card with a bad id has no Open.
 */

export const JUST_NOW = 'just now'
export const OPEN = 'Open'

export function DeliverableCard({ deliverable }: { readonly deliverable: DeliverableRef }) {
  const href = isDeliverableId(deliverable.id) ? deliverableHref(deliverable.id) : null
  return (
    <article className={styles.deliverable} aria-label={`Deliverable: ${deliverable.title}`}>
      <span className={styles.deliverableTile} data-kind={deliverable.kind} aria-hidden="true" />
      <div className={styles.deliverableText}>
        <h3 className={styles.deliverableTitle}>{deliverable.title}</h3>
        <p className={styles.deliverableMeta}>
          {KIND_WORDS[deliverable.kind]} · {JUST_NOW}
        </p>
      </div>
      {href && (
        <Link href={href} className={styles.openButton} aria-label={`${OPEN} ${deliverable.title}`}>
          {OPEN}
        </Link>
      )}
    </article>
  )
}
