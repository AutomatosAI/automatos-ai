'use client'

import Link from 'next/link'
import { useState } from 'react'

import { deliverableHref, isDeliverableId, UNTITLED } from '@/components/play/chat/deliverables'
import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { kindOf, shelfDate, typeLabel } from '@/lib/play/shelf'

import styles from './shelf-card.module.css'

/**
 * PRD-260 §4, K8 — one Deliverable on Play's shelf (Ember mock 04), ported
 * from Automatos Play's components/deliverables/{ShelfCard,TypeBadge}.tsx:
 * the picture over a printed sheet (all it shows while the picture loads,
 * and when there is none or it fails), the title, the type in words and the
 * date. The whole card opens the core's /deliverables?deliverable=<id>.
 *
 * The picture comes through the core's own authenticated path, as the
 * Deliverables gallery's cards load theirs (DocumentThumbnail and
 * DeliverableCard): useAuthenticatedBlobUrl over the image's preview_url, or
 * else the thumbnail_url GET /api/deliverables/{id}/thumbnail serves.
 */

export const NEWEST = 'Newest'

/** A Deliverable's type in words ("Report", "Slides"), set in its card's label. */
export function TypeBadge({ kind }: { readonly kind: string }) {
  return (
    <span className={styles.typeBadge} data-kind={kind}>
      {typeLabel(kind)}
    </span>
  )
}

/** The gallery's choice: an image shows itself, anything else its first-page thumbnail when it has one. */
function pictureUrl(deliverable: Deliverable): string | undefined {
  if (deliverable.artifact_type === 'image' && deliverable.preview_url) return deliverable.preview_url
  return deliverable.thumbnail_url ?? undefined
}

function ShelfPicture({ deliverable }: { readonly deliverable: Deliverable }) {
  const { src } = useAuthenticatedBlobUrl(pictureUrl(deliverable))
  const [broken, setBroken] = useState<string | null>(null)
  const onError = () => {
    console.warn(`[shelf] the picture for Deliverable ${deliverable.id} didn't load, showing a sheet instead`)
    setBroken(src)
  }
  if (!src || broken === src) return null
  // A blob: URL from the authenticated fetch, which next/image can't optimise.
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt="" className={styles.thumbImage} onError={onError} />
}

export interface ShelfCardProps {
  readonly deliverable: Deliverable
  readonly now: Date
  /** The newest on the shelf gets the spotlight. */
  readonly newest: boolean
}

export function ShelfCard({ deliverable, now, newest }: ShelfCardProps) {
  const kind = kindOf(deliverable)
  const title = deliverable.title?.trim() || UNTITLED
  const date = shelfDate(deliverable.created_at, now)
  const body = (
    <>
      <div className={styles.thumb} data-kind={kind} data-testid="shelf-thumb">
        <ShelfPicture deliverable={deliverable} />
        {newest && <span className={styles.newest}>{NEWEST}</span>}
      </div>
      <div className={styles.cardText}>
        <h2 className={styles.cardTitle}>{title}</h2>
        <p className={styles.cardMeta}>
          <TypeBadge kind={kind} />
          {date && <time dateTime={deliverable.created_at}>{date}</time>}
        </p>
      </div>
    </>
  )
  return (
    <li className={styles.item} data-newest={newest ? 'true' : undefined}>
      {isDeliverableId(deliverable.id) ? (
        <Link href={deliverableHref(deliverable.id)} className={styles.card}>
          {body}
        </Link>
      ) : (
        <div className={styles.card}>{body}</div>
      )}
    </li>
  )
}
