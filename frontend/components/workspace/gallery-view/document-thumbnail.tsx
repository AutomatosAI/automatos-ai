'use client'

/**
 * DocumentThumbnail (F353, issue #947)
 * ====================================
 *
 * A document, sheet or report card shows its first page, the way an image card
 * shows its image. The picture is the Deliverable's `thumbnail_url`
 * (GET /api/deliverables/{id}/thumbnail), fetched with the caller's auth through
 * `useAuthenticatedBlobUrl`, as image cards fetch theirs. While it loads, and
 * when there is none (not drawn yet, or the render failed), the card keeps its
 * type artwork.
 */

import { DeliverableArtwork } from '@/components/deliverables/deliverable-artwork'
import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'

interface DocumentThumbnailProps {
  type: string
  thumbnailUrl?: string | null
  title: string
}

export function DocumentThumbnail({ type, thumbnailUrl, title }: DocumentThumbnailProps) {
  const { src } = useAuthenticatedBlobUrl(thumbnailUrl ?? undefined)

  if (!src) return <DeliverableArtwork type={type} className="absolute inset-0" />

  return (
    <img
      src={src}
      alt={title}
      className="h-full w-full bg-white object-cover object-top"
    />
  )
}
