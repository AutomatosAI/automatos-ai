'use client'

/**
 * F298 (night 8) — /deliverables?deliverable=<id> opens that Deliverable.
 *
 * The PDF link an agent put on card #0394.2 was the storage share link, a
 * 600-character signed URL the agent copied into its answer and got one
 * character of the signature wrong: 403 SignatureDoesNotMatch, while the same
 * file opened from Deliverables. generate_document now gives the agent a short
 * link to this page for the owner instead. The preview fetches the file through
 * apiClient, so it opens in both editions: anonymously in the local edition,
 * with the signed-in owner's session in the hosted one. Closing it drops the
 * parameter and leaves the page as it was.
 */

import { useCallback } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'

export const DELIVERABLE_PARAM = 'deliverable'

export function DeliverableDeepLink() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const deliverableId = searchParams?.get(DELIVERABLE_PARAM) || null

  const handleOpenChange = useCallback(
    (open: boolean) => {
      if (open) return
      const rest = new URLSearchParams(searchParams?.toString() ?? '')
      rest.delete(DELIVERABLE_PARAM)
      const query = rest.toString()
      router.replace(query ? `/deliverables?${query}` : '/deliverables')
    },
    [router, searchParams],
  )

  return (
    <DeliverablePreview
      deliverableId={deliverableId}
      open={deliverableId !== null}
      onOpenChange={handleOpenChange}
    />
  )
}

export default DeliverableDeepLink
