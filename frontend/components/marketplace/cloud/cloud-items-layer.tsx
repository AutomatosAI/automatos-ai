'use client'

/**
 * PRD-270 W2 US-270-14d, US-270-14e (§5.6, K17, K19) — the Marketplace's Automatos Cloud items layer, mounted
 * once in MarketplaceHomepage, local edition only (nothing on the hosted edition).
 *
 * On mount it registers the install gate's opener (lib/marketplace/cloud-install-gate.ts) and unregisters it
 * on unmount. When an Add or Enable of a Cloud item reaches apiClient, the gate asks the layer, which shows
 * that item's disclosure (CloudDisclosureDialog): "Add to my workspace" lets that one request go and closes
 * the dialog (the calling card shows its own progress and result); Close refuses it, and nothing is sent.
 * One review at a time: an install arriving while one is open is refused ("Finish the review that is open
 * first …"). A review still open when the layer unmounts is refused too, so no request waits forever.
 *
 * The gate reads which items are Cloud items through the layer: the react-query list the badges share
 * (hooks/use-cloud-items.ts `fetchCloudItems`), read afresh for each install.
 *
 * The update notices (update-notices.tsx, US-270-14e) render here, above the tabs. A notice's "Review" opens
 * the same dialog with no request waiting: its Add is the notice's (a new copy through the normal install,
 * which the gate lets through once), and the dialog closes once that copy is added; an updated plugin's
 * review has no Add. A Review while another review is open says so ("Finish the review …") and opens nothing.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { fetchCloudItems } from '@/hooks/use-cloud-items'
import { isLocal } from '@/lib/auth-edition'
import type { CloudInstallCall } from '@/lib/marketplace/cloud-install-call'
import {
  CLOUD_INSTALL_BUSY, CloudInstallRefused, registerCloudInstallOpener,
} from '@/lib/marketplace/cloud-install-gate'

import { CloudDisclosureDialog } from './disclosure-dialog'
import { CloudUpdateNotices } from './update-notices'
import type { ReviewUpdate } from './update-notices'

/** The review on screen: the item, what its Add does (none: a review only) and what its Close does. */
interface ShownReview extends CloudInstallCall {
  add?: () => void | Promise<unknown>
  close: () => void
}

/** The one review on screen; `show` refuses a second (false), `end` takes off only the review it names. */
function useReviewSlot() {
  const [review, setReview] = useState<ShownReview | null>(null)
  const shown = useRef<ShownReview | null>(null)
  const show = useCallback((next: ShownReview): boolean => {
    if (shown.current) return false
    shown.current = next
    setReview(next)
    return true
  }, [])
  const end = useCallback((done: ShownReview): boolean => {
    if (shown.current !== done) return false
    shown.current = null
    setReview(null)
    return true
  }, [])
  const current = useCallback(() => shown.current, [])
  return { review, show, end, current }
}

function LocalCloudItemsLayer() {
  const queryClient = useQueryClient()
  const { review, show, end, current } = useReviewSlot()

  useEffect(() => {
    const open = (call: CloudInstallCall) => new Promise<boolean>((resolve, reject) => {
      const next: ShownReview = {
        ...call,
        add: () => { if (end(next)) resolve(true) },
        close: () => { if (end(next)) resolve(false) },
      }
      if (!show(next)) reject(new CloudInstallRefused(CLOUD_INSTALL_BUSY))
    })
    const unregister = registerCloudInstallOpener({ open, items: () => fetchCloudItems(queryClient) })
    return () => {
      unregister()
      current()?.close()
    }
  }, [queryClient, show, end, current])

  const reviewUpdate = useCallback<ReviewUpdate>((call, add) => {
    const next: ShownReview = {
      ...call,
      add: add && (async () => {
        if (await add()) end(next)
      }),
      close: () => { end(next) },
    }
    if (!show(next)) toast.error(CLOUD_INSTALL_BUSY)
  }, [show, end])

  return (
    <>
      <CloudUpdateNotices onReview={reviewUpdate} />
      {review && (
        <CloudDisclosureDialog kind={review.kind} id={review.key} open onAdd={review.add} onClose={review.close} />
      )}
    </>
  )
}

export function CloudItemsLayer() {
  return isLocal ? <LocalCloudItemsLayer /> : null
}
