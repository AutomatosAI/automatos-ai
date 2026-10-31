'use client'

/**
 * PRD-270 W2 US-270-14d (§5.6, K19) — the Marketplace's Automatos Cloud items layer, mounted once in
 * MarketplaceHomepage, local edition only (nothing on the hosted edition).
 *
 * On mount it registers the install gate's opener (lib/marketplace/cloud-install-gate.ts) and unregisters it
 * on unmount. When an Add or Enable of a Cloud item reaches apiClient, the gate asks the layer, which shows
 * that item's disclosure (CloudDisclosureDialog): "Add to my workspace" lets that one request go and closes
 * the dialog (the calling card shows its own progress and result); Close refuses it, and nothing is sent.
 * One review at a time: an install arriving while one is open is refused ("Finish the review that is open
 * first …"). A review still open when the layer unmounts is refused too, so no request waits forever.
 *
 * The gate reads which items are Cloud items through the layer: the react-query list the badges share
 * (hooks/use-cloud-items.ts `fetchCloudItems`), read afresh for each install. The update notices
 * (US-270-14e) render here, above the tabs.
 */
import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { fetchCloudItems } from '@/hooks/use-cloud-items'
import { isLocal } from '@/lib/auth-edition'
import type { CloudInstallCall } from '@/lib/marketplace/cloud-install-call'
import {
  CLOUD_INSTALL_BUSY, CloudInstallRefused, registerCloudInstallOpener,
} from '@/lib/marketplace/cloud-install-gate'

import { CloudDisclosureDialog } from './disclosure-dialog'

/** The review on screen: the item an install names, and the answer its request waits for. */
interface PendingReview extends CloudInstallCall {
  decide: (added: boolean) => void
}

function LocalCloudItemsLayer() {
  const queryClient = useQueryClient()
  const [review, setReview] = useState<PendingReview | null>(null)
  const pending = useRef<PendingReview | null>(null)

  useEffect(() => {
    const open = (call: CloudInstallCall) => new Promise<boolean>((resolve, reject) => {
      if (pending.current) {
        reject(new CloudInstallRefused(CLOUD_INSTALL_BUSY))
        return
      }
      const next: PendingReview = {
        ...call,
        decide: (added) => {
          if (pending.current !== next) return
          pending.current = null
          setReview(null)
          resolve(added)
        },
      }
      pending.current = next
      setReview(next)
    })
    const unregister = registerCloudInstallOpener({ open, items: () => fetchCloudItems(queryClient) })
    return () => {
      unregister()
      pending.current?.decide(false)
    }
  }, [queryClient])

  return (
    <>
      {/* The update notices (US-270-14e) render here, above the tabs. */}
      {review && (
        <CloudDisclosureDialog
          kind={review.kind}
          id={review.key}
          open
          onAdd={() => review.decide(true)}
          onClose={() => review.decide(false)}
        />
      )}
    </>
  )
}

export function CloudItemsLayer() {
  return isLocal ? <LocalCloudItemsLayer /> : null
}
