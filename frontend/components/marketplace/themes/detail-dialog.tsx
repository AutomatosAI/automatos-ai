'use client'

/**
 * PRD-261 C-04b — a listing's detail page, as a dialog over the Themes tab.
 *
 * The listing is fetched when it opens (C-04a `GET /api/marketplace/themes/{listingId}`): its
 * name, version and publisher, a warning revocation's reason, "Add to my workspace" (or "Use this
 * theme" once it is here), previews per preset, what it changes, the versions map and its docs.
 * The body is keyed by listing, so a result of "Add to my workspace" never carries over to the
 * next listing opened. The detail and the Add control come from the listing source
 * (listing-source.tsx, PRD-270 K20): the member's route and AddToWorkspace by default, the public
 * catalogue and sign-in on the signed-out visitor's page. Opened from a `?item=&add=1` link
 * (add-link.tsx, US-270-09b), Add takes focus once the listing loads.
 */
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import type { ThemeListingDetail } from '@/lib/api-client'
import { themeErrorMessage } from '@/lib/theme-host/api'

import { DetailDocs, DetailFacts, DetailPreviews, DetailVersions } from './detail-sections'
import { ListingBadges } from './listing-badges'
import { LOADING_LISTING, listingName } from './listing-format'
import { useListingSource } from './listing-source'

interface DetailDialogProps {
  /** The listing shown; null closes the dialog. */
  readonly listingId: string | null
  readonly canManage: boolean
  /** Focus "Add to my workspace" once the listing loads. */
  readonly focusAdd?: boolean
  readonly onClose: () => void
}

interface DetailStateProps {
  readonly canManage: boolean
  readonly focusAdd: boolean
}

function DetailBody({ detail, canManage, focusAdd }: DetailStateProps & { detail: ThemeListingDetail }) {
  const { AddControl } = useListingSource()
  return (
    <div className="space-y-5">
      <DialogHeader>
        <DialogTitle className="flex flex-wrap items-center gap-2">
          {listingName(detail)}
          <span className="text-sm font-normal text-muted-foreground">{detail.version}</span>
        </DialogTitle>
        <DialogDescription>{detail.description ?? detail.themeId}</DialogDescription>
      </DialogHeader>
      <ListingBadges listing={detail} />
      {detail.revoked && <p className="text-sm text-warning">{detail.revoked.reason}</p>}
      <AddControl listing={detail} canManage={canManage} autoFocus={focusAdd} />
      <DetailPreviews detail={detail} />
      <DetailFacts detail={detail} />
      <DetailVersions detail={detail} />
      <DetailDocs detail={detail} />
    </div>
  )
}

function DetailState({ listingId, canManage, focusAdd }: DetailStateProps & { listingId: string }) {
  const { useDetail } = useListingSource()
  const detail = useDetail(listingId)
  if (detail.isLoading) {
    return (
      <DialogHeader>
        <DialogTitle>{LOADING_LISTING}</DialogTitle>
      </DialogHeader>
    )
  }
  if (detail.isError || !detail.data) {
    return (
      <DialogHeader>
        <DialogTitle>Theme</DialogTitle>
        <DialogDescription className="text-destructive">{themeErrorMessage(detail.error)}</DialogDescription>
      </DialogHeader>
    )
  }
  return <DetailBody detail={detail.data} canManage={canManage} focusAdd={focusAdd} />
}

export function DetailDialog({ listingId, canManage, focusAdd = false, onClose }: DetailDialogProps) {
  return (
    <Dialog open={listingId !== null} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto" data-testid="theme-listing-detail">
        {listingId !== null && (
          <DetailState key={listingId} listingId={listingId} canManage={canManage} focusAdd={focusAdd} />
        )}
      </DialogContent>
    </Dialog>
  )
}
