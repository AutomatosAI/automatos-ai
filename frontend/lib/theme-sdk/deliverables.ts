/**
 * PRD-261 A-10a — SDK area `deliverables`: the deliverables list, one deliverable and its files (the feed
 * filters, the tab labels and the authenticated blob URL planned by A-10c, docs/theme-host/sdk-surface-map.md). Stubbed here; A-12a fills the bodies.
 * Until then every name throws "not in this SDK build" (internal/stub.ts), so a theme built
 * against a later SDK fails loudly. The names are the plan's (surface/deliverables.json): the
 * area's lane replaces bodies, never names, unless the A-10c surface map changes first.
 */
import { notInThisBuild } from './internal/stub'

const AREA = 'deliverables'

export const useDeliverables = notInThisBuild(AREA, 'useDeliverables')
export const useDeliverable = notInThisBuild(AREA, 'useDeliverable')
export const useFileDownload = notInThisBuild(AREA, 'useFileDownload')
export const deliverableDownloadUrl = notInThisBuild(AREA, 'deliverableDownloadUrl')
export const deliverableFileUrl = notInThisBuild(AREA, 'deliverableFileUrl')
export const deliverableFilename = notInThisBuild(AREA, 'deliverableFilename')
export const fileDownloadUrl = notInThisBuild(AREA, 'fileDownloadUrl')
export const visibleDeliverableTabs = notInThisBuild(AREA, 'visibleDeliverableTabs')
export const deliverableLabel = notInThisBuild(AREA, 'deliverableLabel')
export const useAuthenticatedBlobUrl = notInThisBuild(AREA, 'useAuthenticatedBlobUrl')
export const DELIVERABLE_FEED_FILTERS = notInThisBuild(AREA, 'DELIVERABLE_FEED_FILTERS')
export const DELIVERABLE_TAB_LABELS = notInThisBuild(AREA, 'DELIVERABLE_TAB_LABELS')
