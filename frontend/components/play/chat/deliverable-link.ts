import { DELIVERABLE_PARAM, DELIVERABLES_HREF, type HostDeliverable } from '@automatos/theme-sdk'

/**
 * PRD-260 §4, K8, PRD-261 A-14a — a Deliverable's link and type in words, for
 * the card inline in Auto's reply: the id is checked once more before it goes
 * into the link (the core's DELIVERABLE_ID), and the link is the core's own
 * page for one Deliverable, /deliverables?deliverable=<id>, built on the SDK's
 * routes. Copied into Play from the core's lib/chat/deliverables.ts
 * (docs/theme-host/sdk-surface-map.md); lib/__tests__/prd261-port-chat.test.ts
 * holds each to the core's. Pure.
 */

const DELIVERABLE_ID = /^[A-Za-z0-9_-]{1,128}$/

/** The kind in words, for the card's type badge. */
export const KIND_WORDS: Readonly<Record<HostDeliverable['kind'], string>> = {
  document: 'Document',
  image: 'Image',
  video: 'Video',
}

export function isDeliverableId(id: string): boolean {
  return DELIVERABLE_ID.test(id)
}

/** The core's page for one Deliverable (K8). */
export function deliverableHref(id: string): string {
  return `${DELIVERABLES_HREF}?${DELIVERABLE_PARAM}=${encodeURIComponent(id)}`
}
