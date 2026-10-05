'use client'

/**
 * The Open control on a Deliverable row in a mission's or a ticket's Deliverables list.
 *
 * A row's preview_url / content_url is usually an API route behind auth
 * (/api/documents/generated/<file>, /api/workspaces/<id>/files/…). A plain <a href> to
 * that relative path reaches the frontend's own origin without the Authorization header,
 * so it failed in both editions (F358). Open now shows the Deliverable in the same
 * preview the Deliverables page uses (DeliverablePreview): it loads the file through the
 * API client with auth, previews it, and downloads it. A link to another site stays a
 * plain link in a new tab.
 */
import { useState } from 'react'
import { ExternalLink } from 'lucide-react'

import { DeliverablePreview } from '@/components/workspace/gallery-view/deliverable-preview'
import { apiClient } from '@/lib/api-client'

const ABSOLUTE_HTTP = /^https?:\/\//i
const OPEN_CLASS = 'shrink-0 text-muted-foreground hover:text-foreground transition-colors'

export interface OpenableDeliverable {
  id: string
  title: string
  preview_url?: string | null
  content_url?: string | null
}

function originOf(url: string): string | null {
  try {
    return new URL(url).origin
  } catch {
    return null
  }
}

/** True for an http(s) link to a site that is neither this app nor its API. */
export function isExternalLink(href: string): boolean {
  if (!ABSOLUTE_HTTP.test(href)) return false
  const origin = originOf(href)
  if (!origin) return false
  const apiBase = apiClient.getBaseUrl()
  const apiOrigin = ABSOLUTE_HTTP.test(apiBase) ? originOf(apiBase) : null
  return origin !== window.location.origin && origin !== apiOrigin
}

export function DeliverableOpenButton({ deliverable }: { deliverable: OpenableDeliverable }) {
  const [open, setOpen] = useState(false)
  const href = deliverable.preview_url ?? deliverable.content_url ?? null
  if (!href) return null
  const label = `Open ${deliverable.title}`

  if (isExternalLink(href)) {
    return (
      <a href={href} target="_blank" rel="noreferrer" aria-label={label} className={OPEN_CLASS}>
        <ExternalLink className="w-3.5 h-3.5" />
      </a>
    )
  }
  return (
    <>
      <button type="button" aria-label={label} onClick={() => setOpen(true)} className={OPEN_CLASS}>
        <ExternalLink className="w-3.5 h-3.5" />
      </button>
      {open && <DeliverablePreview deliverableId={deliverable.id} open={open} onOpenChange={setOpen} />}
    </>
  )
}
