/**
 * DeliverablePreviewActions (PRD-129; F354, 5 Oct)
 * =================================================
 *
 * The action row of a Deliverable's panel: Download, Open in Explorer, Delete, and,
 * for a document, Add to Knowledge (F354); for a PDF or a web page, Full screen. Its own component so the panel
 * (deliverable-preview.tsx) stays within the 150-line component limit.
 */

'use client'

import { Download, ExternalLink, Loader2, Maximize2, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Deliverable } from '@/hooks/use-deliverables-api'
import { canAddToKnowledge } from '@/hooks/use-deliverable-knowledge'
import { AddToKnowledgeButton } from './add-to-knowledge-button'

export interface DeliverablePreviewActionsProps {
  deliverable: Deliverable
  downloadUrl: string | null
  downloading: boolean
  deleting: boolean
  onDownload: () => void
  onOpenInExplorer: () => void
  onDelete: () => void
  /** Shown for a preview that fills the panel (a PDF, a web page): opens it full screen. */
  onFullscreen?: () => void
}

export function DeliverablePreviewActions({
  deliverable,
  downloadUrl,
  downloading,
  deleting,
  onDownload,
  onOpenInExplorer,
  onDelete,
  onFullscreen,
}: DeliverablePreviewActionsProps) {
  return (
    <div className="flex flex-wrap gap-2 pt-1">
      {downloadUrl && (
        <Button variant="outline" size="sm" onClick={onDownload} disabled={downloading}>
          {downloading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Download className="mr-2 h-4 w-4" />
          )}
          Download
        </Button>
      )}
      <Button variant="outline" size="sm" onClick={onOpenInExplorer}>
        <ExternalLink className="mr-2 h-4 w-4" />
        Open in Explorer
      </Button>
      {onFullscreen && (
        <Button variant="outline" size="sm" onClick={onFullscreen}>
          <Maximize2 className="mr-2 h-4 w-4" />
          Full screen
        </Button>
      )}
      {canAddToKnowledge(deliverable) && <AddToKnowledgeButton deliverable={deliverable} />}
      <Button
        variant="outline"
        size="sm"
        onClick={onDelete}
        disabled={deleting}
        className="text-destructive hover:bg-destructive/10 hover:text-destructive"
      >
        {deleting ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        ) : (
          <Trash2 className="mr-2 h-4 w-4" />
        )}
        Delete
      </Button>
    </div>
  )
}
