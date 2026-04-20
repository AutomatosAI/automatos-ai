/**
 * DeliverablePreview (PRD-129: Workspace Outputs Hub)
 * =====================================================
 *
 * Slide-over sheet that shows a single deliverable's content with Download,
 * Delete, and "Open in Explorer" actions. Fetches full content via useDeliverable
 * with include_content=true. All rendering is delegated to the shared
 * FilePreview component so Outputs, Chat, and Explorer stay aligned.
 */

'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { formatDistanceToNow } from 'date-fns'
import {
  Download,
  ExternalLink,
  FileWarning,
  Loader2,
  Trash2,
} from 'lucide-react'
import {
  FilePreview,
  inferPreviewType,
} from '@/components/widgets/FileWidget/FilePreview'

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import {
  useDeliverable,
  useDeleteDeliverable,
  type Deliverable,
} from '@/hooks/use-deliverables-api'
import { apiClient } from '@/lib/api-client'

interface DeliverablePreviewProps {
  deliverableId: string | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

// ============= HELPERS =============

/**
 * Derive a FilePreview-compatible type for a deliverable. Combines deliverable
 * metadata (artifact_type, file_type) with the filename-based inference used
 * by the shared FilePreview component, so the three preview surfaces
 * (Chat, Outputs, Explorer) stay aligned.
 */
function getPreviewTypeForDeliverable(d: Deliverable) {
  if (d.artifact_type === 'report') return 'markdown' as const
  return inferPreviewType(
    d.file_name || d.file_path || '',
    d.file_type || '',
  )
}

/**
 * Download a file through the API client (with auth headers) by fetching as
 * blob and triggering a browser download via object URL.
 */
async function downloadViaApi(url: string, filename: string): Promise<void> {
  const headers = await apiClient.getAuthHeaders()
  const fullUrl = `${apiClient.getBaseUrl()}${url}`
  const res = await fetch(fullUrl, { headers })
  if (!res.ok) throw new Error(`Download failed: ${res.status}`)
  const blob = await res.blob()
  const objectUrl = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = objectUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(objectUrl)
}

// ============= FALLBACK =============

function ContentUnavailable({
  message,
  downloadUrl,
  filename,
}: {
  message: string
  downloadUrl: string | null
  filename: string
}) {
  const [downloading, setDownloading] = useState(false)

  const handleDownload = async () => {
    if (!downloadUrl) return
    setDownloading(true)
    try {
      await downloadViaApi(downloadUrl, filename)
    } catch {
      window.open(`${apiClient.getBaseUrl()}${downloadUrl}`, '_blank')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border/50 bg-muted/10 p-10 text-center">
      <FileWarning className="h-10 w-10 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">{message}</p>
      {downloadUrl && (
        <Button variant="outline" size="sm" onClick={handleDownload} disabled={downloading}>
          {downloading ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Download className="mr-2 h-4 w-4" />
          )}
          Download
        </Button>
      )}
    </div>
  )
}

// ============= MAIN COMPONENT =============

export function DeliverablePreview({
  deliverableId,
  open,
  onOpenChange,
}: DeliverablePreviewProps) {
  const router = useRouter()
  const { data, isLoading, isError } = useDeliverable(
    open ? deliverableId : null,
    true,
  )
  const deleteMutation = useDeleteDeliverable()
  const [downloading, setDownloading] = useState(false)

  const deliverable = data?.deliverable ?? null

  // Close on Escape — Radix Sheet handles this by default, but we guarantee it.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onOpenChange(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onOpenChange])

  const downloadUrl = deliverable?.content_url || deliverable?.preview_url || null
  const filename =
    deliverable?.file_name ||
    deliverable?.file_path?.split('/').pop() ||
    'download'

  const handleDownload = useCallback(async () => {
    if (!downloadUrl) return
    setDownloading(true)
    try {
      await downloadViaApi(downloadUrl, filename)
    } catch {
      window.open(`${apiClient.getBaseUrl()}${downloadUrl}`, '_blank')
    } finally {
      setDownloading(false)
    }
  }, [downloadUrl, filename])

  const handleDelete = useCallback(() => {
    if (!deliverable) return
    deleteMutation.mutate(deliverable.id, {
      onSuccess: () => onOpenChange(false),
    })
  }, [deliverable, deleteMutation, onOpenChange])

  const handleOpenInCanvas = () => {
    if (!deliverable) return
    const path = encodeURIComponent(deliverable.file_path)
    router.push(`/workspace?view=explorer&path=${path}`)
    onOpenChange(false)
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full overflow-y-auto sm:max-w-2xl lg:max-w-3xl"
      >
        {isLoading && (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}

        {isError && !isLoading && (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-destructive">Failed to load deliverable</p>
          </div>
        )}

        {deliverable && !isLoading && (
          <div className="flex flex-col gap-6">
            <SheetHeader className="space-y-3">
              <SheetTitle className="pr-8 text-left text-lg leading-tight">
                {deliverable.title}
              </SheetTitle>
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                {deliverable.agent_name && (
                  <>
                    <span>{deliverable.agent_name}</span>
                    <span>·</span>
                  </>
                )}
                <span>
                  {formatDistanceToNow(new Date(deliverable.created_at), {
                    addSuffix: true,
                  })}
                </span>
                <span>·</span>
                <span className="capitalize">{deliverable.artifact_type}</span>
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                {downloadUrl && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDownload}
                    disabled={downloading}
                  >
                    {downloading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Download className="mr-2 h-4 w-4" />
                    )}
                    Download
                  </Button>
                )}
                <Button variant="outline" size="sm" onClick={handleOpenInCanvas}>
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open in Explorer
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDelete}
                  disabled={deleteMutation.isLoading}
                  className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                >
                  {deleteMutation.isLoading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="mr-2 h-4 w-4" />
                  )}
                  Delete
                </Button>
              </div>
            </SheetHeader>

            {/* Content body — delegated to shared FilePreview */}
            <div>
              {deliverable.content_error ? (
                <ContentUnavailable
                  message={`Unable to load content: ${deliverable.content_error}`}
                  downloadUrl={downloadUrl}
                  filename={filename}
                />
              ) : (
                <FilePreview
                  content={
                    typeof deliverable.content === 'string'
                      ? deliverable.content
                      : undefined
                  }
                  url={downloadUrl ?? undefined}
                  previewType={getPreviewTypeForDeliverable(deliverable)}
                  filename={filename}
                  className="rounded-lg border border-border/50"
                />
              )}
            </div>

            {/* Summary card */}
            {deliverable.summary && (
              <div className="rounded-lg border border-border/50 bg-muted/10 p-4">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Summary
                </h3>
                <p className="text-sm text-foreground/90">{deliverable.summary}</p>
              </div>
            )}
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
