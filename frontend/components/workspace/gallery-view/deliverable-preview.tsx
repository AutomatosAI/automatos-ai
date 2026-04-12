/**
 * DeliverablePreview (PRD-129: Workspace Outputs Hub)
 * =====================================================
 *
 * Slide-over sheet that shows a single deliverable's content with Download,
 * Delete, and "Open in Explorer" actions. Fetches full content via useDeliverable
 * with include_content=true. Renders images, code (Prism), markdown/reports
 * (react-markdown), and plain text. Unsupported types show a friendly
 * fallback with a Download link.
 */

'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { formatDistanceToNow } from 'date-fns'
import ReactMarkdown from 'react-markdown'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-json'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-sql'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-markup'
import 'prismjs/components/prism-go'
import 'prismjs/components/prism-yaml'
import 'prismjs/components/prism-markdown'
import 'prismjs/components/prism-docker'
import {
  Download,
  ExternalLink,
  FileWarning,
  Loader2,
  Trash2,
} from 'lucide-react'

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

const EXT_TO_LANG: Record<string, string> = {
  py: 'python',
  ts: 'typescript',
  tsx: 'typescript',
  js: 'javascript',
  jsx: 'javascript',
  json: 'json',
  sh: 'bash',
  bash: 'bash',
  zsh: 'bash',
  sql: 'sql',
  css: 'css',
  html: 'markup',
  xml: 'markup',
  go: 'go',
  yaml: 'yaml',
  yml: 'yaml',
  md: 'markdown',
  markdown: 'markdown',
  dockerfile: 'docker',
}

export function getLanguageFromPath(filePath: string | null | undefined): string {
  if (!filePath) return ''
  const base = filePath.split('/').pop() || ''
  if (base.toLowerCase() === 'dockerfile') return 'docker'
  const lastDot = base.lastIndexOf('.')
  if (lastDot === -1) return ''
  const ext = base.slice(lastDot + 1).toLowerCase()
  return EXT_TO_LANG[ext] || ''
}

function isMarkdownDeliverable(d: Deliverable): boolean {
  if (d.artifact_type === 'report') return true
  const ft = (d.file_type || '').toLowerCase()
  if (ft === 'md' || ft === 'markdown') return true
  const lang = getLanguageFromPath(d.file_path)
  return lang === 'markdown'
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

// ============= CONTENT RENDERERS =============

function ImageContent({ url, title }: { url: string; title: string }) {
  return (
    <div className="flex items-center justify-center rounded-lg border border-border/50 bg-muted/20 p-4">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={title}
        className="max-h-[70vh] w-auto max-w-full rounded object-contain"
      />
    </div>
  )
}

function CodeContent({ code, language }: { code: string; language: string }) {
  const codeRef = useRef<HTMLElement>(null)

  useEffect(() => {
    if (codeRef.current) {
      Prism.highlightElement(codeRef.current)
    }
  }, [code, language])

  return (
    <pre className="overflow-x-auto rounded-lg border border-border/50 bg-[#1a1a1a] p-4 text-[13px] leading-relaxed">
      <code
        ref={codeRef}
        className={language ? `language-${language}` : undefined}
      >
        {code}
      </code>
    </pre>
  )
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="prose prose-sm dark:prose-invert max-w-none rounded-lg border border-border/50 bg-background p-6">
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  )
}

function PlainTextContent({ content }: { content: string }) {
  return (
    <pre className="overflow-x-auto whitespace-pre-wrap rounded-lg border border-border/50 bg-muted/20 p-4 text-[13px] leading-relaxed">
      {content}
    </pre>
  )
}

function UnavailableContent({
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
      // fallback: open in new tab
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

  const language = useMemo(
    () => (deliverable ? getLanguageFromPath(deliverable.file_path) : ''),
    [deliverable],
  )

  const downloadUrl = deliverable?.content_url || deliverable?.preview_url || null
  const filename = deliverable?.file_name || deliverable?.file_path?.split('/').pop() || 'download'

  const handleDownload = useCallback(async () => {
    if (!downloadUrl) return
    setDownloading(true)
    try {
      await downloadViaApi(downloadUrl, filename)
    } catch {
      // fallback: open in new tab
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

            {/* Content body */}
            <div>
              <PreviewBody deliverable={deliverable} language={language} />
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

function PreviewBody({
  deliverable,
  language,
}: {
  deliverable: Deliverable
  language: string
}) {
  const filename = deliverable.file_name || deliverable.file_path?.split('/').pop() || 'download'

  // 1. Image — render <img> via content_url (backend provides this for images)
  if (deliverable.artifact_type === 'image') {
    const url = deliverable.content_url || deliverable.preview_url
    if (!url) {
      return (
        <UnavailableContent
          message="Image preview unavailable"
          downloadUrl={null}
          filename={filename}
        />
      )
    }
    return <ImageContent url={url} title={deliverable.title} />
  }

  // Error fetching content
  if (deliverable.content_error) {
    return (
      <UnavailableContent
        message={`Unable to load content: ${deliverable.content_error}`}
        downloadUrl={deliverable.content_url ?? deliverable.preview_url ?? null}
        filename={filename}
      />
    )
  }

  // 2. Code — syntax highlight via Prism
  if (deliverable.artifact_type === 'code') {
    if (typeof deliverable.content !== 'string') {
      return (
        <UnavailableContent
          message="Code content unavailable"
          downloadUrl={deliverable.content_url ?? deliverable.preview_url ?? null}
          filename={filename}
        />
      )
    }
    return <CodeContent code={deliverable.content} language={language} />
  }

  // 3. Report or markdown — render via react-markdown
  if (isMarkdownDeliverable(deliverable)) {
    if (typeof deliverable.content !== 'string') {
      return (
        <UnavailableContent
          message="Report content unavailable"
          downloadUrl={deliverable.content_url ?? deliverable.preview_url ?? null}
          filename={filename}
        />
      )
    }
    return <MarkdownContent content={deliverable.content} />
  }

  // 4. Other text content — plain <pre>
  if (typeof deliverable.content === 'string') {
    return <PlainTextContent content={deliverable.content} />
  }

  // 5. Unsupported
  return (
    <UnavailableContent
      message="Preview not available for this file type"
      downloadUrl={deliverable.content_url ?? deliverable.preview_url ?? null}
      filename={filename}
    />
  )
}
