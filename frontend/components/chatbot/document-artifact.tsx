'use client'

/**
 * A generated document in the chat's artifact viewer: its name, the PDF preview and the
 * Download button.
 *
 * preview_url and download_url are API routes behind auth (/api/documents/generated/…).
 * An <iframe src> or window.open on them reaches the frontend's own origin without the
 * Authorization header, so neither worked (F358). The preview goes through FilePreview,
 * which fetches the PDF with the API client's auth and frames a blob: URL, and the
 * download through useApiFileDownload, as the Deliverables panel does.
 */
import { Download, FileText, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { FilePreview } from '@/components/widgets/FileWidget/FilePreview'
import { downloadFilename, useApiFileDownload } from '@/hooks/use-api-file-download'
import type { Artifact } from '@/types'

export function DocumentArtifact({ artifact }: { artifact: Artifact }) {
  const metadata = artifact.metadata ?? {}
  const format = typeof metadata.format === 'string' ? metadata.format : undefined
  const previewUrl = typeof metadata.preview_url === 'string' ? metadata.preview_url : undefined
  const downloadUrl = typeof metadata.download_url === 'string' ? metadata.download_url : undefined
  const { download, downloading } = useApiFileDownload()

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <FileText className="h-8 w-8 text-primary" />
        <div>
          <h3 className="font-semibold">{artifact.title}</h3>
          <p className="text-sm text-muted-foreground">
            {format?.toUpperCase()} {metadata.size_kb ? `• ${metadata.size_kb}KB` : ''}
          </p>
        </div>
      </div>
      {format === 'pdf' && previewUrl && (
        <div className="h-[600px] overflow-hidden rounded-lg border">
          <FilePreview url={previewUrl} previewType="pdf" filename={artifact.title} />
        </div>
      )}
      <div className="flex gap-2">
        {downloadUrl && (
          <Button
            onClick={() => download(downloadUrl, downloadFilename(downloadUrl, metadata.filename))}
            disabled={downloading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            {downloading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Download className="h-4 w-4 mr-2" />}
            Download {format?.toUpperCase()}
          </Button>
        )}
        {format && format !== 'pdf' && (
          <Button variant="outline" disabled>
            <FileText className="h-4 w-4 mr-2" />
            Convert to PDF
          </Button>
        )}
      </div>
    </div>
  )
}
