'use client'

/**
 * Download a file that an authenticated API route serves (a generated document under
 * /api/documents/generated/…, a knowledge document under /api/documents/{id}/download).
 *
 * A plain <a href> or window.open on that relative path goes to the frontend's own origin
 * and sends no Authorization header, so it fails in both editions (F358). This hands the
 * path to downloadGeneratedFile, which fetches it from the API base URL with the API
 * client's auth headers and saves the bytes, as the Deliverables panel does.
 */
import { useCallback, useState } from 'react'
import { toast } from 'sonner'

import { downloadGeneratedFile } from '@/components/documents/blocks/api'

export const API_FILE_DOWNLOAD_FAILED = "Couldn't download the file. Try again from Deliverables."

/** `failedMessage` is the toast a failed download shows; the default points to Deliverables. */
export function useApiFileDownload(failedMessage: string = API_FILE_DOWNLOAD_FAILED): {
  download: (url: string, filename: string) => Promise<void>
  downloading: boolean
} {
  const [downloading, setDownloading] = useState(false)

  const download = useCallback(async (url: string, filename: string) => {
    setDownloading(true)
    try {
      await downloadGeneratedFile(url, filename)
    } catch (error) {
      console.error('[api-file-download] download failed', url, error)
      toast.error(failedMessage)
    } finally {
      setDownloading(false)
    }
  }, [failedMessage])

  return { download, downloading }
}

/** The name to save a file under: the one given, else the last segment of its route. */
export function downloadFilename(url: string, filename?: string | null): string {
  if (filename) return filename
  const lastSegment = url.split('?')[0].split('/').filter(Boolean).pop()
  return lastSegment || 'download'
}
