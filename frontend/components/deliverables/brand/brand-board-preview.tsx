'use client'

/**
 * PRD-255 US-010 — the brand board at the top of the Brand kit tab: page 1 of the board
 * (GET /api/documents/brand-kit/board?format=png), printed by the server from the kit alone,
 * with the PDF and the PNG to download. The picture and the files are fetched with the caller's auth.
 *
 * F372: the picture is keyed on the kit's `updated_at` (useBrandKitStamp), not on the page's own
 * saves, so a change by Auto, the designer or the API draws it again too: the stamp is read on
 * load, when the window regains focus, every KIT_STAMP_POLL_MS, and after each change the page
 * stores (`changes`). The board route answers no-store, so the new path is always a fresh print.
 */
import { useState } from 'react'
import { Download, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

import { brandBoardPath, downloadGeneratedFile, type BrandBoardFormat } from '@/components/documents/blocks/api'
import { Button } from '@/components/ui/button'
import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import { useBrandKitStamp } from '@/hooks/use-brand-kit-stamp'

export const BOARD_FILE_NAME: Record<BrandBoardFormat, string> = { pdf: 'brand-board.pdf', png: 'brand-board.png' }
export const BOARD_UNAVAILABLE = 'The brand board could not be drawn. Try again in a moment.'
const DOWNLOADS: { format: BrandBoardFormat; label: string }[] = [
  { format: 'pdf', label: 'Download PDF' },
  { format: 'png', label: 'Download PNG' },
]

interface BrandBoardPreviewProps {
  /** Counts the changes the page itself stored: each one reads the kit's stamp again. */
  changes: number
}

export function BrandBoardPreview({ changes }: BrandBoardPreviewProps) {
  const stamp = useBrandKitStamp(changes)
  const { src, error } = useAuthenticatedBlobUrl(stamp === undefined ? undefined : brandBoardPath('png', stamp))
  const [downloading, setDownloading] = useState<BrandBoardFormat | null>(null)

  const download = async (format: BrandBoardFormat) => {
    setDownloading(format)
    try {
      await downloadGeneratedFile(brandBoardPath(format), BOARD_FILE_NAME[format])
    } catch {
      toast.error(BOARD_UNAVAILABLE)
    } finally {
      setDownloading(null)
    }
  }

  return (
    <section aria-label="Brand board" className="flex flex-col gap-3 rounded-xl border bg-card p-4 md:flex-row md:items-start">
      <div className="w-full max-w-xs shrink-0 overflow-hidden rounded-md border bg-white">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt="Brand board, page 1" className="h-auto w-full" />
        ) : (
          <div className="flex aspect-[210/297] items-center justify-center p-4 text-center text-xs text-muted-foreground">
            {error ? BOARD_UNAVAILABLE : <Loader2 className="h-5 w-5 animate-spin" aria-label="Drawing the brand board" />}
          </div>
        )}
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="text-base font-semibold text-foreground">Brand board</h3>
        <p className="text-sm text-muted-foreground">
          Your whole brand on one page, drawn from this kit: share it with anyone who makes things for you. It is drawn again each time the kit changes, here or by Auto.
        </p>
        <div className="flex flex-wrap gap-2">
          {DOWNLOADS.map(({ format, label }) => (
            <Button key={format} variant="outline" size="sm" disabled={downloading !== null} onClick={() => void download(format)}>
              {downloading === format ? <Loader2 className="mr-1 h-4 w-4 animate-spin" /> : <Download className="mr-1 h-4 w-4" />}
              {label}
            </Button>
          ))}
        </div>
      </div>
    </section>
  )
}
