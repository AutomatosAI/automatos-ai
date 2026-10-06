'use client'

/**
 * PRD-251B US-B301 — the logo and the logo mark on the Brand kit tab: upload (PNG or JPEG,
 * checked on the server), replace, remove, or a public image URL while none is stored.
 * PRD-255: the logo's dark and mono variants use it too, uploads only (no URL field).
 */
import { useRef, type ChangeEvent } from 'react'
import { ImagePlus, Loader2, Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FieldHelp } from '@/components/ui/help-tooltip'
import type { BrandImage } from '@/components/documents/blocks/useBrandImage'

interface BrandImageFieldProps {
  title: string
  help?: string
  image: BrandImage
  stored: boolean
  url?: string
  /** The words on the upload button while none is stored, and once one is. */
  uploadLabel: string
  replaceLabel: string
  fileLabel?: string
  urlId?: string
  urlPlaceholder?: string
  disabled: boolean
  /** Absent: the image is an upload only, with no URL field. */
  onUrl?: (url: string) => void
}

export function BrandImageField(props: BrandImageFieldProps) {
  const { title, help, image, stored, url, uploadLabel, replaceLabel, fileLabel, urlId, urlPlaceholder, disabled, onUrl } = props
  const input = useRef<HTMLInputElement | null>(null)
  const pick = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    void image.upload(file)
  }
  return (
    <div className="rounded-md border p-3">
      <Label className="flex items-center text-xs">
        {title} {help && <FieldHelp id={help} />}
      </Label>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input ref={input} type="file" accept="image/png,image/jpeg" className="hidden" aria-label={fileLabel} onChange={pick} />
        <Button type="button" size="sm" variant="outline" className="gap-1.5" disabled={disabled || image.busy} onClick={() => input.current?.click()}>
          {image.busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden /> : <ImagePlus className="h-3.5 w-3.5" aria-hidden />}
          {stored ? replaceLabel : uploadLabel}
        </Button>
        {stored && (
          <Button type="button" size="sm" variant="ghost" className="gap-1.5 text-destructive" disabled={disabled || image.busy} onClick={image.remove}>
            <Trash2 className="h-3.5 w-3.5" aria-hidden /> Remove
          </Button>
        )}
      </div>
      {!stored && onUrl && (
        <div className="mt-2">
          <Label htmlFor={urlId} className="text-xs text-muted-foreground">…or a public image URL</Label>
          <Input id={urlId} value={url ?? ''} disabled={disabled} onChange={(e) => onUrl(e.target.value)} placeholder={urlPlaceholder} />
        </div>
      )}
    </div>
  )
}
