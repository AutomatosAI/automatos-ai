'use client'

/**
 * PRD-251B US-B109 — where a post's visual comes from, beside a template: a file the
 * person drops (POST /api/socials/posts/{id}/media; the server sniffs the type and holds
 * the limits) or one of the workspace's image and video Deliverables (the Library). For a
 * template's photo spot (editor-look-photo.tsx) both take pictures only. A Library picture's
 * preview_url is an API path behind auth, so it loads as the Outputs gallery loads it (3 Oct
 * 2026: a plain <img> showed none).
 */
import { useDropzone } from 'react-dropzone'
import { Loader2, Upload } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useAuthenticatedBlobUrl } from '@/components/widgets/FileWidget/FilePreview'
import { cn } from '@/lib/utils'
import type { DeliverableSummary } from '@/lib/api-client'
import { HINT, Hint } from './editor-ui'

export const DROP_TITLE = 'Drop an image or video here'
export const DROP_HINT = 'PNG, JPG or MP4. It is cropped for each channel, and you check each crop in the preview.'
export const DROP_PHOTO_TITLE = 'Drop a photo here'
export const DROP_PHOTO_HINT = 'PNG, JPEG or WebP. The template fits it behind its words.'
const PICTURES = { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'], 'image/webp': ['.webp'] }
const ACCEPT = { ...PICTURES, 'video/mp4': ['.mp4'] }

interface UploadDropProps {
  busy: boolean
  onFile: (file: File) => void
  /** PRD-251B: the file fills a template's photo spot, which takes a picture only. */
  pictureOnly?: boolean
}

export function UploadDrop({ busy, onFile, pictureOnly = false }: UploadDropProps) {
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: pictureOnly ? PICTURES : ACCEPT,
    multiple: false,
    noClick: true,
    disabled: busy,
    onDropAccepted: (files) => files[0] && onFile(files[0]),
  })
  return (
    <div
      {...getRootProps()}
      aria-label="Upload a file"
      className={cn(
        'flex min-h-[150px] flex-col items-center justify-center gap-2 rounded-xl border-[1.5px] border-dashed border-border p-4 text-center',
        isDragActive && 'border-accent bg-accent/10',
      )}
    >
      <input {...getInputProps()} aria-label="File to upload" />
      {busy ? <Loader2 className="h-7 w-7 animate-spin text-muted-foreground" aria-hidden /> : <Upload className="h-7 w-7 text-muted-foreground" aria-hidden />}
      <span className="font-medium text-foreground">{pictureOnly ? DROP_PHOTO_TITLE : DROP_TITLE}</span>
      <span className={HINT}>{pictureOnly ? DROP_PHOTO_HINT : DROP_HINT}</span>
      <Button type="button" size="sm" variant="secondary" onClick={open} disabled={busy}>
        Browse files
      </Button>
    </div>
  )
}

const THUMB = 'aspect-[4/3] w-full rounded-lg'

/** A Library picture, fetched with the caller's auth; a blank tile until it arrives. */
function LibraryThumb({ url }: { url: string }) {
  const { src } = useAuthenticatedBlobUrl(url)
  if (!src) return <span className={cn(THUMB, 'block bg-muted')} aria-hidden />
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt="" className={cn(THUMB, 'object-cover')} />
}

interface LibraryGridProps {
  items: ReadonlyArray<DeliverableSummary> | undefined
  loading: boolean
  busy: boolean
  onPick: (item: DeliverableSummary) => void
}

export function LibraryGrid({ items, loading, busy, onPick }: LibraryGridProps) {
  if (loading) return <Hint>Loading the Library…</Hint>
  if (!items || items.length === 0) return <Hint>No images or videos in Deliverables yet.</Hint>
  return (
    <ul aria-label="Library" className="grid grid-cols-2 gap-3 lg:grid-cols-3">
      {items.map((item) => (
        <li key={item.id}>
          <button
            type="button"
            disabled={busy}
            onClick={() => onPick(item)}
            className="flex w-full flex-col gap-1.5 rounded-xl border border-border bg-background/60 p-2 text-left"
          >
            {item.artifact_type === 'image' && item.preview_url ? (
              <LibraryThumb url={item.preview_url} />
            ) : (
              <span className={cn(THUMB, 'flex items-center justify-center bg-muted font-mono text-xs uppercase text-muted-foreground')}>
                {item.artifact_type}
              </span>
            )}
            <span className="truncate text-[13px] font-medium text-foreground">{item.title}</span>
            <span className={HINT}>{item.artifact_type === 'video' ? 'Video' : 'Image'}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}
