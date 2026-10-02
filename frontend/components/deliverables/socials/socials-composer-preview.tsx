'use client'

/**
 * PRD-251 S2.2b (US-208) — the composer's preview, like the Template Studio's
 * PreviewPane: a video previews at half resolution (the post's `preview`), an
 * image through its real render (the post's media). Both show through the shared
 * FilePreview. An edit after the render marks it stale: "Preview out of date —
 * render again".
 */
import { Loader2, Play } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { FilePreview, inferPreviewType } from '@/components/widgets/FileWidget/FilePreview'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPostMedia } from '@/hooks/use-socials-api'

export const PREVIEW_STALE_MESSAGE = 'Preview out of date — render again'

interface PreviewFile {
  key: string
  url: string
  name: string
  contentType: string
  label: string
}

function PreviewFiles({ files }: { files: PreviewFile[] }) {
  return (
    <div className="grid gap-3">
      {files.map((file) => (
        <figure key={file.key} className="space-y-1">
          <div className="h-80 overflow-hidden rounded-lg border border-border/60">
            <FilePreview url={file.url} filename={file.name} previewType={inferPreviewType(file.name, file.contentType)} />
          </div>
          <figcaption className="text-xs text-muted-foreground">{file.label}</figcaption>
        </figure>
      ))}
    </div>
  )
}

/** The files the preview shows: a video's preview render, an image's rendered media. */
function usePreviewFiles(post: SocialPost | undefined, video: boolean): PreviewFile[] {
  const hasMedia = !video && !!post && Object.keys(post.media ?? {}).length > 0
  const media = useSocialPostMedia(post?.id ?? '', post?.content_hash ?? '', hasMedia)
  if (!post) return []
  if (video) {
    return (post.preview?.status === 'done' ? post.preview.files : []).map((f) => ({
      key: f.name, url: f.url, name: f.name, contentType: f.content_type ?? '', label: `${f.width ?? ''}×${f.height ?? ''}`,
    }))
  }
  return (media.data ?? []).filter((f) => f.url).map((f) => ({
    key: `${f.aspect}-${f.deliverable_id}`, url: f.url as string, name: f.name ?? '', contentType: f.content_type ?? '', label: f.aspect,
  }))
}

interface SocialsComposerPreviewProps {
  post: SocialPost | undefined
  video: boolean
  /** The draft on screen differs from what was rendered. */
  stale: boolean
  busy: boolean
  onRender: () => void
}

export function SocialsComposerPreview({ post, video, stale, busy, onRender }: SocialsComposerPreviewProps) {
  const files = usePreviewFiles(post, video)
  const rendering = post?.status === 'rendering' || post?.preview?.status === 'rendering'
  const failed = video ? (post?.preview?.status === 'failed' ? post.preview.error : null) : null
  return (
    <section aria-label="Preview" className="space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-foreground">Preview</h4>
        <Button type="button" size="sm" variant="outline" onClick={onRender} disabled={busy || rendering}>
          {busy || rendering ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : <Play className="mr-2 h-4 w-4" aria-hidden />}
          {files.length > 0 ? 'Render again' : video ? 'Render a preview' : 'Render'}
        </Button>
      </div>
      {rendering && <p role="status" className="text-sm text-muted-foreground">Rendering… this can take a few minutes.</p>}
      {failed && <p role="alert" className="text-sm text-destructive">The preview failed: {failed}</p>}
      {stale && files.length > 0 && !rendering && (
        <p role="status" className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-foreground">
          {PREVIEW_STALE_MESSAGE}
        </p>
      )}
      {files.length > 0 ? (
        <PreviewFiles files={files} />
      ) : (
        !rendering && <p className="text-sm text-muted-foreground">No preview yet.</p>
      )}
    </section>
  )
}
