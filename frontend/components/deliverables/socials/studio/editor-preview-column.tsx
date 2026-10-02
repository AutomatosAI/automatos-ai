'use client'

/**
 * PRD-251B US-B109 — the editor's right column until US-B110's per-channel preview: the
 * base copy every channel starts from, and the post's render (a video's half-resolution
 * preview, an image's render) through the shared FilePreview, marked out of date after an
 * edit.
 */
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialPost } from '@/lib/api-client'
import { SocialsComposerPreview } from '../socials-composer-preview'
import { CARD, Hint } from './editor-ui'

interface EditorPreviewColumnProps {
  post: SocialPost | null
  base: string
  video: boolean
  stale: boolean
  busy: boolean
  onBase: (base: string) => void
  onRender: () => void
}

export function EditorPreviewColumn({ post, base, video, stale, busy, onBase, onRender }: EditorPreviewColumnProps) {
  return (
    <section aria-label="Preview and copy" className={`${CARD} lg:sticky lg:top-4`}>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="socials-editor-copy">Copy</Label>
        <Textarea id="socials-editor-copy" aria-label="Copy" rows={5} value={base} onChange={(event) => onBase(event.target.value)} />
        <Hint>Each channel starts from the same base copy.</Hint>
      </div>
      <SocialsComposerPreview post={post ?? undefined} video={video} stale={stale} busy={busy} onRender={onRender} />
    </section>
  )
}
