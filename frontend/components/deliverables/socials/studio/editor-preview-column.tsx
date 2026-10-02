'use client'

/**
 * PRD-251B US-B109/B110 — the editor's right column: the base copy every channel starts
 * from, and the Preview (studio/socials-preview.tsx), a tab per ticked channel.
 */
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialChannel, SocialPost } from '@/lib/api-client'
import type { EditorDraft } from './editor-model'
import { CARD, Hint } from './editor-ui'
import { SocialsPreview } from './socials-preview'

interface EditorPreviewColumnProps {
  post: SocialPost | null
  draft: EditorDraft
  channels: ReadonlyArray<SocialChannel>
  templateSizes: ReadonlyArray<string>
  onBase: (base: string) => void
  onChannelCopy: (toolkit: string, text: string) => void
}

export function EditorPreviewColumn({ post, draft, channels, templateSizes, onBase, onChannelCopy }: EditorPreviewColumnProps) {
  return (
    <div className="flex min-w-0 flex-col gap-4 lg:sticky lg:top-4">
      <section aria-label="Base copy" className={CARD}>
        <Label htmlFor="socials-editor-copy">Copy</Label>
        <Textarea id="socials-editor-copy" aria-label="Copy" rows={4} value={draft.base} onChange={(event) => onBase(event.target.value)} />
        <Hint>The copy every channel starts from.</Hint>
      </section>
      <SocialsPreview post={post} draft={draft} channels={channels} templateSizes={templateSizes} onChannelCopy={onChannelCopy} />
    </div>
  )
}
