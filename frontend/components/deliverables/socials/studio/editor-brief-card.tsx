'use client'

/** PRD-251B US-B109 — the editor's Brief: what the post is about, and Write with Auto
 * (the composer, with the editor's format, channels, template and length: US-B103, and the
 * fields the person filled in, which it keeps). It was "Redraft with Auto": a new post has
 * nothing to redraft, and the button is how the caption and the image's words get written.
 * F378: afterwards, what Auto still needs from the owner ("Auto needs: …"). */
import { Loader2, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { EditorCard, Hint } from './editor-ui'

export const WRITE_WITH_AUTO = 'Write with Auto'
export const BRIEF_HINT = 'Auto writes the caption and the words on the image from this. Anything you have written yourself is kept.'

interface EditorBriefCardProps {
  brief: string
  busy: boolean
  canRedraft: boolean
  onChange: (brief: string) => void
  onRedraft: () => void
  /** F378: "Auto needs: …" after a redraft, what the brief did not give. */
  needs?: string | null
}

export function EditorBriefCard({ brief, busy, canRedraft, onChange, onRedraft, needs }: EditorBriefCardProps) {
  const action = canRedraft ? (
    <Button type="button" size="sm" variant="secondary" onClick={onRedraft} disabled={busy || !brief.trim()}>
      {busy ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden /> : <Sparkles className="mr-1.5 h-4 w-4" aria-hidden />}
      {WRITE_WITH_AUTO}
    </Button>
  ) : null
  return (
    <EditorCard label="Brief" action={action}>
      <Textarea
        aria-label="Brief"
        value={brief}
        rows={3}
        placeholder="What this post says, and for whom"
        onChange={(event) => onChange(event.target.value)}
      />
      {canRedraft && <Hint>{BRIEF_HINT}</Hint>}
      {needs && <p role="status" className="mt-2 text-[13px] text-warning">{needs}</p>}
    </EditorCard>
  )
}
