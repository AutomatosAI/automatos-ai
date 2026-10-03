'use client'

/** PRD-251B US-B109 — the editor's Brief: what the post is about, and Redraft with Auto
 * (the composer, with the editor's format, channels, template and length: US-B103). */
import { Loader2, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { EditorCard } from './editor-ui'

interface EditorBriefCardProps {
  brief: string
  busy: boolean
  canRedraft: boolean
  onChange: (brief: string) => void
  onRedraft: () => void
}

export function EditorBriefCard({ brief, busy, canRedraft, onChange, onRedraft }: EditorBriefCardProps) {
  const action = canRedraft ? (
    <Button type="button" size="sm" variant="secondary" onClick={onRedraft} disabled={busy || !brief.trim()}>
      {busy ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden /> : <Sparkles className="mr-1.5 h-4 w-4" aria-hidden />}
      Redraft with Auto
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
    </EditorCard>
  )
}
