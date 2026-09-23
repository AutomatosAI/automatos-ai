'use client'

/**
 * PRD-251 S0.5 — "New draft": a title, a brief and the base copy. It creates a
 * `draft` post through POST /api/socials/posts; the list refetches and shows it
 * under Draft.
 */
import { useState, type FormEvent } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialPost } from '@/lib/api-client'
import { useCreateSocialPost } from '@/hooks/use-socials-api'
import { SOCIAL_POST_TITLE_MAX_CHARS } from './socials-status'

interface SocialsNewDraftProps {
  /** Called with the created post, or null when the form is cancelled. */
  onDone: (post: SocialPost | null) => void
}

export function SocialsNewDraft({ onDone }: SocialsNewDraftProps) {
  const create = useCreateSocialPost()
  const [title, setTitle] = useState('')
  const [brief, setBrief] = useState('')
  const [base, setBase] = useState('')
  const titleOk = title.trim().length > 0

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!titleOk || create.isLoading) return
    create.mutate(
      { title: title.trim(), brief: brief.trim() || null, copy: { base } },
      { onSuccess: (post) => onDone(post) },
    )
  }

  return (
    <form
      onSubmit={handleSubmit}
      aria-label="New draft"
      className="space-y-3 rounded-xl border border-border bg-card/40 p-4"
    >
      <div className="space-y-1.5">
        <Label htmlFor="socials-draft-title">Title</Label>
        <Input
          id="socials-draft-title"
          value={title}
          maxLength={SOCIAL_POST_TITLE_MAX_CHARS}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="What the post is called on the board"
          required
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="socials-draft-brief">Brief</Label>
        <Textarea
          id="socials-draft-brief"
          value={brief}
          rows={2}
          onChange={(event) => setBrief(event.target.value)}
          placeholder="The ask this post answers"
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="socials-draft-copy">Copy</Label>
        <Textarea
          id="socials-draft-copy"
          value={base}
          rows={4}
          onChange={(event) => setBase(event.target.value)}
          placeholder="The base text every channel starts from"
        />
      </div>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" size="sm" onClick={() => onDone(null)}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!titleOk || create.isLoading}>
          {create.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
          Create draft
        </Button>
      </div>
    </form>
  )
}
