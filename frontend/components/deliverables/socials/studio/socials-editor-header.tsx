'use client'

/**
 * PRD-251B US-B109 — the editor's header (Editor.dc.html): Back to calendar, the title
 * with the post's status, when it publishes, and the three actions: Save draft, Render
 * preview, Submit for approval (orange, owner choice 2). ``extra`` goes before them (the
 * post's Delete, 3 Oct 2026).
 */
import type { ReactNode } from 'react'
import { ArrowLeft, Loader2 } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { SocialPost } from '@/lib/api-client'
import { slotDateLabel } from '@/lib/social-time'
import { SOCIAL_STATUS_LABELS } from '../socials-status'
import { slotInput, type EditorDraft } from './editor-model'
import { PRIMARY_ACTION } from './studio-nav'

export const SUBMITTABLE_STATUSES: ReadonlySet<string> = new Set(['draft', 'changes_requested'])

interface SocialsEditorHeaderProps {
  post: SocialPost | null
  draft: EditorDraft
  busy: 'save' | 'render' | 'submit' | null
  overLimit: boolean
  /** The post is the person's own file: nothing to render, so Render preview is off. */
  nothingToRender?: boolean
  onTitle: (title: string) => void
  onBack: () => void
  onSave: () => void
  onRender: () => void
  onSubmit: () => void
  /** More actions before Save draft (the post's Delete). */
  extra?: ReactNode
}

function publishesLine(draft: EditorDraft): string | null {
  const slot = slotInput(draft.slot)
  return slot.plannedFor ? `Publishes ${slotDateLabel(slot.plannedFor, slot.timezone)} once approved` : null
}

export function SocialsEditorHeader(props: SocialsEditorHeaderProps) {
  const { post, draft, busy, overLimit, nothingToRender = false, onTitle, onBack, onSave, onRender, onSubmit, extra } = props
  const status = post ? SOCIAL_STATUS_LABELS[post.status] : 'New'
  const canSubmit = !post || SUBMITTABLE_STATUSES.has(post.status)
  const line = publishesLine(draft)
  const spinner = (what: typeof busy) => busy === what && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden />
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="flex min-w-0 flex-col gap-1.5">
        <Button type="button" variant="ghost" size="sm" className="-ml-2 w-fit text-muted-foreground" onClick={onBack}>
          <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden />
          Back to calendar
        </Button>
        <div className="flex flex-wrap items-center gap-3">
          <input
            aria-label="Title"
            value={draft.title}
            placeholder="Untitled post"
            onChange={(event) => onTitle(event.target.value)}
            className="min-w-0 bg-transparent font-serif text-[32px] font-normal leading-[1.1] tracking-[-0.01em] text-foreground outline-none max-md:text-[26px]"
          />
          <Badge variant="outline" className="h-6 rounded-full px-2.5" data-testid="socials-editor-status">
            {status}
          </Badge>
        </div>
        {line && <p className="m-0 text-[12.5px] text-muted-foreground">{line}</p>}
      </div>
      <div className="socials-editor-actions flex flex-wrap gap-2">
        {extra}
        <Button type="button" variant="secondary" onClick={onSave} disabled={busy !== null}>
          {spinner('save')}Save draft
        </Button>
        <Button type="button" variant="secondary" onClick={onRender} disabled={busy !== null || draft.format === 'text' || nothingToRender}>
          {spinner('render')}Render preview
        </Button>
        {canSubmit && (
          <Button type="button" className={PRIMARY_ACTION} onClick={onSubmit} disabled={busy !== null || overLimit}>
            {spinner('submit')}Submit for approval
          </Button>
        )}
      </div>
    </div>
  )
}
