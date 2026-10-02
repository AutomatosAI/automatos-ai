'use client'

/**
 * PRD-251B US-B109 — the editor's Look (Editor.dc.html): where the visual comes from.
 * Template: the format's templates (GET /api/socials/templates), "Let Auto pick" first
 * (no template), each with its thumbnail (a format placeholder until the backfill made
 * one) and name. Upload: a file of the person's own. Library: one of the workspace's
 * image and video Deliverables. AI-made visuals are Wave 3: not offered here.
 */
import { useState } from 'react'

import { cn } from '@/lib/utils'
import type { DeliverableSummary, SocialTemplateSummary } from '@/lib/api-client'
import { useSocialLibrary } from '@/hooks/use-socials-editor'
import { LibraryGrid, UploadDrop } from './editor-look-sources'
import { EditorCard, Hint, Segmented } from './editor-ui'

export type LookSource = 'template' | 'upload' | 'library'
export const LOOK_SOURCES: ReadonlyArray<{ value: LookSource; label: string }> = [
  { value: 'template', label: 'Template' },
  { value: 'upload', label: 'Upload' },
  { value: 'library', label: 'Library' },
]
export const AUTO_PICK = 'Let Auto pick'

interface TemplateCardProps {
  name: string
  thumbnail: string | null
  kind: string
  chosen: boolean
  onPick: () => void
}

function TemplateCard({ name, thumbnail, kind, chosen, onPick }: TemplateCardProps) {
  return (
    <li>
      <button
        type="button"
        aria-pressed={chosen}
        onClick={onPick}
        className={cn(
          'flex w-full flex-col gap-2 rounded-xl border-2 bg-background/60 p-2 text-left',
          chosen ? 'border-accent' : 'border-transparent',
        )}
      >
        {thumbnail ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumbnail} alt="" className="aspect-[4/3] w-full rounded-lg object-cover" />
        ) : (
          <span className="flex aspect-[4/3] w-full items-end rounded-lg bg-[#141210] p-2.5 font-serif text-base leading-[1.1] text-foreground">
            {kind}
          </span>
        )}
        <span className="text-[13px] font-medium text-foreground">{name}</span>
      </button>
    </li>
  )
}

interface TemplateGalleryProps {
  templates: ReadonlyArray<SocialTemplateSummary>
  loading: boolean
  chosen: string | null
  onPick: (templateId: string | null) => void
}

function TemplateGallery({ templates, loading, chosen, onPick }: TemplateGalleryProps) {
  if (loading) return <Hint>Loading templates…</Hint>
  return (
    <ul aria-label="Templates" className="grid grid-cols-2 gap-3 lg:grid-cols-3">
      <TemplateCard name={AUTO_PICK} thumbnail={null} kind="Auto" chosen={chosen === null} onPick={() => onPick(null)} />
      {templates.map((t) => (
        <TemplateCard key={t.id} name={t.name} thumbnail={t.thumbnail_url} kind={t.kind} chosen={chosen === t.id} onPick={() => onPick(t.id)} />
      ))}
    </ul>
  )
}

interface EditorLookCardProps extends TemplateGalleryProps {
  busy: boolean
  onUpload: (file: File) => void
  onLibrary: (item: DeliverableSummary) => void
}

export function EditorLookCard({ templates, loading, chosen, onPick, busy, onUpload, onLibrary }: EditorLookCardProps) {
  const [source, setSource] = useState<LookSource>('template')
  const library = useSocialLibrary(source === 'library')
  return (
    <EditorCard label="Look">
      <Segmented label="Where the visual comes from" choices={LOOK_SOURCES} value={source} onChange={setSource} />
      {source === 'template' && <TemplateGallery templates={templates} loading={loading} chosen={chosen} onPick={onPick} />}
      {source === 'upload' && <UploadDrop busy={busy} onFile={onUpload} />}
      {source === 'library' && <LibraryGrid items={library.data} loading={library.isLoading} busy={busy} onPick={onLibrary} />}
    </EditorCard>
  )
}
