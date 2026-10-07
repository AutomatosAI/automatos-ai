'use client'

/**
 * PRD-251B US-B109 — the editor's Look (Editor.dc.html): where the visual comes from.
 * Template: the format's templates (GET /api/socials/templates), "Let Auto pick" first
 * (no template), each with its thumbnail (a format placeholder until the backfill made
 * one) and name. Upload: a file of the person's own. Library: one of the workspace's
 * image and video Deliverables. AI-made (US-B305): AI images for the template's image slots
 * (editor-look-ai.tsx); AI footage for the hook and b-roll is Format's switch.
 * Each source says in one line what it does; a template that shows a picture is marked
 * Photo, and "Let Auto pick" says what Auto does with it. F377: a template made for a software
 * product says so under its name.
 */
import { useState } from 'react'

import { cn } from '@/lib/utils'
import type { DeliverableSummary, SocialFootageKind, SocialPost, SocialTemplateSummary } from '@/lib/api-client'
import { useSocialLibrary } from '@/hooks/use-socials-editor'
import { EditorLookAi } from './editor-look-ai'
import { LibraryGrid, UploadDrop } from './editor-look-sources'
import { PhotoTargetPicker, WHOLE_POST, photoTarget, type PhotoSpot } from './editor-look-photo'
import { EditorCard, Hint, Segmented } from './editor-ui'

export type LookSource = 'template' | 'upload' | 'library' | 'ai'
export const LOOK_SOURCES: ReadonlyArray<{ value: LookSource; label: string }> = [
  { value: 'template', label: 'Template' },
  { value: 'upload', label: 'Upload' },
  { value: 'library', label: 'Library' },
  { value: 'ai', label: 'AI-made' },
]
/** What each source does, in one line under the switch. */
export const LOOK_HINTS: Record<LookSource, string> = {
  template: 'Your words set in a designed card. A template marked Photo also shows a picture behind them.',
  upload: 'Your own photo or video. With a Photo template it goes behind the words; otherwise it is the whole post, cropped for each channel.',
  library: 'A picture or video you already have in Deliverables, used just like an upload.',
  ai: "Pictures made by AI for a Photo template, in your brand kit's style.",
}
export const AUTO_PICK = 'Let Auto pick'
export const AUTO_PICK_NOTE = 'Auto picks a template and writes its words from your brief when you render.'
export const PHOTO_BADGE = 'Photo'
/** F377: under a template made for a software product (`made_for`), whom it is for. */
export const SOFTWARE_NOTE = "For a software product: it shows the product's own screens."

/** The line under a template's name: whom it is for, when it is not any business. */
export function templateNote(template: Pick<SocialTemplateSummary, 'made_for'>): string | undefined {
  return template.made_for === 'software' ? SOFTWARE_NOTE : undefined
}

interface TemplateCardProps {
  name: string
  thumbnail: string | null
  kind: string
  chosen: boolean
  onPick: () => void
  /** The template shows a picture (it has an image slot). */
  photo?: boolean
  /** A line under the name. */
  note?: string
}

function TemplateCard({ name, thumbnail, kind, chosen, onPick, photo = false, note }: TemplateCardProps) {
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
        <span className="relative block">
          {thumbnail ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={thumbnail} alt="" className="aspect-[4/3] w-full rounded-lg object-cover" />
          ) : (
            <span className="flex aspect-[4/3] w-full items-end rounded-lg bg-[#141210] p-2.5 font-serif text-base leading-[1.1] text-foreground">
              {kind}
            </span>
          )}
          {photo && (
            <span className="absolute left-2 top-2 rounded-full bg-background/85 px-2 py-0.5 text-[11px] font-medium text-foreground">
              {PHOTO_BADGE}
            </span>
          )}
        </span>
        <span className="text-[13px] font-medium text-foreground">{name}</span>
        {note && <span className="text-[12px] leading-snug text-muted-foreground">{note}</span>}
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
      <TemplateCard name={AUTO_PICK} thumbnail={null} kind="Auto" chosen={chosen === null} onPick={() => onPick(null)} note={AUTO_PICK_NOTE} />
      {templates.map((t) => (
        <TemplateCard
          key={t.id} name={t.name} thumbnail={t.thumbnail_url} kind={t.kind} chosen={chosen === t.id} onPick={() => onPick(t.id)}
          photo={(t.image_slots ?? []).length > 0} note={templateNote(t)}
        />
      ))}
    </ul>
  )
}

/** PRD-251B US-B305: what the AI-made source needs. */
export interface LookAiProps {
  post: SocialPost | null
  imageSlots: ReadonlyArray<string>
  images: SocialFootageKind | undefined
  aiBusy: boolean
  onAiMake: (slot: string, prompt: string) => void
  onAiPick: (slot: string, name: string) => void
}

interface EditorLookCardProps extends TemplateGalleryProps {
  busy: boolean
  /** The chosen template's photo spots (PRD-251B): an upload or a Library pick fills one. */
  photoSpots: ReadonlyArray<PhotoSpot>
  /** `spot`: the photo spot (the template's slot) the file fills; none, the whole post. */
  onUpload: (file: File, spot?: string) => void
  onLibrary: (item: DeliverableSummary, spot?: string) => void
  ai: LookAiProps
}

export function EditorLookCard({ templates, loading, chosen, onPick, busy, photoSpots, onUpload, onLibrary, ai }: EditorLookCardProps) {
  const [source, setSource] = useState<LookSource>('template')
  const [asked, setAsked] = useState<string | null>(null)
  const library = useSocialLibrary(source === 'library')
  const target = photoTarget(photoSpots, asked)
  const spot = target === WHOLE_POST ? undefined : target
  const items = spot ? library.data?.filter((item) => item.artifact_type === 'image') : library.data
  return (
    <EditorCard label="Look">
      <Segmented label="Where the visual comes from" choices={LOOK_SOURCES} value={source} onChange={setSource} />
      <Hint>{LOOK_HINTS[source]}</Hint>
      {source === 'template' && <TemplateGallery templates={templates} loading={loading} chosen={chosen} onPick={onPick} />}
      {(source === 'upload' || source === 'library') && (
        <PhotoTargetPicker spots={photoSpots} target={target} post={ai.post} onTarget={setAsked} />
      )}
      {source === 'upload' && <UploadDrop busy={busy} pictureOnly={!!spot} onFile={(file) => onUpload(file, spot)} />}
      {source === 'library' && <LibraryGrid items={items} loading={library.isLoading} busy={busy} onPick={(item) => onLibrary(item, spot)} />}
      {source === 'ai' && (
        <EditorLookAi post={ai.post} imageSlots={ai.imageSlots} images={ai.images} busy={ai.aiBusy} onMake={ai.onAiMake} onPick={ai.onAiPick} />
      )}
    </EditorCard>
  )
}
