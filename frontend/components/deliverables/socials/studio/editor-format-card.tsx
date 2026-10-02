'use client'

/**
 * PRD-251B US-B109 — the editor's Format: Image · Carousel · Video · Text only. A video
 * chooses its length among the ones its template declares (B5), with how many spoken
 * words fit, its voice (the post's voice picker, once the post is saved), and whether
 * AI footage fills the template's hook and b-roll slots; a carousel its number of slides;
 * a text post says which channels take one.
 */
import { Minus, Plus } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { SocialFootageKind, SocialPost } from '@/lib/api-client'
import { SocialsVoicePicker } from '../socials-voice-picker'
import { lengthLabel } from './socials-calendar-model'
import { EDITOR_FORMATS, MAX_SLIDES, MIN_SLIDES, wordsHint, type EditorDraft, type EditorFormat } from './editor-model'
import { ChipGroup, EditorCard, Hint, Segmented } from './editor-ui'

export const TEXT_ONLY_NOTE = 'No media. Only X and LinkedIn take a text-only post.'
export const MUSIC_NOTE = "Music: the template's own track, with its credit line added to the copy."
export const FOOTAGE_NOTE =
  'Product screens are rendered from the app and every word on screen is template text. AI footage only fills the hook and the b-roll.'
const LABEL = 'text-sm font-medium text-foreground'

interface VideoOptionsProps {
  draft: EditorDraft
  durations: ReadonlyArray<number>
  footageSlots: ReadonlyArray<string>
  footage: SocialFootageKind | undefined
  post: SocialPost | null
  canEdit: boolean
  onChange: (draft: EditorDraft) => void
}

function VideoOptions({ draft, durations, footageSlots, footage, post, canEdit, onChange }: VideoOptionsProps) {
  const footageReady = !!footage?.available && footageSlots.length > 0
  return (
    <div className="flex flex-col gap-3.5">
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>Length</span>
        {durations.length > 0 ? (
          <ChipGroup
            label="Length"
            choices={durations.map((seconds) => ({ value: seconds, label: lengthLabel(seconds) }))}
            value={draft.lengthSeconds}
            onChange={(seconds) => onChange({ ...draft, lengthSeconds: seconds })}
          />
        ) : (
          <Hint>Pick a template to choose a length.</Hint>
        )}
        {draft.lengthSeconds && <Hint>{wordsHint(draft.lengthSeconds)}</Hint>}
      </div>
      {post ? <SocialsVoicePicker post={post} editable={canEdit} /> : <Hint>Save the draft to choose its voice.</Hint>}
      <Hint>{MUSIC_NOTE}</Hint>
      <div className="flex flex-col gap-1.5">
        <span className={LABEL}>AI footage for the hook and b-roll</span>
        <Segmented
          label="AI footage"
          choices={[
            { value: 'off', label: 'Off' },
            { value: 'on', label: footage?.label ?? 'No footage toolkit connected', disabled: !footageReady },
          ]}
          value={draft.footageOn && footageReady ? 'on' : 'off'}
          onChange={(value) => onChange({ ...draft, footageOn: value === 'on' })}
        />
        <Hint>{footageReady ? FOOTAGE_NOTE : footage?.reason ?? 'This template has no slot for AI footage.'}</Hint>
      </div>
    </div>
  )
}

function SlidesStepper({ slides, onSlides }: { slides: number; onSlides: (slides: number) => void }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className={LABEL}>Slides</span>
      <div className="flex items-center gap-2">
        <Button type="button" size="sm" variant="secondary" aria-label="One slide fewer" disabled={slides <= MIN_SLIDES} onClick={() => onSlides(slides - 1)}>
          <Minus className="h-4 w-4" aria-hidden />
        </Button>
        <span className="min-w-[32px] text-center font-mono text-base" aria-live="polite">{slides}</span>
        <Button type="button" size="sm" variant="secondary" aria-label="One slide more" disabled={slides >= MAX_SLIDES} onClick={() => onSlides(slides + 1)}>
          <Plus className="h-4 w-4" aria-hidden />
        </Button>
      </div>
      <Hint>One point per slide; the last slide is the call to action.</Hint>
    </div>
  )
}

interface EditorFormatCardProps extends VideoOptionsProps {
  slides: number
  onFormat: (format: string) => void
  onSlides: (slides: number) => void
}

export function EditorFormatCard({ slides, onFormat, onSlides, ...video }: EditorFormatCardProps) {
  const format = video.draft.format
  return (
    <EditorCard label="Format">
      <Segmented label="Format" choices={EDITOR_FORMATS} value={format as EditorFormat} onChange={onFormat} />
      {format === 'video' && <VideoOptions {...video} />}
      {format === 'carousel' && <SlidesStepper slides={slides} onSlides={onSlides} />}
      {format === 'text' && <Hint>{TEXT_ONLY_NOTE}</Hint>}
    </EditorCard>
  )
}
