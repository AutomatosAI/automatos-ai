'use client'

/**
 * PRD-251B US-B110 — the editor's Preview (Editor.dc.html's sticky card): a tab per ticked
 * channel; the post as it will look there (the brand's name, the channel and the slot,
 * the channel's copy, the media at the channel's aspect through the shared FilePreview, a
 * video's length); "Preview out of date" after an edit to what a render reads; the note on
 * what was rendered or what the full render will cost; and the channel's own copy with its
 * count against the channel's limit (red over it, and Submit waits).
 */
import { useState } from 'react'

import { FilePreview, inferPreviewType } from '@/components/widgets/FileWidget/FilePreview'
import type { SocialChannel, SocialPost } from '@/lib/api-client'
import { slotDateLabel } from '@/lib/social-time'
import { useBrandName } from '@/hooks/use-socials-editor'
import { ChannelCopyField } from '../socials-channel-copy'
import { lengthLabel } from './socials-calendar-model'
import { aspectOf, renderRatios, slotInput, type EditorDraft } from './editor-model'
import { CARD, HINT, Hint, Segmented } from './editor-ui'
import { PREVIEW_STALE, fileForAspect, previewStale, renderNote, renderedAt, type PreviewFile } from './preview-model'
import { usePreviewMedia } from './use-preview-media'

export const CHANNEL_COPY_HINT = 'Each channel starts from the same base copy. An edit here changes this channel only.'

interface FrameProps {
  brand: string
  channel: SocialChannel
  slot: string
  copy: string
  file: PreviewFile | null
  aspect: string
  length: string | null
  stale: boolean
  textOnly: boolean
}

function PostFrame({ brand, channel, slot, copy, file, aspect, length, stale, textOnly }: FrameProps) {
  return (
    <div className="flex flex-col gap-3 rounded-[14px] border border-border bg-[#121110] p-4">
      <div className="flex items-center gap-2.5">
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-accent font-bold text-accent-foreground" aria-hidden>
          {brand.slice(0, 1).toUpperCase()}
        </span>
        <span className="flex flex-col">
          <span className="font-semibold text-foreground">{brand}</span>
          <span className={HINT}>{channel.label} · {slot}</span>
        </span>
      </div>
      <p className="m-0 whitespace-pre-line leading-[1.55] text-foreground">{copy}</p>
      {!textOnly && (
        <div
          data-testid="socials-preview-frame"
          className="relative mx-auto w-full max-w-[360px] overflow-hidden rounded-lg bg-[#141210]"
          style={{ aspectRatio: aspect.replace(':', ' / ') }}
        >
          {file ? (
            <FilePreview url={file.url} filename={file.name} previewType={inferPreviewType(file.name, file.contentType)} />
          ) : (
            <span className="absolute inset-0 flex items-center justify-center font-mono text-[11px] uppercase tracking-[.12em] text-accent">
              Not rendered yet
            </span>
          )}
          {length && <span className="absolute bottom-2 right-2 font-mono text-[11px] text-foreground">{length}</span>}
          {stale && file && (
            <span role="status" className="absolute inset-x-0 top-0 bg-[hsl(var(--warning)/0.9)] px-2 py-1 text-xs font-medium text-accent-foreground">
              {PREVIEW_STALE}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

interface SocialsPreviewProps {
  post: SocialPost | null
  draft: EditorDraft
  channels: ReadonlyArray<SocialChannel>
  /** The chosen template's sizes: how many files the full render makes. */
  templateSizes: ReadonlyArray<string>
  onChannelCopy: (toolkit: string, text: string) => void
}

export function SocialsPreview({ post, draft, channels, templateSizes, onChannelCopy }: SocialsPreviewProps) {
  const ticked = channels.filter((channel) => draft.kinds[channel.toolkit])
  const [tab, setTab] = useState<string | null>(null)
  const active = ticked.find((channel) => channel.toolkit === tab) ?? ticked[0] ?? null
  const brand = useBrandName()
  const files = usePreviewMedia(post, draft.format === 'video')
  const textOnly = draft.format === 'text'
  const slot = slotInput(draft.slot)
  const note = renderNote({
    format: draft.format, lengthSeconds: draft.lengthSeconds, ratios: renderRatios(draft, templateSizes), sizes: templateSizes.length, renderedAt: renderedAt(post),
  })
  const copyOf = (toolkit: string) => draft.perChannel[toolkit] ?? draft.base
  return (
    <section aria-label="Preview" className={CARD}>
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        <h2 className="text-[15px] font-semibold text-foreground">Preview</h2>
        {ticked.length > 0 && (
          <Segmented label="Preview channel" choices={ticked.map((c) => ({ value: c.toolkit, label: c.label }))} value={active?.toolkit ?? null} onChange={setTab} />
        )}
      </div>
      {active ? (
        <>
          <PostFrame
            brand={brand} channel={active} copy={copyOf(active.toolkit)} textOnly={textOnly} stale={previewStale(post, draft)}
            slot={slot.plannedFor ? slotDateLabel(slot.plannedFor, slot.timezone) : 'No slot yet'}
            aspect={aspectOf(active.toolkit, draft.kinds[active.toolkit]) ?? '1:1'}
            file={fileForAspect(files, aspectOf(active.toolkit, draft.kinds[active.toolkit]))}
            length={draft.format === 'video' && draft.lengthSeconds ? lengthLabel(draft.lengthSeconds) : null}
          />
          <Hint>{note}</Hint>
          <ChannelCopyField
            toolkit={active.toolkit} label={active.label} value={copyOf(active.toolkit)} limits={active.copy_limits}
            onChange={(text) => onChannelCopy(active.toolkit, text)}
          />
          <Hint>{CHANNEL_COPY_HINT}</Hint>
        </>
      ) : (
        <Hint>Tick a channel to see the post as it will look there.</Hint>
      )}
    </section>
  )
}
