'use client'

/**
 * PRD-251B US-B109 — the editor's Channels and sizes: one row per connected channel
 * (GET /api/socials/channels) with its badge, its name and the size the format gives on
 * its post kind; a channel the format excludes is disabled with why. "Renders" lists the
 * ratios of the ticked channels. YouTube's privacy and category show once it is ticked.
 * PRD-251C (US-C301): a ticked channel that posts stories offers "As a story" for an image
 * or a video, rendered 9:16 inside the story's safe zone.
 * PRD-251D (E5, US-D105): under the rows, each supported channel that is not connected has its
 * Connect button (editor-connect-channels.tsx); with none connected, the buttons are the card.
 */
import { cn } from '@/lib/utils'
import type { SocialChannel, SocialPostTargetOptions } from '@/lib/api-client'
import { YOUTUBE_TOOLKIT, YoutubeOptions } from '../socials-channel-row'
import { CHANNEL_BADGE } from './socials-calendar-chip'
import { channelBadge } from './socials-calendar-model'
import { EditorConnectChannels } from './editor-connect-channels'
import { canStory, channelBlock, renderRatios, sizeFor, type EditorDraft } from './editor-model'
import { EditorCard, Hint } from './editor-ui'

interface ChannelLineProps {
  channel: SocialChannel
  draft: EditorDraft
  templateSizes: ReadonlyArray<string>
  onTick: (channel: SocialChannel, on: boolean) => void
  onStory: (channel: SocialChannel, on: boolean) => void
  onOptions: (toolkit: string, options: SocialPostTargetOptions) => void
}

function ChannelLine({ channel, draft, templateSizes, onTick, onStory, onOptions }: ChannelLineProps) {
  const blocked = channelBlock(channel, draft.format)
  const kind = draft.kinds[channel.toolkit] ?? null
  const id = `socials-editor-channel-${channel.toolkit}`
  return (
    <li aria-label={channel.label} className={cn('flex flex-col gap-2 rounded-[10px] border border-border px-3 py-1.5', blocked && 'opacity-60')}>
      <label htmlFor={id} className="flex min-h-[46px] items-center gap-3">
        <input id={id} type="checkbox" checked={kind !== null} disabled={!!blocked} onChange={(e) => onTick(channel, e.target.checked)} />
        <span className={CHANNEL_BADGE}>{channelBadge(channel.toolkit)}</span>
        <span className="flex-grow font-medium text-foreground">{channel.label}</span>
        <span className="font-mono text-xs text-muted-foreground">
          {blocked ?? (kind ? sizeFor(channel.toolkit, kind, templateSizes) : '')}
        </span>
      </label>
      {kind !== null && canStory(channel, draft.format) && (
        <label className="flex items-center gap-2 pb-1 pl-7 text-[13px] text-muted-foreground">
          <input type="checkbox" checked={kind === 'story'} onChange={(e) => onStory(channel, e.target.checked)} />
          As a story
        </label>
      )}
      {kind !== null && channel.toolkit === YOUTUBE_TOOLKIT && (
        <YoutubeOptions options={draft.options[channel.toolkit] ?? {}} onChange={(o) => onOptions(channel.toolkit, o)} />
      )}
    </li>
  )
}

interface EditorChannelsCardProps extends Omit<ChannelLineProps, 'channel'> {
  channels: ReadonlyArray<SocialChannel>
  loading: boolean
  /** Reads the connected channels again, once a connect window reports back. */
  onRefetch: () => unknown
}

export function EditorChannelsCard({ channels, loading, draft, onRefetch, ...line }: EditorChannelsCardProps) {
  const ratios = renderRatios(draft, line.templateSizes)
  const renders = draft.format === 'text' ? 'Text only' : ratios.length > 0 ? `Renders ${ratios.join(' · ')}` : 'Pick a channel'
  return (
    <EditorCard label="Channels and sizes" action={<span className="text-[12.5px] text-muted-foreground">{renders}</span>}>
      {loading ? (
        <Hint>Loading channels…</Hint>
      ) : (
        <>
          {channels.length > 0 && (
            <ul aria-label="Channels" className="flex flex-col gap-2">
              {channels.map((channel) => (
                <ChannelLine key={channel.toolkit} channel={channel} draft={draft} {...line} />
              ))}
            </ul>
          )}
          <EditorConnectChannels channels={channels} refetch={onRefetch} />
        </>
      )}
    </EditorCard>
  )
}
