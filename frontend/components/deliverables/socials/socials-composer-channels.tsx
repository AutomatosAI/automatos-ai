'use client'

/**
 * PRD-251 S2.2c (US-209) — the composer's formats and channels: the aspect ratios
 * the template renders in, and each connected channel with the post kind it takes
 * (GET /api/socials/channels). Saving writes the post's targets
 * (PUT /api/socials/posts/{id}/targets), which voids an approval if it had one.
 */
import type { SocialChannel, SocialPostKind, SocialPostTargetOptions } from '@/lib/api-client'
import { ChannelRow, DEFAULT_YOUTUBE_OPTIONS, YOUTUBE_TOOLKIT } from './socials-channel-row'
import { aspectLabel, withChannel, type ComposerDraft } from './socials-composer-model'

interface SocialsComposerChannelsProps {
  draft: ComposerDraft
  channels: ReadonlyArray<SocialChannel>
  onChange: (draft: ComposerDraft) => void
}

export function SocialsComposerChannels({ draft, channels, onChange }: SocialsComposerChannelsProps) {
  const sizes = draft.template?.sizes ?? []
  const toggle = (channel: SocialChannel, on: boolean) => {
    const next = withChannel(draft, channel, on)
    const wantsDefaults = on && channel.toolkit === YOUTUBE_TOOLKIT && !draft.options[channel.toolkit]
    onChange(wantsDefaults ? { ...next, options: { ...next.options, [channel.toolkit]: DEFAULT_YOUTUBE_OPTIONS } } : next)
  }
  const setKind = (toolkit: string, kind: SocialPostKind) => onChange({ ...draft, kinds: { ...draft.kinds, [toolkit]: kind } })
  const setOptions = (toolkit: string, options: SocialPostTargetOptions) =>
    onChange({ ...draft, options: { ...draft.options, [toolkit]: options } })

  return (
    <div className="space-y-4">
      <section aria-label="Formats" className="space-y-1">
        <h4 className="text-sm font-medium text-foreground">Formats</h4>
        {sizes.length > 0 ? (
          <p className="text-sm text-muted-foreground">Renders at {sizes.map(aspectLabel).join(', ')}.</p>
        ) : (
          <p className="text-sm text-muted-foreground">Choose a template to set the formats this post renders in.</p>
        )}
      </section>
      <section aria-label="Channels" className="space-y-2">
        <h4 className="text-sm font-medium text-foreground">Channels</h4>
        {channels.length === 0 ? (
          <p className="text-sm text-muted-foreground">No social channel is connected yet: connect one in Composio.</p>
        ) : (
          <ul className="space-y-2">
            {channels.map((channel) => (
              <ChannelRow
                key={channel.toolkit}
                channel={channel}
                kind={draft.kinds[channel.toolkit] ?? null}
                options={draft.options[channel.toolkit] ?? {}}
                onToggle={(on) => toggle(channel, on)}
                onKind={(kind) => setKind(channel.toolkit, kind)}
                onOptions={(options) => setOptions(channel.toolkit, options)}
              />
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
