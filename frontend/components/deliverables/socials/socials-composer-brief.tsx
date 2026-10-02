'use client'

/**
 * PRD-251 S2.2a (US-207) — the composer's first step: a brief, and the connected
 * channels to write for as chips (all chosen to start). "Draft it" asks the
 * server for a proposal; nothing is saved yet.
 */
import { useEffect, useState, type FormEvent } from 'react'
import { Loader2, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import type { SocialChannel } from '@/lib/api-client'

// The server's limit (api/socials_compose.py BRIEF_MAX_CHARS).
export const SOCIAL_BRIEF_MAX_CHARS = 4000

interface ChannelChipsProps {
  channels: ReadonlyArray<SocialChannel>
  chosen: ReadonlyArray<string>
  onToggle: (toolkit: string) => void
}

export function ChannelChips({ channels, chosen, onToggle }: ChannelChipsProps) {
  return (
    <div role="group" aria-label="Channels" className="flex flex-wrap gap-2">
      {channels.map((channel) => {
        const on = chosen.includes(channel.toolkit)
        return (
          <button
            key={channel.toolkit}
            type="button"
            aria-pressed={on}
            onClick={() => onToggle(channel.toolkit)}
            className={cn(
              'rounded-full border px-3 py-1 text-sm transition-colors',
              on ? 'border-primary bg-primary/10 text-foreground' : 'border-border text-muted-foreground hover:text-foreground',
            )}
          >
            {channel.label}
          </button>
        )
      })}
    </div>
  )
}

interface SocialsComposerBriefProps {
  /** The brief to start from: the last one, when the person comes back to it. */
  initialBrief?: string
  channels: ReadonlyArray<SocialChannel>
  channelsLoading: boolean
  busy: boolean
  onDraft: (brief: string, channels: string[]) => void
  onCancel: () => void
}

export function SocialsComposerBrief({
  initialBrief = '',
  channels,
  channelsLoading,
  busy,
  onDraft,
  onCancel,
}: SocialsComposerBriefProps) {
  const [brief, setBrief] = useState(initialBrief)
  const [chosen, setChosen] = useState<string[]>([])

  // Every connected channel is chosen to start with.
  useEffect(() => {
    setChosen(channels.map((channel) => channel.toolkit))
  }, [channels])

  const toggle = (toolkit: string) =>
    setChosen((now) => (now.includes(toolkit) ? now.filter((t) => t !== toolkit) : [...now, toolkit]))

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!brief.trim() || busy) return
    onDraft(brief.trim(), chosen)
  }

  return (
    <form onSubmit={submit} aria-label="Write the brief" className="space-y-3">
      <div className="space-y-1.5">
        <Label htmlFor="socials-composer-brief">Brief</Label>
        <Textarea
          id="socials-composer-brief"
          value={brief}
          rows={4}
          maxLength={SOCIAL_BRIEF_MAX_CHARS}
          onChange={(event) => setBrief(event.target.value)}
          placeholder="What the post should say, for whom, and any figures it should use"
        />
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">Channels</p>
        {channelsLoading ? (
          <p className="text-sm text-muted-foreground">Loading channels…</p>
        ) : channels.length === 0 ? (
          <p className="text-sm text-muted-foreground">No social channel is connected yet: connect one in Composio.</p>
        ) : (
          <ChannelChips channels={channels} chosen={chosen} onToggle={toggle} />
        )}
      </div>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={!brief.trim() || busy}>
          {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden /> : <Sparkles className="mr-2 h-4 w-4" aria-hidden />}
          Draft it
        </Button>
      </div>
    </form>
  )
}
