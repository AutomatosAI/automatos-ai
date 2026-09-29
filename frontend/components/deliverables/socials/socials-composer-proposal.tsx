'use client'

/**
 * PRD-251 S2.2a (US-207) — the proposal fills the composer: its title, the base
 * copy and each channel's own text, all editable, with what the server had to
 * correct listed as warnings. US-209: each chosen channel's text counts against
 * its limits.
 */
import { AlertTriangle } from 'lucide-react'

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialChannel } from '@/lib/api-client'
import { SOCIAL_POST_TITLE_MAX_CHARS } from './socials-status'
import type { ComposerDraft } from './socials-composer-model'
import { ChannelCopyField } from './socials-channel-copy'

export function ComposerWarnings({ warnings }: { warnings: ReadonlyArray<string> }) {
  if (warnings.length === 0) return null
  return (
    <ul aria-label="Warnings" className="space-y-1 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2">
      {warnings.map((warning) => (
        <li key={warning} className="flex gap-2 text-sm text-foreground">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
          {warning}
        </li>
      ))}
    </ul>
  )
}

interface SocialsComposerProposalProps {
  draft: ComposerDraft
  channels: ReadonlyArray<SocialChannel>
  onChange: (draft: ComposerDraft) => void
}

export function SocialsComposerProposal({ draft, channels, onChange }: SocialsComposerProposalProps) {
  const channelOf = (toolkit: string) => channels.find((c) => c.toolkit === toolkit)
  const labelOf = (toolkit: string) => channelOf(toolkit)?.label ?? toolkit
  const setChannelText = (toolkit: string, text: string) =>
    onChange({ ...draft, perChannel: { ...draft.perChannel, [toolkit]: text } })

  return (
    <div className="space-y-3">
      <ComposerWarnings warnings={draft.warnings} />
      <div className="space-y-1.5">
        <Label htmlFor="socials-composer-title">Title</Label>
        <Input
          id="socials-composer-title"
          value={draft.title}
          maxLength={SOCIAL_POST_TITLE_MAX_CHARS}
          onChange={(event) => onChange({ ...draft, title: event.target.value })}
        />
      </div>
      <div className="space-y-1.5">
        <Label htmlFor="socials-composer-base">Base copy</Label>
        <Textarea
          id="socials-composer-base"
          value={draft.base}
          rows={4}
          onChange={(event) => onChange({ ...draft, base: event.target.value })}
        />
      </div>
      {Object.keys(draft.kinds).map((toolkit) => (
        <ChannelCopyField
          key={toolkit}
          toolkit={toolkit}
          label={labelOf(toolkit)}
          value={draft.perChannel[toolkit] ?? draft.base}
          limits={channelOf(toolkit)?.copy_limits}
          onChange={(text) => setChannelText(toolkit, text)}
        />
      ))}
    </div>
  )
}
