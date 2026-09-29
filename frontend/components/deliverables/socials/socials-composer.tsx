'use client'

/**
 * PRD-251 S2.2 (US-207..US-209) — the composer, opened by "New post".
 *
 * 1. The brief and the channels → "Draft it": the server's proposal (not saved).
 * 2. The proposal fills the composer: title, base copy and each channel's text.
 * 3. "Save draft" creates the post, then sets its channels (its targets).
 *
 * The bare form stays reachable as "Blank draft" (SocialsNewDraft).
 */
import { useMemo, useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { SocialPost } from '@/lib/api-client'
import { useComposeSocialPost, useSaveComposedDraft, useSocialChannels } from '@/hooks/use-socials-composer'
import { SocialsComposerBrief } from './socials-composer-brief'
import { SocialsComposerProposal } from './socials-composer-proposal'
import { draftFromProposal, draftTargets, postInput, type ComposerDraft } from './socials-composer-model'

interface SocialsComposerProps {
  /** Called with the saved post, or null when the composer is closed. */
  onDone: (post: SocialPost | null) => void
}

export function SocialsComposer({ onDone }: SocialsComposerProps) {
  const { data, isLoading: channelsLoading } = useSocialChannels()
  const channels = useMemo(() => data ?? [], [data])
  const compose = useComposeSocialPost()
  const save = useSaveComposedDraft()
  const [draft, setDraft] = useState<ComposerDraft | null>(null)
  const [lastBrief, setLastBrief] = useState('')

  const draftIt = (brief: string, chosen: string[]) => {
    setLastBrief(brief)
    compose.mutate(
      { brief, channels: chosen },
      { onSuccess: (proposal) => setDraft(draftFromProposal(brief, proposal, channels)) },
    )
  }

  const saveDraft = () => {
    if (!draft || !draft.title.trim()) return
    save.mutate({ post: postInput(draft), targets: draftTargets(draft) }, { onSuccess: (post) => onDone(post) })
  }

  return (
    <section aria-label="Composer" className="socials-composer space-y-4 rounded-xl border border-border bg-card/40 p-4">
      <h3 className="text-sm font-semibold text-foreground">New post</h3>
      {draft === null ? (
        <SocialsComposerBrief
          initialBrief={lastBrief}
          channels={channels}
          channelsLoading={channelsLoading}
          busy={compose.isLoading}
          onDraft={draftIt}
          onCancel={() => onDone(null)}
        />
      ) : (
        <>
          <SocialsComposerProposal draft={draft} channels={channels} onChange={setDraft} />
          <div className="flex flex-wrap justify-end gap-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => onDone(null)}>
              Cancel
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={() => setDraft(null)} disabled={save.isLoading}>
              Back to the brief
            </Button>
            <Button type="button" size="sm" onClick={saveDraft} disabled={!draft.title.trim() || save.isLoading}>
              {save.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
              Save draft
            </Button>
          </div>
        </>
      )}
    </section>
  )
}
