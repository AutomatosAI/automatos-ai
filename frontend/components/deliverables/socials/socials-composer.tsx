'use client'

/**
 * PRD-251 S2.2 (US-207..US-209) — the composer, opened by "New post".
 *
 * 1. The brief and the channels → "Draft it": the server's proposal (not saved).
 * 2. Copy: the proposal's title, base copy and each channel's text.
 * 3. Variables and preview (US-208): the template's fields and a render. A render
 *    saves the draft first; a video previews at half resolution, an image renders
 *    for real. An edit after it marks the preview stale.
 *
 * "Save draft" creates the post (or updates the one a render saved), then sets
 * its channels. The bare form stays reachable as "Blank draft" (SocialsNewDraft).
 */
import { useMemo, useState } from 'react'
import { Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { SocialPost } from '@/lib/api-client'
import {
  useComposeSocialPost,
  useComposerPost,
  usePreviewComposedDraft,
  useSaveComposedDraft,
  useSocialChannels,
} from '@/hooks/use-socials-composer'
import { SocialsComposerBrief } from './socials-composer-brief'
import { ComposerStepBody, ComposerStepNav, type ComposerStep } from './socials-composer-steps'
import { draftFromProposal, draftTargets, isVideoDraft, postInput, type ComposerDraft } from './socials-composer-model'

interface SocialsComposerProps {
  /** Called with the saved post, or null when the composer is closed. */
  onDone: (post: SocialPost | null) => void
}

/** What a render depends on: an edit to any of it makes the preview stale. */
function renderKey(draft: ComposerDraft): string {
  return JSON.stringify([draft.title, draft.base, draft.perChannel, draft.variables, draft.sources, draft.kinds])
}

const SHELL = 'socials-composer space-y-4 rounded-xl border border-border bg-card/40 p-4'

export function SocialsComposer({ onDone }: SocialsComposerProps) {
  const { data, isLoading: channelsLoading } = useSocialChannels()
  const channels = useMemo(() => data ?? [], [data])
  const compose = useComposeSocialPost()
  const save = useSaveComposedDraft()
  const preview = usePreviewComposedDraft()
  const [draft, setDraft] = useState<ComposerDraft | null>(null)
  const [lastBrief, setLastBrief] = useState('')
  const [step, setStep] = useState<ComposerStep>('copy')
  const [postId, setPostId] = useState<string | null>(null)
  const [renderedKey, setRenderedKey] = useState<string | null>(null)
  const { data: post } = useComposerPost(postId)

  const draftIt = (brief: string, chosen: string[]) => {
    setLastBrief(brief)
    compose.mutate(
      { brief, channels: chosen },
      { onSuccess: (proposal) => setDraft(draftFromProposal(brief, proposal, channels)) },
    )
  }

  if (draft === null) {
    return (
      <section aria-label="Composer" className={SHELL}>
        <h3 className="text-sm font-semibold text-foreground">New post</h3>
        <SocialsComposerBrief
          initialBrief={lastBrief}
          channels={channels}
          channelsLoading={channelsLoading}
          busy={compose.isLoading}
          onDraft={draftIt}
          onCancel={() => onDone(null)}
        />
      </section>
    )
  }

  const saving = { post: postInput(draft), targets: draftTargets(draft), postId }
  const saveDraft = () => {
    if (draft.title.trim()) save.mutate(saving, { onSuccess: (saved) => onDone(saved) })
  }
  const renderDraft = () =>
    preview.mutate(
      { ...saving, video: isVideoDraft(draft) },
      {
        onSuccess: (saved) => {
          setPostId(saved.id)
          setRenderedKey(renderKey(draft))
        },
      },
    )
  const stale = renderedKey !== null && renderedKey !== renderKey(draft)

  return (
    <section aria-label="Composer" className={SHELL}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-foreground">New post</h3>
        <ComposerStepNav step={step} onStep={setStep} />
      </div>
      <ComposerStepBody
        step={step} draft={draft} channels={channels} post={post} stale={stale}
        rendering={preview.isLoading} onChange={setDraft} onRender={renderDraft}
      />
      <div className="flex flex-wrap justify-end gap-2">
        <Button type="button" variant="ghost" size="sm" onClick={() => onDone(null)}>
          Cancel
        </Button>
        {!postId && (
          <Button type="button" variant="outline" size="sm" onClick={() => setDraft(null)} disabled={save.isLoading}>
            Back to the brief
          </Button>
        )}
        {step === 'copy' && (
          <Button type="button" variant="outline" size="sm" onClick={() => setStep('variables')}>
            Next: variables
          </Button>
        )}
        <Button type="button" size="sm" onClick={saveDraft} disabled={!draft.title.trim() || save.isLoading}>
          {save.isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />}
          Save draft
        </Button>
      </div>
    </section>
  )
}
