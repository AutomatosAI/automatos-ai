'use client'

/**
 * PRD-251 S2.2 (US-207..US-209) — the composer's steps after the brief, and what
 * each shows. Variables and preview (US-208) sit side by side from 1024 px and
 * stack below it; formats and channels (US-209) come last.
 */
import { cn } from '@/lib/utils'
import type { SocialChannel, SocialPost } from '@/lib/api-client'
import { SocialsComposerChannels } from './socials-composer-channels'
import { SocialsComposerProposal } from './socials-composer-proposal'
import { SocialsComposerPreview } from './socials-composer-preview'
import { SocialsVariablesForm } from './socials-variables-form'
import { isVideoDraft, type ComposerDraft } from './socials-composer-model'

export type ComposerStep = 'copy' | 'variables' | 'channels'

export const COMPOSER_STEPS: ReadonlyArray<{ id: ComposerStep; label: string }> = [
  { id: 'copy', label: 'Copy' },
  { id: 'variables', label: 'Variables and preview' },
  { id: 'channels', label: 'Formats and channels' },
]

export function ComposerStepNav({ step, onStep }: { step: ComposerStep; onStep: (step: ComposerStep) => void }) {
  return (
    <nav aria-label="Composer steps" className="flex flex-wrap gap-1">
      {COMPOSER_STEPS.map((s, index) => (
        <button
          key={s.id}
          type="button"
          aria-current={s.id === step ? 'step' : undefined}
          onClick={() => onStep(s.id)}
          className={cn(
            'rounded-md px-2.5 py-1 text-xs',
            s.id === step ? 'bg-primary/10 font-medium text-foreground' : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {index + 1}. {s.label}
        </button>
      ))}
    </nav>
  )
}

interface ComposerStepBodyProps {
  step: ComposerStep
  draft: ComposerDraft
  channels: ReadonlyArray<SocialChannel>
  post: SocialPost | undefined
  stale: boolean
  rendering: boolean
  onChange: (draft: ComposerDraft) => void
  onRender: () => void
}

export function ComposerStepBody({ step, draft, channels, post, stale, rendering, onChange, onRender }: ComposerStepBodyProps) {
  if (step === 'copy') return <SocialsComposerProposal draft={draft} channels={channels} onChange={onChange} />
  if (step === 'channels') return <SocialsComposerChannels draft={draft} channels={channels} onChange={onChange} />
  return (
    <div className="socials-composer-grid grid gap-4 lg:grid-cols-2">
      {draft.template ? (
        <SocialsVariablesForm
          schema={draft.template.variables_schema}
          variables={draft.variables}
          sources={draft.sources}
          onChange={(variables, sources) => onChange({ ...draft, variables, sources })}
        />
      ) : (
        <p className="text-sm text-muted-foreground">This post has no template yet, so it has no variables to fill.</p>
      )}
      <SocialsComposerPreview post={post} video={isVideoDraft(draft)} stale={stale} busy={rendering} onRender={onRender} />
    </div>
  )
}
