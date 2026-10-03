'use client'

/**
 * PRD-251B (3 Oct 2026 pass) — Plan with Auto, at the top of a new plan: the person says what
 * their socials should do (a greyed example shows the kind of thing), ticks what Auto may
 * read, and Auto drafts the plan into the steps below. Nothing is saved until they save it.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { browserTimezone } from '@/lib/social-time'
import type { SocialPlanAutoDraft, SocialPlanAutoSources } from '@/lib/socials-plan-types'
import { useDraftSocialPlan } from '@/hooks/use-socials-plan-draft'
import {
  ALL_AUTO_SOURCES,
  AUTO_EXAMPLE,
  AUTO_LABEL,
  AUTO_LEAD,
  AUTO_SOURCES,
  AUTO_SOURCES_LABEL,
  AUTO_TITLE,
  draftButtonLabel,
  draftStatusNote,
} from './plan-auto-model'

const REQUEST_MAX_CHARS = 2000

function SourcePicker({ sources, onToggle }: { sources: SocialPlanAutoSources; onToggle: (key: keyof SocialPlanAutoSources) => void }) {
  return (
    <fieldset className="flex flex-wrap gap-2">
      <legend className="mb-1.5 text-[12.5px] text-muted-foreground">{AUTO_SOURCES_LABEL}</legend>
      {AUTO_SOURCES.map((source) => (
        <label key={source.key} className="flex min-h-[40px] items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-[13px]">
          <input type="checkbox" checked={sources[source.key]} onChange={() => onToggle(source.key)} />
          <span className="font-medium text-foreground">{source.name}</span>
          <span className="text-muted-foreground">{source.what}</span>
        </label>
      ))}
    </fieldset>
  )
}

interface PlanWithAutoProps {
  canEdit: boolean
  drafted: boolean
  onDrafted: (draft: SocialPlanAutoDraft) => void
}

export function PlanWithAuto({ canEdit, drafted, onDrafted }: PlanWithAutoProps) {
  const [request, setRequest] = useState('')
  const [sources, setSources] = useState<SocialPlanAutoSources>(ALL_AUTO_SOURCES)
  const draft = useDraftSocialPlan()
  const ask = () => draft.mutate({ request: request.trim(), timezone: browserTimezone(), sources }, { onSuccess: onDrafted })
  return (
    <section aria-label={AUTO_TITLE} className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <div className="flex flex-col gap-1">
        <h2 className="m-0 text-[17px] font-semibold text-foreground">{AUTO_TITLE}</h2>
        <p className="m-0 max-w-[72ch] text-[13.5px] leading-[1.5] text-muted-foreground">{AUTO_LEAD}</p>
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-auto-request">{AUTO_LABEL}</Label>
        <Textarea id="plan-auto-request" rows={3} value={request} maxLength={REQUEST_MAX_CHARS} placeholder={AUTO_EXAMPLE}
          disabled={!canEdit} onChange={(e) => setRequest(e.target.value)} />
      </div>
      <SourcePicker sources={sources} onToggle={(key) => setSources({ ...sources, [key]: !sources[key] })} />
      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" aria-busy={draft.isLoading} disabled={!canEdit || !request.trim() || draft.isLoading} onClick={ask}>
          {draftButtonLabel(draft.isLoading, drafted)}
        </Button>
        <span role="status" className="text-[12.5px] text-muted-foreground">{draftStatusNote(draft.isLoading, drafted)}</span>
      </div>
    </section>
  )
}
