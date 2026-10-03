'use client'

/**
 * PRD-251B (3 Oct 2026 pass) — what a new plan drafted by Auto shows besides its form: Auto's
 * notes (it drafted the plan; each thing in its answer the plan could not carry), and on the
 * Content bank step the ideas it heard, each of which joins the bank on save unless left out.
 */
import { Button } from '@/components/ui/button'
import type { SocialPlanAutoTopic } from '@/lib/socials-plan-types'
import { DRAFTED_NOTE, SUGGESTIONS_LEAD, SUGGESTIONS_TITLE } from './plan-auto-model'

export function PlanAutoNotes({ warnings }: { warnings: ReadonlyArray<string> }) {
  return (
    <section aria-label="Auto's notes" className="flex flex-col gap-1.5 rounded-xl border border-border bg-secondary/40 p-3.5 text-[13px]">
      <p className="m-0 font-medium text-foreground">{DRAFTED_NOTE}</p>
      {warnings.length > 0 && (
        <ul className="m-0 flex list-disc flex-col gap-0.5 pl-5 text-muted-foreground">
          {warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
    </section>
  )
}

interface SuggestionsProps {
  topics: ReadonlyArray<SocialPlanAutoTopic>
  onLeaveOut: (title: string) => void
}

export function PlanAutoSuggestions({ topics, onLeaveOut }: SuggestionsProps) {
  return (
    <section aria-label={SUGGESTIONS_TITLE} className="flex flex-col gap-3">
      <div className="flex flex-col gap-0.5">
        <h3 className="m-0 text-[15px] font-semibold text-foreground">{SUGGESTIONS_TITLE}</h3>
        <p className="m-0 text-[13px] text-muted-foreground">{SUGGESTIONS_LEAD}</p>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {topics.map((topic) => (
          <article key={topic.title} aria-label={topic.title} className="flex flex-col gap-2 rounded-xl border border-border bg-card p-3.5">
            <h4 className="m-0 font-serif text-[19px] font-normal leading-[1.2] text-foreground">{topic.title}</h4>
            {topic.angle && <p className="m-0 text-sm text-muted-foreground">{topic.angle}</p>}
            <Button type="button" variant="ghost" size="sm" className="self-start" onClick={() => onLeaveOut(topic.title)}>
              Leave out
            </Button>
          </article>
        ))}
      </div>
    </section>
  )
}
