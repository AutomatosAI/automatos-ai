'use client'

/**
 * PRD-251B US-B207, step 5 — Content bank (Plan.dc.html): the plan's topics, each with its
 * formats, whether it is used or pinned, its angle and its facts with their sources. For
 * each slot Auto takes the next unused topic that suits the slot's format (a topic pinned to
 * the slot's day first). Add a topic by hand, or Research again. The server refuses a fact
 * without a source, a title the bank holds and the plan's never-say words, with the reason.
 * PRD-251C US-C101: the bank says, in the server's words, when research cannot run.
 */
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SocialPlanAutoTopic, SocialTopic } from '@/lib/socials-plan-types'
import { useResearchSocialPlan, useSocialPlanTopics, useWriteSocialTopic } from '@/hooks/use-socials-plans'
import { PlanAutoSuggestions } from './plan-auto-notes'
import { PlanTopicForm } from './plan-topic-form'
import { PlanStepHeading } from './plan-ui'

export const SAVE_FIRST = 'Save the plan first: its content bank fills once it exists.'
const SHOWN_AT_FIRST = 6
const FORMAT_WORDS: Record<string, string> = { image: 'Image', carousel: 'Carousel', video: 'Video', fact_card: 'Fact card', infographic: 'Infographic', text: 'Text' }

export function topicUse(topic: Pick<SocialTopic, 'used_at' | 'pinned_on'>): string {
  if (topic.used_at) return 'Used'
  return topic.pinned_on ? `Pinned to ${topic.pinned_on}` : 'Unused'
}

function TopicCard({ topic, onPin, onDelete }: { topic: SocialTopic; onPin: (day: string | null) => void; onDelete: () => void }) {
  const formats = topic.formats.length ? topic.formats.map((f) => FORMAT_WORDS[f] ?? f).join(' · ') : 'Any format'
  return (
    <article aria-label={topic.title} className="flex flex-col gap-2.5 rounded-xl border border-border bg-card p-3.5">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-muted-foreground">
        <span className="rounded-full border border-border px-2 py-0.5">{formats}</span>
        <span>{topicUse(topic)}</span>
        {topic.origin === 'research' && <span>· researched</span>}
      </div>
      <h3 className="m-0 font-serif text-[21px] font-normal leading-[1.15] text-foreground">{topic.title}</h3>
      {topic.angle && <p className="m-0 text-sm text-muted-foreground">{topic.angle}</p>}
      {topic.facts.map((fact, i) => (
        <div key={i} className="flex flex-col gap-0.5 rounded-lg bg-secondary/60 px-2.5 py-2">
          <span className="text-[13px] text-foreground">{fact.text}</span>
          <span className="text-[11px] text-muted-foreground">{fact.source.label} · {fact.source.kind}</span>
        </div>
      ))}
      {!topic.used_at && (
        <div className="flex flex-wrap items-center gap-2">
          <Input aria-label={`Pin ${topic.title} to a day`} type="date" className="h-[34px] w-[170px]" value={topic.pinned_on ?? ''}
            onChange={(e) => onPin(e.target.value || null)} />
          <Button type="button" variant="ghost" size="sm" onClick={onDelete}>Delete</Button>
        </div>
      )}
    </article>
  )
}

interface PlanStepBankProps {
  planId: string | null
  /** A new plan's ideas from Auto: they join the bank when it is saved. */
  ideas?: ReadonlyArray<SocialPlanAutoTopic>
  onLeaveOut?: (title: string) => void
}

export function PlanStepBank({ planId, ideas = [], onLeaveOut = () => undefined }: PlanStepBankProps) {
  const [adding, setAdding] = useState(false)
  const [all, setAll] = useState(false)
  const { data } = useSocialPlanTopics(planId)
  const write = useWriteSocialTopic(planId ?? '')
  const research = useResearchSocialPlan()
  if (!planId) {
    return (
      <div className="flex flex-col gap-4">
        <PlanStepHeading title="Content bank" lead="The topics each post is made from." />
        <p className="m-0 text-sm text-muted-foreground">{SAVE_FIRST}</p>
        {ideas.length > 0 && <PlanAutoSuggestions topics={ideas} onLeaveOut={onLeaveOut} />}
      </div>
    )
  }
  const topics = data?.topics ?? []
  const shown = all ? topics : topics.slice(0, SHOWN_AT_FIRST)
  const actions = (
    <div className="flex gap-2">
      <Button type="button" variant="outline" onClick={() => setAdding(true)}>Add a topic</Button>
      <Button type="button" disabled={research.isLoading} onClick={() => research.mutate(planId)}>Research again</Button>
    </div>
  )
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="Content bank" action={actions}
        lead={`${data?.total ?? 0} topics, ${data?.unused ?? 0} unused. For each slot Auto takes the next unused topic that suits the slot's format.`} />
      {data?.research_note && <p role="status" aria-label="Research" className="m-0 rounded-lg bg-secondary/60 px-3 py-2 text-sm text-foreground">{data.research_note}</p>}
      {adding && (
        <PlanTopicForm busy={write.isLoading} onCancel={() => setAdding(false)}
          onSave={(input) => write.mutate({ kind: 'add', input }, { onSuccess: () => setAdding(false) })} />
      )}
      <div className="grid gap-3 lg:grid-cols-2">
        {shown.map((topic) => (
          <TopicCard key={topic.id} topic={topic}
            onPin={(day) => write.mutate({ kind: 'pin', topicId: topic.id, pinnedOn: day })}
            onDelete={() => write.mutate({ kind: 'delete', topicId: topic.id })} />
        ))}
      </div>
      {!all && topics.length > SHOWN_AT_FIRST && (
        <Button type="button" variant="ghost" className="self-start" onClick={() => setAll(true)}>Show all {topics.length} topics</Button>
      )}
    </div>
  )
}
