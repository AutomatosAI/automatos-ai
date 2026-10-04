'use client'

/**
 * PRD-251B US-B207, step 3 — What to research (Plan.dc.html): the sources Auto reads (the
 * workspace's knowledge, its Deliverables, the brand kit's website, and GitHub through
 * Composio when connected), notes for Auto, the never-say list, and when research runs
 * again (weekly at a day and time, or only when asked). Every claim in a post has to come
 * from the bank. PRD-251C US-C104: how long a posted idea stays off research's list.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialWeekday } from '@/lib/socials-plan-types'
import { MAX_REPEAT_AFTER_DAYS, WEEKDAY_LABELS, WEEKDAYS, repeatDays, type PlanDraft } from './plan-model'
import { PlanStepHeading } from './plan-ui'

const SELECT = 'h-[42px] w-full rounded-md border border-input bg-background px-2 text-sm'
type SourceKey = keyof Omit<PlanDraft['sources'], 'notes'>
const SOURCES: ReadonlyArray<{ key: SourceKey; name: string; what: string }> = [
  { key: 'knowledge', name: 'Knowledge', what: 'All documents in this workspace' },
  { key: 'deliverables', name: 'Deliverables', what: 'Blog posts and reports' },
  { key: 'github', name: 'GitHub', what: 'README, docs, releases and merged pull requests, when connected in Composio' },
  { key: 'website', name: 'Website', what: 'The site in your brand kit' },
]
export const BANK_NOTE = 'New releases and documents join the bank. A topic that has been used is not repeated unless you pin it.'
export const REPEAT_NOTE = "Research adds nothing close to a topic in any plan's bank, or to a post from this many days back."

export function PlanStepResearch({ draft, set }: { draft: PlanDraft; set: (changes: Partial<PlanDraft>) => void }) {
  const toggle = (key: SourceKey) => set({ sources: { ...draft.sources, [key]: !draft.sources[key] } })
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="What to research" lead="Auto reads these, pulls out features, facts and stories with their sources, and keeps them in the content bank. Every claim in a post has to come from the bank." />
      {SOURCES.map((source) => (
        <label key={source.key} className="flex min-h-[54px] items-center gap-3 rounded-xl border border-border px-3 py-2.5">
          <input type="checkbox" checked={draft.sources[source.key]} onChange={() => toggle(source.key)} />
          <span className="flex flex-col gap-0.5">
            <span className="font-medium text-foreground">{source.name}</span>
            <span className="text-[12.5px] text-muted-foreground">{source.what}</span>
          </span>
        </label>
      ))}
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-notes">Notes for Auto</Label>
        <Textarea id="plan-notes" rows={3} value={draft.sources.notes} maxLength={2000}
          onChange={(e) => set({ sources: { ...draft.sources, notes: e.target.value } })} />
      </div>
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_200px_120px]">
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-avoid">Never say (comma separated)</Label>
          <Input id="plan-avoid" value={draft.neverSay} onChange={(e) => set({ neverSay: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-refresh">Research again</Label>
          <select id="plan-refresh" className={SELECT} value={draft.researchEnabled ? draft.researchDay : 'ask'}
            onChange={(e) => (e.target.value === 'ask' ? set({ researchEnabled: false }) : set({ researchEnabled: true, researchDay: e.target.value as SocialWeekday }))}>
            {WEEKDAYS.map((day) => <option key={day} value={day}>Every {WEEKDAY_LABELS[day]}</option>)}
            <option value="ask">Only when I ask</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-refresh-time">At</Label>
          <Input id="plan-refresh-time" type="time" value={draft.researchTime} disabled={!draft.researchEnabled}
            onChange={(e) => set({ researchTime: e.target.value })} />
        </div>
      </div>
      <div className="flex flex-col gap-1 sm:w-[260px]">
        <Label htmlFor="plan-repeat">Not again for (days)</Label>
        <Input id="plan-repeat" type="number" min={1} max={MAX_REPEAT_AFTER_DAYS} value={draft.repeatAfterDays}
          onChange={(e) => set({ repeatAfterDays: repeatDays(e.target.value) })} />
        <span className="text-[12.5px] text-muted-foreground">{REPEAT_NOTE}</span>
      </div>
      <p className="m-0 text-[12.5px] text-muted-foreground">{BANK_NOTE}</p>
    </div>
  )
}
