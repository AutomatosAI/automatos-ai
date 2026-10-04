'use client'

/**
 * PRD-251B US-B207, step 4 — Making and approving (Plan.dc.html): when images and videos are
 * made (on their day or the day before, at the make time), the visual mix, the AI tools the
 * workspace has (set in the Brand kit's AI tools), what happens to a post not approved by
 * its slot, and the render minutes the plan needs.
 *
 * PRD-251C US-C202: the Rhythm. Daily keeps the above; Weekly makes the next 7 days' posts
 * on its day at the make time, Monthly the next month's on its date, each approved in one
 * sitting in the Queue; the evening reminder's time; and when the next batch is made.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { SocialPlanRhythm, SocialWeekday } from '@/lib/socials-plan-types'
import { Segmented } from '../studio/editor-ui'
import {
  LATE_CHOICES, MAX_BATCH_DATE, MIX_PRESETS, WEEKDAYS, WEEKDAY_LABELS, cadenceSummary, type MixKey, type PlanDraft,
} from './plan-model'
import { PlanAiTools } from './plan-ai-tools'
import { RHYTHM_CHOICES, batchDayChange, nextBatchLine, ordinal, rhythmSummary } from './plan-rhythm'
import { PlanStepHeading } from './plan-ui'

const SELECT = 'h-[42px] w-full rounded-md border border-input bg-background px-2 text-sm'
export const QUEUE_NOTE = 'Every post waits in your Queue for approval. You are told in Automatos when the day\'s posts are ready.'
export const BATCH_QUEUE_NOTE =
  'The batch waits in your Queue as one group: approve it in one go. You are told when it is ready, and the evening before for any post still waiting.'
const BATCH_DATES = Array.from({ length: MAX_BATCH_DATE }, (_, i) => i + 1)

function DayChoice({ id, label, value, time, onChange }: { id: string; label: string; value: number; time: string; onChange: (early: number) => void }) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor={id}>{label}</Label>
      <select id={id} className={SELECT} value={value} onChange={(e) => onChange(Number(e.target.value))}>
        <option value={0}>On their day at {time}</option>
        <option value={1}>The day before at {time}</option>
      </select>
    </div>
  )
}

type SetDraft = (changes: Partial<PlanDraft>) => void

function MakeTime({ draft, set, label }: { draft: PlanDraft; set: SetDraft; label: string }) {
  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor="plan-make-time">{label}</Label>
      <Input id="plan-make-time" type="time" value={draft.makeTime} onChange={(e) => set({ makeTime: e.target.value })} />
    </div>
  )
}

function DailyTiming({ draft, set }: { draft: PlanDraft; set: SetDraft }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <DayChoice id="plan-make-images" label="Images and carousels are made" value={draft.imagesEarly} time={draft.makeTime} onChange={(imagesEarly) => set({ imagesEarly })} />
      <DayChoice id="plan-make-videos" label="Videos are made" value={draft.videosEarly} time={draft.makeTime} onChange={(videosEarly) => set({ videosEarly })} />
      <MakeTime draft={draft} set={set} label="Make time" />
    </div>
  )
}

function BatchTiming({ draft, set }: { draft: PlanDraft; set: SetDraft }) {
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {draft.rhythm === 'weekly' ? (
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-batch-day">The week is made on</Label>
          <select id="plan-batch-day" className={SELECT} value={draft.batchDay}
            onChange={(e) => set(batchDayChange(draft, e.target.value as SocialWeekday))}>
            {WEEKDAYS.map((day) => <option key={day} value={day}>{WEEKDAY_LABELS[day]}</option>)}
          </select>
        </div>
      ) : (
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-batch-date">The month is made on the</Label>
          <select id="plan-batch-date" className={SELECT} value={draft.batchDate} onChange={(e) => set({ batchDate: Number(e.target.value) })}>
            {BATCH_DATES.map((day) => <option key={day} value={day}>{ordinal(day)}</option>)}
          </select>
        </div>
      )}
      <MakeTime draft={draft} set={set} label="At" />
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-remind-at">Evening reminder at</Label>
        <Input id="plan-remind-at" type="time" value={draft.remindAt} onChange={(e) => set({ remindAt: e.target.value })} />
      </div>
    </div>
  )
}

interface MakingProps {
  draft: PlanDraft
  set: SetDraft
  /** The saved plan's next batch (ISO), from the server; none for a daily or new plan. */
  nextBatchAt?: string | null
}

export function PlanStepMaking({ draft, set, nextBatchAt = null }: MakingProps) {
  const summary = cadenceSummary(draft)
  const mix = MIX_PRESETS.find((preset) => preset.key === draft.mix) ?? MIX_PRESETS[0]
  const batched = draft.rhythm !== 'daily'
  const next = batched ? nextBatchLine(nextBatchAt, draft.timezone) : null
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="Making and approving" lead={rhythmSummary(draft)} />
      <div className="flex flex-col gap-1.5">
        <span className="text-sm font-medium text-foreground">Rhythm</span>
        <Segmented<SocialPlanRhythm> label="Rhythm" choices={RHYTHM_CHOICES} value={draft.rhythm} onChange={(rhythm) => set({ rhythm })} />
        {next && <span className="text-[12.5px] text-muted-foreground">{next}</span>}
      </div>
      {batched ? <BatchTiming draft={draft} set={set} /> : <DailyTiming draft={draft} set={set} />}
      <div className="flex flex-col gap-1.5">
        <span className="text-sm font-medium text-foreground">Visuals</span>
        <Segmented<MixKey> label="Visual mix" choices={MIX_PRESETS.map((p) => ({ value: p.key, label: p.label }))} value={draft.mix} onChange={(key) => set({ mix: key })} />
        <span className="text-[12.5px] text-muted-foreground">{mix.note}</span>
      </div>
      <PlanAiTools />
      <div className="flex flex-col gap-1.5">
        <span className="text-sm font-medium text-foreground">If a post is not approved by its slot</span>
        <Segmented label="Late approval" choices={LATE_CHOICES} value={draft.latePolicy} onChange={(latePolicy) => set({ latePolicy })} />
        <span className="text-[12.5px] text-muted-foreground">{batched ? BATCH_QUEUE_NOTE : QUEUE_NOTE}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-foreground">Render minutes</span>
        <span className="text-[15px]">About {summary.renderMinutes} for this plan</span>
        <span className="text-[12.5px] text-muted-foreground">{summary.videos} videos. Images use none.</span>
      </div>
    </div>
  )
}
