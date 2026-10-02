'use client'

/**
 * PRD-251B US-B207, step 4 — Making and approving (Plan.dc.html): when images and videos are
 * made (on their day or the day before, at the make time), the visual mix, the AI tools the
 * workspace has (set in the Brand kit's AI tools), what happens to a post not approved by
 * its slot, and the render minutes the plan needs. Posts are made close to their slot, so
 * the work and the AI spend are spread over the plan.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Segmented } from '../studio/editor-ui'
import { LATE_CHOICES, MIX_PRESETS, cadenceSummary, type MixKey, type PlanDraft } from './plan-model'
import { PlanAiTools } from './plan-ai-tools'
import { PlanStepHeading } from './plan-ui'

const SELECT = 'h-[42px] w-full rounded-md border border-input bg-background px-2 text-sm'
export const QUEUE_NOTE = 'Every post waits in your Queue for approval. You are told in Automatos when the day\'s posts are ready.'

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

export function PlanStepMaking({ draft, set }: { draft: PlanDraft; set: (changes: Partial<PlanDraft>) => void }) {
  const summary = cadenceSummary(draft)
  const mix = MIX_PRESETS.find((preset) => preset.key === draft.mix) ?? MIX_PRESETS[0]
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="Making and approving" lead="Posts are made close to their slot, so the work and the AI spend are spread over the plan instead of spent up front." />
      <div className="grid gap-3 sm:grid-cols-3">
        <DayChoice id="plan-make-images" label="Images and carousels are made" value={draft.imagesEarly} time={draft.makeTime} onChange={(imagesEarly) => set({ imagesEarly })} />
        <DayChoice id="plan-make-videos" label="Videos are made" value={draft.videosEarly} time={draft.makeTime} onChange={(videosEarly) => set({ videosEarly })} />
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-make-time">Make time</Label>
          <Input id="plan-make-time" type="time" value={draft.makeTime} onChange={(e) => set({ makeTime: e.target.value })} />
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-sm font-medium text-foreground">Visuals</span>
        <Segmented<MixKey> label="Visual mix" choices={MIX_PRESETS.map((p) => ({ value: p.key, label: p.label }))} value={draft.mix} onChange={(key) => set({ mix: key })} />
        <span className="text-[12.5px] text-muted-foreground">{mix.note}</span>
      </div>
      <PlanAiTools />
      <div className="flex flex-col gap-1.5">
        <span className="text-sm font-medium text-foreground">If a post is not approved by its slot</span>
        <Segmented label="Late approval" choices={LATE_CHOICES} value={draft.latePolicy} onChange={(latePolicy) => set({ latePolicy })} />
        <span className="text-[12.5px] text-muted-foreground">{QUEUE_NOTE}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-foreground">Render minutes</span>
        <span className="text-[15px]">About {summary.renderMinutes} for this plan</span>
        <span className="text-[12.5px] text-muted-foreground">{summary.videos} videos. Images use none.</span>
      </div>
    </div>
  )
}
