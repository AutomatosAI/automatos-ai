'use client'

/**
 * PRD-251B US-B207, step 1 — Goal and dates (Plan.dc.html): what the plan is for (Auto reads
 * it before every post it makes), who it is for, its first and last day and its timezone.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { timezoneChoices } from '../studio/editor-when-card'
import type { PlanDraft } from './plan-model'
import { PlanStepHeading } from './plan-ui'

const SELECT = 'h-[42px] w-full rounded-md border border-input bg-background px-2 text-sm'

export function PlanStepGoal({ draft, set }: { draft: PlanDraft; set: (changes: Partial<PlanDraft>) => void }) {
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="Goal and dates" lead="What the plan is for. Auto reads this before every post it makes." />
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-name">Name</Label>
        <Input id="plan-name" value={draft.name} maxLength={200} onChange={(e) => set({ name: e.target.value })} />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-goal">Goal</Label>
        <Textarea id="plan-goal" rows={4} value={draft.goal} maxLength={1000} onChange={(e) => set({ goal: e.target.value })} />
      </div>
      <div className="flex flex-col gap-1">
        <Label htmlFor="plan-audience">Who it is for</Label>
        <Input id="plan-audience" value={draft.audience} maxLength={500} onChange={(e) => set({ audience: e.target.value })} />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-from">Starts</Label>
          <Input id="plan-from" type="date" value={draft.startsOn} onChange={(e) => set({ startsOn: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-to">Ends</Label>
          <Input id="plan-to" type="date" value={draft.endsOn} min={draft.startsOn} onChange={(e) => set({ endsOn: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="plan-tz">Timezone</Label>
          <select id="plan-tz" className={SELECT} value={draft.timezone} onChange={(e) => set({ timezone: e.target.value })}>
            {timezoneChoices(draft.timezone).map((zone) => (
              <option key={zone} value={zone}>{zone}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}
