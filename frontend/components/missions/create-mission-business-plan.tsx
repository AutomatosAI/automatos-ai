'use client'

/**
 * The Business Plan template's extra fields in the New mission form. Moved out of
 * create-mission-modal.tsx; the ids, labels and placeholders are unchanged.
 */

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { BusinessPlanFields as BusinessPlanState } from './use-create-mission-form'

interface PlanInputProps {
  id: string
  label: string
  placeholder: string
  value: string
  onChange: (value: string) => void
}

function PlanInput({ id, label, placeholder, value, onChange }: PlanInputProps) {
  return (
    <div className="space-y-1">
      <Label htmlFor={id} className="text-[11px]">{label}</Label>
      <Input
        id={id}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 text-xs"
      />
    </div>
  )
}

/* Business Plan extra fields */
export function BusinessPlanFields({ plan }: { plan: BusinessPlanState }) {
  return (
    <div className="space-y-3 rounded-lg border border-primary/20 bg-primary/5 p-3">
      <Label className="text-xs font-medium text-primary">Business Plan Details</Label>
      <div className="grid grid-cols-2 gap-2">
        <PlanInput id="bp-name" label="Business Name *" placeholder="e.g. BrewCraft" value={plan.businessName} onChange={plan.setBusinessName} />
        <PlanInput id="bp-type" label="Business Type *" placeholder="e.g. SaaS, retail, service" value={plan.businessType} onChange={plan.setBusinessType} />
      </div>
      <PlanInput id="bp-industry" label="Industry *" placeholder="e.g. Coffee & Beverages" value={plan.industry} onChange={plan.setIndustry} />
      <div className="grid grid-cols-2 gap-2">
        <PlanInput id="bp-market" label="Target Market" placeholder="e.g. Urban millennials" value={plan.targetMarket} onChange={plan.setTargetMarket} />
        <PlanInput id="bp-goals" label="Goals" placeholder="e.g. Launch in 6 months" value={plan.businessGoals} onChange={plan.setBusinessGoals} />
      </div>
    </div>
  )
}
