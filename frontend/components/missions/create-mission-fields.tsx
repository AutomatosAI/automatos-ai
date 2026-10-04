'use client'

/**
 * The New mission form's sections: mission type, the Business Plan fields, name and description,
 * power mode, the on/off switches and tags. Moved out of create-mission-modal.tsx; the markup,
 * ids and labels are unchanged.
 */

import { HelpCircle } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  MISSION_TEMPLATES,
  POWER_MODES,
  type MissionTemplateOption,
  type PowerMode,
} from './create-mission-options'
import type { MissionFormFields } from './use-create-mission-form'

const OPTION_CLASS_SELECTED = 'border-primary bg-primary/5 ring-1 ring-primary/30'
const OPTION_CLASS_IDLE = 'border-border hover:border-muted-foreground/40'

interface TemplateOptionProps {
  tmpl: MissionTemplateOption
  isSelected: boolean
  onSelect: (id: string | null) => void
}

function TemplateOption({ tmpl, isSelected, onSelect }: TemplateOptionProps) {
  const Icon = tmpl.icon
  return (
    <button
      type="button"
      onClick={() => onSelect(tmpl.id)}
      className={cn(
        'flex items-start gap-2.5 rounded-lg border p-2.5 text-left transition-colors',
        isSelected ? OPTION_CLASS_SELECTED : OPTION_CLASS_IDLE,
      )}
    >
      <Icon className={cn('w-4 h-4 mt-0.5 shrink-0', isSelected ? 'text-primary' : 'text-muted-foreground')} />
      <div className="min-w-0">
        <div className="text-xs font-medium truncate">{tmpl.name}</div>
        <div className="text-[10px] text-muted-foreground truncate">{tmpl.description}</div>
        <div className="text-[9px] text-muted-foreground/60 mt-0.5">{tmpl.estimatedCost}</div>
      </div>
    </button>
  )
}

/* Template selector */
export function TemplatePicker({ selected, onSelect }: { selected: string | null; onSelect: (id: string | null) => void }) {
  return (
    <div className="space-y-2">
      <Label>Mission Type</Label>
      <div className="grid grid-cols-2 gap-2">
        {MISSION_TEMPLATES.map((tmpl) => (
          <TemplateOption key={tmpl.id ?? 'custom'} tmpl={tmpl} isSelected={selected === tmpl.id} onSelect={onSelect} />
        ))}
      </div>
    </div>
  )
}

/* Name + Description (shown for non-business-plan or as additional context) */
export function MissionTextFields({ form, isBusinessPlan }: { form: MissionFormFields; isBusinessPlan: boolean }) {
  return (
    <>
      {!isBusinessPlan && (
        <div className="space-y-2">
          <Label htmlFor="mission-name">Mission Name</Label>
          <Input
            id="mission-name"
            placeholder="e.g. Research top AI agent frameworks"
            value={form.name}
            onChange={(e) => form.setName(e.target.value)}
            autoFocus
          />
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="mission-description">
          {isBusinessPlan ? 'Additional Context' : 'Description'}
        </Label>
        <Textarea
          id="mission-description"
          placeholder={isBusinessPlan
            ? 'Any additional context, constraints, or specific requirements...'
            : 'Describe what you want to accomplish, any constraints, output format...'}
          value={form.description}
          onChange={(e) => form.setDescription(e.target.value)}
          rows={isBusinessPlan ? 2 : 4}
        />
      </div>
    </>
  )
}

interface PowerModeOptionProps {
  mode: (typeof POWER_MODES)[number]
  isSelected: boolean
  orchestratorModel: string | null
  onSelect: (mode: PowerMode) => void
}

function PowerModeOption({ mode, isSelected, orchestratorModel, onSelect }: PowerModeOptionProps) {
  const Icon = mode.icon
  const showModel = mode.id === 'max' && orchestratorModel
  return (
    <button
      type="button"
      onClick={() => onSelect(mode.id)}
      className={cn(
        'relative flex flex-col items-start gap-1 rounded-lg border p-2.5 text-left transition-colors',
        isSelected ? OPTION_CLASS_SELECTED : OPTION_CLASS_IDLE,
      )}
    >
      <div className="flex items-center gap-1.5 w-full">
        <Icon className={cn('w-3.5 h-3.5 shrink-0', isSelected ? 'text-primary' : 'text-muted-foreground')} />
        <span className="text-xs font-medium">{mode.name}</span>
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <HelpCircle className="w-3 h-3 text-muted-foreground hover:text-foreground cursor-help ml-auto shrink-0" />
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-xs">
              <p className="text-xs">{mode.tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <div className="text-[10px] text-muted-foreground leading-tight">{mode.description}</div>
      {showModel && (
        <div className="text-[9px] text-primary/70 truncate w-full">
          All agents on {orchestratorModel?.includes('/') ? orchestratorModel.split('/').pop() : orchestratorModel}
        </div>
      )}
    </button>
  )
}

interface PowerModePickerProps {
  powerMode: PowerMode
  orchestratorModel: string | null
  onSelect: (mode: PowerMode) => void
}

/* Power mode selector */
export function PowerModePicker({ powerMode, orchestratorModel, onSelect }: PowerModePickerProps) {
  return (
    <div className="space-y-2">
      <Label>Power Mode</Label>
      <div className="grid grid-cols-3 gap-2">
        {POWER_MODES.map((mode) => (
          <PowerModeOption
            key={mode.id}
            mode={mode}
            isSelected={powerMode === mode.id}
            orchestratorModel={orchestratorModel}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  )
}

interface SwitchRowProps {
  id: string
  label: string
  hint: string
  checked: boolean
  onToggle: () => void
}

/** One labelled on/off switch: Pause on budget exceeded, Check each step with me (F282). */
export function SwitchRow({ id, label, hint, checked, onToggle }: SwitchRowProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5">
      <div className="space-y-0.5">
        <Label htmlFor={id} className="text-sm cursor-pointer">
          {label}
        </Label>
        <p className="text-[11px] text-muted-foreground">
          {hint}
        </p>
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={onToggle}
        className={cn(
          'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
          checked ? 'bg-primary' : 'bg-muted',
        )}
      >
        <span
          className={cn(
            'pointer-events-none inline-block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
            checked ? 'translate-x-4' : 'translate-x-0',
          )}
        />
      </button>
    </div>
  )
}

export function TagsField({ tags, onChange }: { tags: string; onChange: (tags: string) => void }) {
  return (
    <div className="space-y-2">
      <Label htmlFor="mission-tags">Tags</Label>
      <Input
        id="mission-tags"
        placeholder="e.g. research, competitive-analysis, urgent"
        value={tags}
        onChange={(e) => onChange(e.target.value)}
      />
      <p className="text-xs text-muted-foreground">Comma-separated</p>
    </div>
  )
}
