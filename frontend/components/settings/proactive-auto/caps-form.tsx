'use client'

/**
 * PRD-268 §6.9 — the workspace's caps, an owner's or an admin's to tighten.
 *
 * The seven caps PUT /api/proactive/settings takes (`CapsChange`), each shown at its value with the range the server
 * accepts: a whole number no looser than the platform's (`platform_caps`), and the shortest gap between check-ins
 * no longer than the longest. A value outside it says so and holds Save. Save sends only the caps that changed,
 * shown at once and put back, with the server's sentence, when the server refuses; what was typed stays, to fix.
 */
import { useId, useState } from 'react'
import { PRIMARY_BUTTON_CLASS } from '@/components/proactive/card-parts'
import { useChangeProactiveSettings } from '@/hooks/use-proactive-settings'
import {
  CAP_NAMES,
  capBounds,
  capFits,
  capValues,
  changedCaps,
  type CapBounds,
  type CapDraft,
  type CapName,
  type ProactiveSettings,
} from '@/lib/proactive/settings'
import { cn } from '@/lib/utils'
import { CAPS_INTRO, CAPS_TITLE, CAP_COPY, SAVE_CAPS, SAVING, outOfRangeLine, rangeLine } from './copy'
import { FailureLine } from './parts'

const INPUT_CLASS =
  'h-9 w-28 rounded-xl border border-border bg-background px-3 text-sm text-foreground ' +
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring aria-[invalid=true]:border-[hsl(var(--destructive))]'

interface CapFieldProps {
  readonly name: CapName
  readonly value: string
  readonly bounds: CapBounds
  readonly fits: boolean
  readonly onChange: (name: CapName, value: string) => void
}

function CapField({ name, value, bounds, fits, onChange }: CapFieldProps) {
  const id = `proactive-cap-${name}`
  const { label, note } = CAP_COPY[name]
  const hint = fits ? [note, rangeLine(bounds)].filter(Boolean).join(' ') : outOfRangeLine(bounds)
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-sm font-medium text-foreground">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        min={bounds.min}
        max={bounds.max}
        step={1}
        value={value}
        aria-invalid={!fits}
        aria-describedby={`${id}-hint`}
        onChange={(event) => onChange(name, event.target.value)}
        className={INPUT_CLASS}
      />
      <p id={`${id}-hint`} className={cn('text-xs', fits ? 'text-muted-foreground' : 'text-[hsl(var(--destructive))]')}>
        {hint}
      </p>
    </div>
  )
}

/** The caps, their ranges, and Save. */
export function CapsForm({ settings }: { readonly settings: ProactiveSettings }) {
  const change = useChangeProactiveSettings()
  const introId = useId()
  const [draft, setDraft] = useState<CapDraft>({})
  const values = capValues(settings.caps, draft)
  const fits = Object.fromEntries(CAP_NAMES.map((name) => [name, capFits(name, values, settings)])) as Record<CapName, boolean>
  const changed = changedCaps(values, settings.caps)
  const canSave = !change.isLoading && CAP_NAMES.every((name) => fits[name]) && Object.keys(changed).length > 0

  const edit = (name: CapName, value: string) => {
    if (change.isError) change.reset()
    setDraft((typed) => ({ ...typed, [name]: value }))
  }
  const save = () => change.mutate({ caps: changed }, { onSuccess: () => setDraft({}) })

  return (
    <fieldset aria-describedby={introId} className="space-y-3 border-t border-border pt-4">
      <legend className="float-left w-full text-sm font-medium text-foreground">{CAPS_TITLE}</legend>
      <p id={introId} className="clear-left text-xs text-muted-foreground">
        {CAPS_INTRO}
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        {CAP_NAMES.map((name) => (
          <CapField
            key={name}
            name={name}
            value={values[name]}
            bounds={capBounds(name, values, settings)}
            fits={fits[name]}
            onChange={edit}
          />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" disabled={!canSave} aria-busy={change.isLoading} onClick={save} className={PRIMARY_BUTTON_CLASS}>
          {change.isLoading ? SAVING : SAVE_CAPS}
        </button>
        <FailureLine error={change.isError ? change.error : null} />
      </div>
    </fieldset>
  )
}
