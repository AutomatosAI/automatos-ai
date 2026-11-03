'use client'

/**
 * PRD-268 §6.9 — what the "Proactive Auto" section's rows share: a switch row (a label, one line under it and the
 * switch), the line saying what Auto is doing here, and the one line a refusal or a failed read shows. Tokens
 * only, so the rows read right in Studio light, Studio dark and Classic, as the readiness section beside them does.
 */
import { useId } from 'react'
import { Switch } from '@/components/ui/switch'
import { refusedLine } from '@/components/proactive/use-suggestion-card'
import type { EffectiveMode } from '@/lib/proactive/settings'
import { modeLine } from './copy'

export interface SwitchRowProps {
  readonly label: string
  readonly detail: string
  readonly checked: boolean
  readonly disabled: boolean
  readonly onCheckedChange?: (on: boolean) => void
}

/** A label, its line, and the switch; disabled is read-only, never hidden. */
export function SwitchRow({ label, detail, checked, disabled, onCheckedChange }: SwitchRowProps) {
  const switchId = useId()
  const labelId = useId()
  const detailId = useId()
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <div className="min-w-0 space-y-0.5">
        <label id={labelId} htmlFor={switchId} className="text-sm font-medium text-foreground">
          {label}
        </label>
        <p id={detailId} className="text-xs text-muted-foreground">
          {detail}
        </p>
      </div>
      <Switch
        id={switchId}
        checked={checked}
        disabled={disabled}
        aria-labelledby={labelId}
        aria-describedby={detailId}
        onCheckedChange={onCheckedChange}
      />
    </div>
  )
}

/** What Auto is doing here, in one line (`modeLine`), or nothing. */
export function ModeLine({ mode, switchedOn }: { readonly mode: EffectiveMode | null; readonly switchedOn: boolean }) {
  const line = modeLine(mode, switchedOn)
  if (line === null) return null
  return (
    <p role="status" data-effective-mode={mode ?? undefined} className="text-xs text-foreground">
      {line}
    </p>
  )
}

/** The refusal's one line (the server's sentence when it gave one), or nothing. */
export function FailureLine({ error }: { readonly error: unknown }) {
  const line = error ? refusedLine(error) : null
  if (line === null) return null
  return (
    <p role="alert" className="text-xs text-[hsl(var(--destructive))]">
      {line}
    </p>
  )
}
