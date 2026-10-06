'use client'

/** PRD-255 US-007 — the small inputs the Brand kit page's design-system sections share. */
import type { ReactNode } from 'react'

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export const SELECT_CLASS = 'h-9 w-full rounded-md border border-input bg-background px-2 text-sm'

export function SectionTitle({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="mb-2">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      {children && <p className="text-xs text-muted-foreground">{children}</p>}
    </div>
  )
}

interface NumberFieldProps {
  id: string
  label: string
  value: number | undefined
  step: number
  unit?: string
  onChange: (value: number) => void
}

/** A number the server bounds (a 422 names the field and its range); an empty box changes nothing. */
export function NumberField({ id, label, value, step, unit, onChange }: NumberFieldProps) {
  return (
    <div className="min-w-0">
      <Label htmlFor={id} className="text-xs">{label}{unit ? ` (${unit})` : ''}</Label>
      <Input
        id={id}
        type="number"
        inputMode="decimal"
        step={step}
        value={value ?? ''}
        onChange={(e) => {
          const next = Number(e.target.value)
          if (e.target.value.trim() !== '' && Number.isFinite(next)) onChange(next)
        }}
      />
    </div>
  )
}
