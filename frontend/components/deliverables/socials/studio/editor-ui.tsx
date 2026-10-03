'use client'

/**
 * PRD-251B US-B109 — the editor's shared pieces (Editor.dc.html, TOKENS.md): a card with
 * its heading and an action, a segmented group (role=group, aria-pressed), a chip group,
 * and the hint line.
 */
import type { ReactNode } from 'react'

import { cn } from '@/lib/utils'

export const CARD = 'flex flex-col gap-3 rounded-xl border border-border bg-card p-[18px]'
export const HINT = 'text-[12.5px] leading-[1.45] text-muted-foreground'
const SEGMENT = 'h-[38px] rounded-lg px-3.5 text-sm font-medium text-muted-foreground transition-colors disabled:opacity-50'
const SEGMENT_ON = 'bg-secondary text-foreground ring-1 ring-inset ring-border'
const CHIP = 'h-[38px] rounded-full border border-border px-3.5 text-sm text-muted-foreground transition-colors'
const CHIP_ON = 'border-accent bg-accent/15 text-foreground'

export function EditorCard({ label, action, children }: { label: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section aria-label={label} className={CARD}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-[15px] font-semibold text-foreground">{label}</h2>
        {action}
      </div>
      {children}
    </section>
  )
}

export interface Choice<T extends string | number> {
  value: T
  label: string
  disabled?: boolean
}

interface GroupProps<T extends string | number> {
  label: string
  choices: ReadonlyArray<Choice<T>>
  value: T | null
  onChange: (value: T) => void
}

/** A segmented control: one choice pressed. */
export function Segmented<T extends string | number>({ label, choices, value, onChange }: GroupProps<T>) {
  return (
    <div role="group" aria-label={label} className="flex w-fit flex-wrap rounded-xl border border-border bg-background/60 p-[3px]">
      {choices.map((choice) => (
        <button
          key={String(choice.value)}
          type="button"
          aria-pressed={value === choice.value}
          disabled={choice.disabled}
          className={cn(SEGMENT, value === choice.value && SEGMENT_ON)}
          onClick={() => onChange(choice.value)}
        >
          {choice.label}
        </button>
      ))}
    </div>
  )
}

/** A row of pill chips: one choice pressed. */
export function ChipGroup<T extends string | number>({ label, choices, value, onChange }: GroupProps<T>) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1.5">
      {choices.map((choice) => (
        <button
          key={String(choice.value)}
          type="button"
          aria-pressed={value === choice.value}
          disabled={choice.disabled}
          className={cn(CHIP, value === choice.value && CHIP_ON)}
          onClick={() => onChange(choice.value)}
        >
          {choice.label}
        </button>
      ))}
    </div>
  )
}

export function Hint({ children }: { children: ReactNode }) {
  return <p className={cn(HINT, 'm-0')}>{children}</p>
}
