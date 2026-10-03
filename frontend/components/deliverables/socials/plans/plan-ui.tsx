'use client'

/**
 * PRD-251B US-B207 — the Plan page's small shared pieces: a step's heading and lead, and the
 * summary box (Plan.dc.html).
 */
import type { ReactNode } from 'react'

export function PlanStepHeading({ title, lead, action }: { title: string; lead: string; action?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div className="flex flex-col gap-1">
        <h2 className="text-[20px] font-semibold text-foreground">{title}</h2>
        <p className="m-0 max-w-[62ch] text-[13.5px] leading-[1.5] text-muted-foreground">{lead}</p>
      </div>
      {action}
    </div>
  )
}

export function PlanSummaryBox({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section aria-label={label} className="flex flex-col gap-2 rounded-xl border border-border bg-card/60 p-4 text-sm leading-[1.6]">
      {children}
    </section>
  )
}
