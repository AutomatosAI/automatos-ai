'use client'

/**
 * PRD-268 §6.5.1, §6.12.4 (US-W01a) — Auto's Read, then the Suggested list beside it: one component for
 * both Command Center shells, with no fork.
 *
 * Studio's SummaryTab and Classic's ActivityPage each import it under AutosRead's name, so neither body
 * changes. On a wide screen the list sits beside the Read; on a narrow one, under it. With no open
 * suggestion the list draws nothing and the Read keeps the whole width.
 */
import { AutosRead } from '@/components/activity/autos-read'
import { SuggestedList } from './suggested-list'

export function CommandCenterRead({ period = '1d' }: { readonly period?: string }) {
  return (
    <div className="flex flex-col gap-4 xl:flex-row xl:items-start">
      <div className="min-w-0 flex-1">
        <AutosRead period={period} />
      </div>
      <SuggestedList className="xl:w-96 xl:shrink-0" />
    </div>
  )
}
