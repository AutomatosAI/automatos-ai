/**
 * SystemStrip — quiet diagnostics row at the bottom of the Deliverables feed.
 *
 * Heartbeats are agent self-status, not user-facing output. We hide them from
 * the main feed and surface a single counter strip instead. Click "Open in
 * Explorer" to drill into the full filterable grid where they remain visible.
 */

'use client'

import { useMemo } from 'react'
import { useRouter } from 'next/navigation'
import { ArrowRight, Activity } from 'lucide-react'

import { cn } from '@/lib/utils'
import {
  useDeliverables,
  type FilterState,
} from '@/hooks/use-deliverables-api'

const HEARTBEATS_TODAY: FilterState = {
  artifact_type: null,
  source_type: 'heartbeat',
  source_type_exclude: null,
  agent_id: null,
  date_range: 'today',
  search: '',
}

export interface SystemStripProps {
  className?: string
}

export function SystemStrip({ className }: SystemStripProps) {
  const router = useRouter()
  const { data, isLoading } = useDeliverables(HEARTBEATS_TODAY)

  const total = useMemo(() => data?.pages?.[0]?.total ?? 0, [data])

  if (isLoading || total === 0) return null

  return (
    <div
      className={cn(
        'flex flex-wrap items-center justify-between gap-3 rounded-lg border border-dashed border-border/60 bg-muted/10 px-4 py-3 text-xs',
        className,
      )}
    >
      <div className="flex items-center gap-2 text-muted-foreground">
        <Activity className="h-3.5 w-3.5" />
        <span>System diagnostics ·</span>
        <span className="font-medium text-foreground">
          {total} agent {total === 1 ? 'heartbeat' : 'heartbeats'} today
        </span>
      </div>
      <button
        type="button"
        onClick={() => router.push('/deliverables/explorer?source_type=heartbeat')}
        className="inline-flex items-center gap-1 text-muted-foreground transition hover:text-foreground"
      >
        Open in Explorer
        <ArrowRight className="h-3 w-3" />
      </button>
    </div>
  )
}
