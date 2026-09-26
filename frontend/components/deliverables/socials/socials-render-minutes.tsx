'use client'

/**
 * PRD-251 S1.1c — "Render minutes: used / quota this month". Every plan gets
 * Socials; plans differ only in render minutes (Basic 10, Pro 60, Business
 * 240). A plan with no quota (enterprise, the local edition) shows the minutes
 * used alone. Nothing shows until the reading loads, or if it cannot load.
 * When the last render in flight ends, the reading is fetched again: the render
 * has just booked its seconds.
 */
import { useEffect, useRef } from 'react'
import { Clapperboard } from 'lucide-react'

import { cn } from '@/lib/utils'
import { useSocialsUsage } from '@/hooks/use-socials-api'
import { formatRenderMinutes } from './socials-status'

export function SocialsRenderMinutes({ rendering = false }: { rendering?: boolean }) {
  const { data, refetch } = useSocialsUsage()
  const wasRendering = useRef(rendering)

  useEffect(() => {
    if (wasRendering.current && !rendering) void refetch()
    wasRendering.current = rendering
  }, [rendering, refetch])

  const minutes = data?.render_minutes
  if (!minutes) return null

  const noQuota = minutes.quota_minutes === null
  return (
    <p
      className={cn(
        'flex items-center gap-1.5 text-xs',
        minutes.exhausted ? 'text-destructive' : 'text-muted-foreground',
      )}
      data-testid="socials-render-minutes"
    >
      <Clapperboard className="h-3.5 w-3.5" aria-hidden />
      <span>
        Render minutes: {formatRenderMinutes(minutes.used_minutes, minutes.quota_minutes)} this month
        {noQuota && ' (no monthly quota)'}
        {minutes.exhausted && ' — used up until next month'}
      </span>
    </p>
  )
}
