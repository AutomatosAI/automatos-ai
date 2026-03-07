'use client'

import { useRouter } from 'next/navigation'
import { RefreshCw, Pause, Play, Settings, Clock } from 'lucide-react'
import { formatDistanceToNow, format } from 'date-fns'
import { Button } from '@/components/ui/button'
import { ItemCard } from '@/components/shared/item-card'
import { StatusBadge } from '@/components/shared/status-badge'
import { useToggleHeartbeat } from '@/hooks/use-heartbeats-api'
import type { HeartbeatConfig } from '@/hooks/use-heartbeats-api'

// ─── Helpers ──────────────────────────────────────────────

function formatInterval(minutes: number): string {
  if (minutes < 60) return `Every ${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remaining = minutes % 60
  if (remaining === 0) return `Every ${hours}h`
  return `Every ${hours}h ${remaining}m`
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return 'Never'
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return 'Unknown'
  }
}

function formatNextRun(iso: string | null): string {
  if (!iso) return 'Not scheduled'
  try {
    return format(new Date(iso), 'HH:mm')
  } catch {
    return 'Unknown'
  }
}

// ─── Component ────────────────────────────────────────────

interface RoutineCardProps {
  heartbeat: HeartbeatConfig
  animationDelay?: number
}

export function RoutineCard({ heartbeat, animationDelay = 0 }: RoutineCardProps) {
  const router = useRouter()
  const toggleMutation = useToggleHeartbeat()

  const isActive = heartbeat.enabled

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    toggleMutation.mutate(heartbeat.agent_id)
  }

  const handleEdit = (e: React.MouseEvent) => {
    e.stopPropagation()
    router.push(`/agents?agent=${heartbeat.agent_id}`)
  }

  return (
    <ItemCard
      animationDelay={animationDelay}
      icon={
        <div className="w-10 h-10 rounded-lg bg-[hsl(var(--agent))]/10 flex items-center justify-center">
          <RefreshCw className="w-5 h-5 text-[hsl(var(--agent))]" />
        </div>
      }
      title={heartbeat.agent_name}
      subtitle={`Agent routine`}
      titleBadges={
        <StatusBadge
          status={isActive ? 'success' : 'warning'}
          dot
          size="sm"
        >
          {isActive ? 'Active' : 'Paused'}
        </StatusBadge>
      }
      description={heartbeat.prompt}
      meta={
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {formatInterval(heartbeat.interval_minutes)}
          </span>
          <span>
            Last ran {formatRelativeTime(heartbeat.last_run?.created_at ?? null)}
          </span>
          <span>
            Next: {formatNextRun(heartbeat.next_run_at)}
          </span>
        </div>
      }
      actions={
        <div className="flex items-center gap-2 w-full justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggle}
            disabled={toggleMutation.isLoading}
            className="text-xs"
          >
            {isActive ? (
              <>
                <Pause className="w-3.5 h-3.5 mr-1" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 mr-1" />
                Resume
              </>
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleEdit}
            className="text-xs"
          >
            <Settings className="w-3.5 h-3.5 mr-1" />
            Edit
          </Button>
        </div>
      }
    />
  )
}
