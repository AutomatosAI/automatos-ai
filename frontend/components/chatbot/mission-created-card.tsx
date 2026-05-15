'use client'

import { useRouter } from 'next/navigation'
import { Target, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { MissionStatusBadge } from '@/components/missions/mission-status-badge'
import { useMission } from '@/hooks/use-missions-api'

interface MissionCreatedCardProps {
  missionId: string
}

export function MissionCreatedCard({ missionId }: MissionCreatedCardProps) {
  const router = useRouter()
  const { data: mission, isLoading } = useMission(missionId)

  const planTasks = mission
    ? ((mission.plan as { tasks?: Array<Record<string, unknown>> })?.tasks ?? [])
    : []

  const handleReviewPlan = () => {
    router.push(`/missions/${missionId}` as any)
  }

  if (isLoading) {
    return (
      <div className="bg-card/50 backdrop-blur border border-border rounded-xl p-4 space-y-3 max-w-md">
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-5 rounded" />
          <Skeleton className="h-4 w-24" />
        </div>
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-8 w-28" />
      </div>
    )
  }

  if (!mission) return null

  const goalText = mission.goal.length > 120
    ? `${mission.goal.slice(0, 120)}...`
    : mission.goal

  return (
    <div className="bg-card/50 backdrop-blur border border-border rounded-xl p-4 space-y-3 max-w-md">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Target className="w-4 h-4 text-primary" />
        <span className="text-xs font-medium text-primary">Mission Created</span>
        <MissionStatusBadge state={mission.state} size="sm" />
      </div>

      {/* Goal */}
      <p className="text-sm text-foreground/80 leading-snug">{goalText}</p>

      {/* Task count */}
      {planTasks.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {planTasks.length} task{planTasks.length !== 1 ? 's' : ''} planned
        </p>
      )}

      {/* Review Plan CTA */}
      <Button
        variant="outline"
        size="sm"
        className="border-primary/30 text-primary hover:bg-primary/10"
        onClick={handleReviewPlan}
      >
        Review Plan
        <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
      </Button>
    </div>
  )
}
