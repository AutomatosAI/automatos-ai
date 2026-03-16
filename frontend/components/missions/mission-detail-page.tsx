'use client'

import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Pause, Play, X, Eye, Target } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import { StatsBar } from '@/components/shared/stats-bar'
import { MissionStatusBadge } from './mission-status-badge'
import { MissionDAGCanvas } from './mission-dag-canvas'
import { MissionActivityFeed } from './mission-activity-feed'
import { TaskInspector } from './task-inspector'
import { HumanReviewPanel } from './human-review-panel'
import { useMission, usePauseMission, useResumeMission, useCancelMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import { computeMissionStats, TERMINAL_RUN_STATES } from '@/types/missions'
import { Activity, ListChecks, Clock, Coins } from 'lucide-react'
import { toast } from 'sonner'

interface MissionDetailPageProps {
  missionId: string
}

export function MissionDetailPage({ missionId }: MissionDetailPageProps) {
  const searchParams = useSearchParams()
  const showReview = searchParams?.get('tab') === 'review'

  const { data: mission, isLoading } = useMission(missionId)
  const { selectedTaskId, setSelectedTaskId } = useMissionStore()

  const pauseMutation = usePauseMission()
  const resumeMutation = useResumeMission()
  const cancelMutation = useCancelMission()

  const stats = useMemo(
    () => (mission ? computeMissionStats(mission) : null),
    [mission],
  )

  const selectedTask = useMemo(
    () => mission?.tasks.find((t) => t.id === selectedTaskId) ?? null,
    [mission, selectedTaskId],
  )

  const handleTaskSelect = useCallback(
    (taskId: string) => setSelectedTaskId(taskId),
    [setSelectedTaskId],
  )

  const handleCloseInspector = useCallback(
    () => setSelectedTaskId(null),
    [setSelectedTaskId],
  )

  const formatElapsed = (ms: number) => {
    const minutes = Math.floor(ms / 60_000)
    if (minutes < 60) return `${minutes}m`
    const hours = Math.floor(minutes / 60)
    return `${hours}h ${minutes % 60}m`
  }

  if (isLoading || !mission) {
    return (
      <div className="space-y-6 p-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  const isTerminal = (TERMINAL_RUN_STATES as readonly string[]).includes(mission.state)
  const isReviewable = mission.state === 'awaiting_human'

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 md:p-6 border-b border-border space-y-4">
        {/* Back link */}
        <Link
          href="/activity?tab=missions"
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Missions
        </Link>

        {/* Title row */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-lg font-semibold leading-tight line-clamp-2">
              {mission.goal}
            </h1>
            <div className="flex items-center gap-3 mt-2">
              <MissionStatusBadge state={mission.state} />
              <span className="text-xs text-muted-foreground">
                {stats?.tasksDone}/{stats?.taskCount} tasks
              </span>
              {stats && stats.elapsedMs > 0 && (
                <span className="text-xs text-muted-foreground">
                  {formatElapsed(stats.elapsedMs)} elapsed
                </span>
              )}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2 shrink-0">
            {mission.state === 'running' && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => pauseMutation.mutate(missionId, {
                  onError: (err) => toast.error(err.message),
                })}
                disabled={pauseMutation.isLoading}
              >
                <Pause className="w-3.5 h-3.5 mr-1.5" />
                Pause
              </Button>
            )}
            {mission.state === 'paused' && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => resumeMutation.mutate(missionId, {
                  onError: (err) => toast.error(err.message),
                })}
                disabled={resumeMutation.isLoading}
              >
                <Play className="w-3.5 h-3.5 mr-1.5" />
                Resume
              </Button>
            )}
            {!isTerminal && (
              <Button
                variant="outline"
                size="sm"
                className="text-destructive border-destructive/30"
                onClick={() => cancelMutation.mutate(missionId, {
                  onError: (err) => toast.error(err.message),
                })}
                disabled={cancelMutation.isLoading}
              >
                <X className="w-3.5 h-3.5 mr-1.5" />
                Cancel
              </Button>
            )}
          </div>
        </div>

        {/* Stats bar */}
        {stats && (
          <StatsBar
            stats={[
              {
                label: 'Tasks Done',
                value: `${stats.tasksDone}/${stats.taskCount}`,
                icon: ListChecks,
                iconColor: 'text-[hsl(var(--success))]',
              },
              {
                label: 'Active Now',
                value: stats.tasksActive,
                icon: Activity,
                iconColor: 'text-primary',
              },
              {
                label: 'Elapsed',
                value: stats.elapsedMs > 0 ? formatElapsed(stats.elapsedMs) : '-',
                icon: Clock,
                iconColor: 'text-[hsl(var(--info))]',
              },
              {
                label: 'Tokens Used',
                value: stats.tokensUsed.toLocaleString(),
                icon: Coins,
                iconColor: 'text-[hsl(var(--warning))]',
              },
            ]}
          />
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* DAG panel */}
          <ResizablePanel defaultSize={showReview || isReviewable ? 50 : 60} minSize={30}>
            <div className="relative h-full">
              <MissionDAGCanvas
                tasks={mission.tasks}
                mode={isReviewable ? 'review' : 'execution'}
                selectedTaskId={selectedTaskId}
                onTaskSelect={handleTaskSelect}
                className="h-full"
              />

              {/* Task inspector overlay */}
              {selectedTask && (
                <TaskInspector
                  task={selectedTask}
                  onClose={handleCloseInspector}
                  className="absolute top-4 right-4 w-80 max-h-[80%] z-10"
                />
              )}
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right panel: Activity Feed or Review */}
          <ResizablePanel defaultSize={showReview || isReviewable ? 50 : 40} minSize={25}>
            {(showReview || isReviewable) ? (
              <HumanReviewPanel
                missionId={missionId}
                tasks={mission.tasks}
                className="h-full"
              />
            ) : (
              <div className="h-full flex flex-col">
                <div className="p-3 border-b border-border">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Activity Feed
                  </h3>
                </div>
                <MissionActivityFeed
                  events={mission.recent_events}
                  className="flex-1"
                />
              </div>
            )}
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  )
}
