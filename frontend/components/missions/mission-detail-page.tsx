'use client'

import { useCallback, useMemo, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { ArrowLeft, Pause, Play, X, Eye, Target, Check, XCircle, Save } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
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
import { useMission, usePauseMission, useResumeMission, useCancelMission, useApproveMission, useRejectMission, useSaveAsRoutine } from '@/hooks/use-missions-api'
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
  const { selectedTaskId, setSelectedTaskId, planModifications, clearPlanModifications } = useMissionStore()

  const pauseMutation = usePauseMission()
  const resumeMutation = useResumeMission()
  const cancelMutation = useCancelMission()
  const approveMutation = useApproveMission()
  const rejectMutation = useRejectMission()

  const saveAsRoutineMutation = useSaveAsRoutine()

  const [showRejectInput, setShowRejectInput] = useState(false)
  const [rejectFeedback, setRejectFeedback] = useState('')
  const [showSaveRoutine, setShowSaveRoutine] = useState(false)
  const [routineName, setRoutineName] = useState('')
  const [routineDescription, setRoutineDescription] = useState('')
  const [routineTags, setRoutineTags] = useState('')

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
            {mission.state === 'completed' && (
              <Dialog open={showSaveRoutine} onOpenChange={setShowSaveRoutine}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Save className="w-3.5 h-3.5 mr-1.5" />
                    Save as Routine
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle>Save as Routine</DialogTitle>
                    <DialogDescription>
                      Save this mission&apos;s task structure as a reusable routine template.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 py-2">
                    <div className="space-y-2">
                      <Label htmlFor="routine-name">Name</Label>
                      <Input
                        id="routine-name"
                        placeholder="e.g. Weekly Research Report"
                        value={routineName}
                        onChange={(e) => setRoutineName(e.target.value)}
                        maxLength={255}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="routine-description">Description (optional)</Label>
                      <Textarea
                        id="routine-description"
                        placeholder="Describe what this routine does..."
                        value={routineDescription}
                        onChange={(e) => setRoutineDescription(e.target.value)}
                        rows={3}
                        maxLength={2000}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="routine-tags">Tags (optional, comma-separated)</Label>
                      <Input
                        id="routine-tags"
                        placeholder="research, report, weekly"
                        value={routineTags}
                        onChange={(e) => setRoutineTags(e.target.value)}
                      />
                    </div>
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowSaveRoutine(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      disabled={!routineName.trim() || saveAsRoutineMutation.isLoading}
                      onClick={() => {
                        const tags = routineTags
                          .split(',')
                          .map((t) => t.trim())
                          .filter(Boolean)

                        saveAsRoutineMutation.mutate(
                          {
                            id: missionId,
                            body: {
                              name: routineName.trim(),
                              description: routineDescription.trim() || undefined,
                              tags: tags.length > 0 ? tags : undefined,
                            },
                          },
                          {
                            onSuccess: (data) => {
                              toast.success(
                                `Routine "${data.name}" saved (${data.task_count} tasks)`,
                              )
                              setShowSaveRoutine(false)
                              setRoutineName('')
                              setRoutineDescription('')
                              setRoutineTags('')
                            },
                            onError: (err) => toast.error(err.message),
                          },
                        )
                      }}
                    >
                      {saveAsRoutineMutation.isLoading ? 'Saving...' : 'Save Routine'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
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

      {/* Plan approval bar */}
      {mission.state === 'awaiting_approval' && (
        <div className="px-4 md:px-6 py-3 border-b border-orange-500/30 bg-orange-500/5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-orange-400">
              <Eye className="w-4 h-4 shrink-0" />
              <span>Review the plan below, then approve or reject</span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Button
                variant="outline"
                size="sm"
                className="text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={() => setShowRejectInput((prev) => !prev)}
              >
                <XCircle className="w-3.5 h-3.5 mr-1.5" />
                Reject
              </Button>
              <Button
                size="sm"
                className="bg-green-600 hover:bg-green-700 text-white"
                disabled={approveMutation.isLoading}
                onClick={() => {
                  const hasModifications =
                    Object.keys(planModifications.task_overrides).length > 0 ||
                    Object.keys(planModifications.agent_overrides).length > 0 ||
                    planModifications.notes.length > 0

                  approveMutation.mutate(
                    {
                      id: missionId,
                      body: hasModifications
                        ? { modifications: planModifications }
                        : undefined,
                    },
                    {
                      onSuccess: () => {
                        toast.success('Plan approved — mission is running')
                        clearPlanModifications()
                      },
                      onError: (err) => toast.error(err.message),
                    },
                  )
                }}
              >
                <Check className="w-3.5 h-3.5 mr-1.5" />
                Approve
              </Button>
            </div>
          </div>

          {showRejectInput && (
            <div className="mt-3 flex gap-2">
              <Textarea
                placeholder="Why should this plan be revised?"
                value={rejectFeedback}
                onChange={(e) => setRejectFeedback(e.target.value)}
                className="min-h-[60px] text-sm bg-background/50"
                rows={2}
              />
              <Button
                variant="destructive"
                size="sm"
                className="self-end"
                disabled={rejectMutation.isLoading || !rejectFeedback.trim()}
                onClick={() =>
                  rejectMutation.mutate(
                    { id: missionId, body: { reason: rejectFeedback.trim() } },
                    {
                      onSuccess: () => {
                        toast.success('Plan rejected — provide more details')
                        clearPlanModifications()
                        setRejectFeedback('')
                        setShowRejectInput(false)
                      },
                      onError: (err) => toast.error(err.message),
                    },
                  )
                }
              >
                Send
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* DAG panel */}
          <ResizablePanel defaultSize={showReview || isReviewable ? 50 : 60} minSize={30}>
            <div className="relative h-full">
              <MissionDAGCanvas
                tasks={mission.tasks}
                mode={mission.state === 'awaiting_approval' ? 'plan' : isReviewable ? 'review' : 'execution'}
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
