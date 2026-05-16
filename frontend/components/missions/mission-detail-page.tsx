'use client'

import { useCallback, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { ArrowLeft, Pause, Play, X, Eye, Target, Check, XCircle, Save, RefreshCw, RotateCcw } from 'lucide-react'
import Link from 'next/link'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Textarea } from '@/components/ui/textarea'
import { PageHeader } from '@/components/shared'
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
// StatsBar removed — stats visible in budget bar + task nodes
import { MissionBudgetBar } from './mission-budget-bar'
import { MissionDAGCanvas } from './mission-dag-canvas'
import { MissionActivityFeed } from './mission-activity-feed'
import { TaskInspector } from './task-inspector'
import { HumanReviewPanel } from './human-review-panel'
import { MissionResultsPanel } from './mission-results-panel'
import { MissionFieldPanel } from './mission-field-panel'
import { useMission, usePauseMission, useResumeMission, useCancelMission, useApproveMission, useRejectMission, useSaveAsRoutine, useReplanMission, useRerunMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import { computeMissionStats, TERMINAL_RUN_STATES, RUN_STATE_CONFIG } from '@/types/missions'
import { useIsStudio } from '@/hooks/use-studio-theme'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Brain } from 'lucide-react'
import { toast } from 'sonner'

interface MissionDetailPageProps {
  missionId: string
}

// Derive a single-line headline from a (possibly long) mission brief:
// first sentence (.!?) or first line; hard cap at 100 chars with ellipsis.
function deriveMissionHeadline(text: string): string {
  const firstLine = text.split(/\r?\n/, 1)[0]?.trim() ?? text
  const sentenceEnd = firstLine.search(/[.!?](\s|$)/)
  let head = sentenceEnd > 0 ? firstLine.slice(0, sentenceEnd + 1) : firstLine
  if (head.length > 100) head = head.slice(0, 97).trimEnd() + '…'
  return head
}

export function MissionDetailPage({ missionId }: MissionDetailPageProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const showReview = searchParams?.get('tab') === 'review'
  const isStudio = useIsStudio()

  const { data: mission, isLoading } = useMission(missionId)
  const { selectedTaskId, setSelectedTaskId, planModifications, clearPlanModifications } = useMissionStore()

  const pauseMutation = usePauseMission()
  const resumeMutation = useResumeMission()
  const cancelMutation = useCancelMission()
  const approveMutation = useApproveMission()
  const rejectMutation = useRejectMission()

  const saveAsRoutineMutation = useSaveAsRoutine()
  const replanMutation = useReplanMission()
  const rerunMutation = useRerunMission()

  const [rightTab, setRightTab] = useState<'activity' | 'field'>('activity')
  const [showRejectInput, setShowRejectInput] = useState(false)
  const [rejectFeedback, setRejectFeedback] = useState('')
  const [showReplan, setShowReplan] = useState(false)
  const [replanNotes, setReplanNotes] = useState('')
  const [maxConcurrentOverride, setMaxConcurrentOverride] = useState<string | null>(null)
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

  // Strip a leading "Mission:" / "Mission " prefix, then derive a tight
  // headline so a multi-paragraph brief doesn't become the H1. Full text
  // remains available on hover (title attr) and in the brief panel below.
  const cleanedGoal = mission.goal.replace(/^\s*mission\s*[:\-—]?\s*/i, '').trim() || mission.goal
  const goalTitle = deriveMissionHeadline(cleanedGoal)
  const goalIsTruncated = goalTitle !== cleanedGoal
  const stateLabel = RUN_STATE_CONFIG[mission.state]?.label ?? mission.state

  // ── Stats strip (Studio) ───────────────────────────────────────
  const budgetPct =
    mission.token_budget_estimate && mission.token_budget_estimate > 0
      ? Math.round((mission.tokens_used / mission.token_budget_estimate) * 100)
      : null
  const budgetTone =
    budgetPct == null ? 'muted'
    : budgetPct >= 100 ? 'err'
    : budgetPct >= 80 ? 'err'
    : budgetPct >= 50 ? 'warn'
    : 'ok'
  // Spend mirrors MissionBudgetBar — $4/M tokens placeholder.
  const spendUsd = (mission.tokens_used / 1_000_000) * 4
  const fmtTokens = (n: number) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
    return String(n)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      {isStudio ? (
        <div className="cc-page" style={{ height: 'auto', paddingBottom: 18, gap: 14 }}>
          <Link
            href="/assignments?tab=missions"
            className="inline-flex items-center gap-1.5 text-xs"
            style={{
              color: 'hsl(var(--muted-foreground))',
              fontFamily: 'var(--font-geist-mono, monospace)',
              letterSpacing: '0.04em',
              width: 'fit-content',
            }}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back to Missions
          </Link>

          <div className="cc-headrow">
            <div className="cc-head">
              <p className="cc-eyebrow">Operations · Mission {missionId.slice(0, 8)}</p>
              <h1 className="cc-h1" style={{ maxWidth: '90ch' }} title={goalIsTruncated ? cleanedGoal : undefined}>{goalTitle}</h1>
              <p className="cc-sub">
                <b>{stateLabel}</b>
                {stats ? ` · ${stats.tasksDone}/${stats.taskCount} tasks` : ''}
                {stats && stats.tasksActive > 0 ? ` · ${stats.tasksActive} active` : ''}
                {stats && stats.elapsedMs > 0 ? ` · ${formatElapsed(stats.elapsedMs)} elapsed` : ''}
                {budgetPct != null ? ` · ${budgetPct}% budget` : ''}
              </p>
            </div>
            <div className="cc-actions">
              {mission.state === 'running' && (
                <button
                  type="button"
                  className="cc-btn"
                  onClick={() => pauseMutation.mutate(missionId, {
                    onError: (err) => toast.error(err.message),
                  })}
                  disabled={pauseMutation.isLoading}
                >
                  <Pause style={{ width: 12, height: 12 }} />
                  Pause
                </button>
              )}
              {mission.state === 'paused' && (
                <button
                  type="button"
                  className="cc-btn"
                  onClick={() => resumeMutation.mutate(missionId, {
                    onError: (err) => toast.error(err.message),
                  })}
                  disabled={resumeMutation.isLoading}
                >
                  <Play style={{ width: 12, height: 12 }} />
                  Resume
                </button>
              )}
              {!isTerminal && (
                <button
                  type="button"
                  className="cc-btn"
                  style={{ color: 'hsl(var(--accent))', borderColor: 'hsl(var(--accent) / 0.4)' }}
                  onClick={() => cancelMutation.mutate(missionId, {
                    onError: (err) => toast.error(err.message),
                  })}
                  disabled={cancelMutation.isLoading}
                >
                  <X style={{ width: 12, height: 12 }} />
                  Cancel
                </button>
              )}
              {isTerminal && (
                <button
                  type="button"
                  className="cc-btn primary"
                  disabled={rerunMutation.isLoading}
                  onClick={() =>
                    rerunMutation.mutate(
                      { goal: mission.goal, config: mission.config ?? undefined },
                      {
                        onSuccess: (newMission) => {
                          toast.success('New mission created from same goal')
                          router.push(`/missions/${newMission.id}` as any)
                        },
                        onError: (err) => toast.error(err.message),
                      },
                    )
                  }
                >
                  <RefreshCw style={{ width: 12, height: 12 }} />
                  {rerunMutation.isLoading ? 'Creating…' : 'Re-run'}
                </button>
              )}
            </div>
          </div>

          {/* Editorial stats strip — 5 cells */}
          <div className="cc-stats" style={{ gridTemplateColumns: 'repeat(5, minmax(0, 1fr))' }}>
            <div className="cell">
              <div className="l">TASKS</div>
              <div className="v">{stats ? `${stats.tasksDone}/${stats.taskCount}` : '—'}</div>
              <div className="delta">{stats?.tasksFailed ? `${stats.tasksFailed} failed` : 'on track'}</div>
            </div>
            <div className="cell">
              <div className="l">ACTIVE</div>
              <div className={`v ${stats && stats.tasksActive > 0 ? 'info' : ''}`}>
                {stats?.tasksActive ?? 0}
              </div>
              <div className="delta">running now</div>
            </div>
            <div className="cell">
              <div className="l">TOKENS</div>
              <div className="v">{fmtTokens(mission.tokens_used)}</div>
              <div className="delta">
                {mission.token_budget_estimate
                  ? `of ${fmtTokens(mission.token_budget_estimate)}`
                  : 'no budget set'}
              </div>
            </div>
            <div className="cell">
              <div className="l">BUDGET</div>
              <div className={`v ${budgetTone === 'muted' ? '' : budgetTone}`}>
                {budgetPct != null ? `${budgetPct}%` : '—'}
              </div>
              <div className="delta">
                {budgetPct == null ? 'not capped'
                 : budgetPct >= 100 ? 'over budget'
                 : budgetPct >= 80 ? 'tight'
                 : budgetPct >= 50 ? 'watching'
                 : 'healthy'}
              </div>
            </div>
            <div className="cell">
              <div className="l">SPEND</div>
              <div className="v">${spendUsd.toFixed(2)}</div>
              <div className="delta">~$4 / M tok</div>
            </div>
          </div>
        </div>
      ) : (
      <div className="p-4 md:p-6 border-b border-border space-y-4">
        {/* Back link */}
        <Link
          href="/assignments?tab=missions"
          className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Missions
        </Link>

        {/* Title row */}
        <PageHeader
          title="Mission"
          titleAccent={(mission.config as Record<string, unknown>)?.name as string || mission.goal.split(':')[0]}
          eyebrow={`Operations · mission ${missionId.slice(0, 8)}`}
          lede={
            [
              mission.state,
              stats ? `${stats.tasksDone}/${stats.taskCount} tasks` : null,
              stats && stats.elapsedMs > 0 ? `${formatElapsed(stats.elapsedMs)} elapsed` : null,
            ]
              .filter(Boolean)
              .join(' · ')
          }
          actions={
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

            {/* Replan — failed or cancelled missions */}
            {(mission.state === 'failed' || mission.state === 'cancelled') && (
              <Dialog open={showReplan} onOpenChange={setShowReplan}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
                    Replan
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle>Replan Mission</DialogTitle>
                    <DialogDescription>
                      Generate new tasks for the failed parts of this mission.
                      Completed work is preserved.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-2 py-2">
                    <Label htmlFor="replan-notes">Guidance (optional)</Label>
                    <Textarea
                      id="replan-notes"
                      placeholder="Any notes for the replanner, e.g. 'use a different approach for step 3'..."
                      value={replanNotes}
                      onChange={(e) => setReplanNotes(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowReplan(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      size="sm"
                      disabled={replanMutation.isLoading}
                      onClick={() =>
                        replanMutation.mutate(
                          {
                            id: missionId,
                            notes: replanNotes.trim() || undefined,
                          },
                          {
                            onSuccess: () => {
                              toast.success('Mission replanned — new tasks generated')
                              setShowReplan(false)
                              setReplanNotes('')
                            },
                            onError: (err) => toast.error(err.message),
                          },
                        )
                      }
                    >
                      {replanMutation.isLoading ? 'Replanning...' : 'Replan'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            )}

            {/* Re-run — any terminal mission */}
            {isTerminal && (
              <Button
                variant="outline"
                size="sm"
                disabled={rerunMutation.isLoading}
                onClick={() =>
                  rerunMutation.mutate(
                    {
                      goal: mission.goal,
                      config: mission.config ?? undefined,
                    },
                    {
                      onSuccess: (newMission) => {
                        toast.success('New mission created from same goal')
                        router.push(`/missions/${newMission.id}` as any)
                      },
                      onError: (err) => toast.error(err.message),
                    },
                  )
                }
              >
                <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                {rerunMutation.isLoading ? 'Creating...' : 'Re-run'}
              </Button>
            )}
            </div>
          }
        />

        {/* Stats are shown inline on the budget bar and task nodes */}

        {/* Budget bar */}
        {mission.token_budget_estimate != null && mission.token_budget_estimate > 0 && (
          <MissionBudgetBar
            tokensUsed={mission.tokens_used}
            tokenBudgetEstimate={mission.token_budget_estimate}
            missionState={mission.state}
            onResume={() => resumeMutation.mutate(missionId, {
              onSuccess: () => toast.success('Mission resumed'),
              onError: (err) => toast.error(err.message),
            })}
            isResuming={resumeMutation.isLoading}
          />
        )}
      </div>
      )}

      {/* Plan approval bar */}
      {mission.state === 'awaiting_approval' && (
        <div className="px-4 md:px-6 py-3 border-b border-primary/30 bg-primary/5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 text-sm text-primary">
                <Eye className="w-4 h-4 shrink-0" />
                <span>Review the plan below, then approve or reject</span>
              </div>
              {/* Parallel info + budget estimate */}
              <div className="flex items-center gap-3 mt-1.5 text-[11px] text-muted-foreground">
                {mission.complexity_tier && (
                  <span>Complexity: <span className="font-medium text-foreground">{mission.complexity_tier}</span></span>
                )}
                {mission.token_budget_estimate != null && mission.token_budget_estimate > 0 && (
                  <span>Est. tokens: <span className="font-medium text-foreground">{mission.token_budget_estimate.toLocaleString()}</span></span>
                )}
                {Array.isArray(mission.parallel_groups) && mission.parallel_groups.length > 0 && (
                  <span>Parallel groups: <span className="font-medium text-foreground">{mission.parallel_groups.join(', ')}</span></span>
                )}
                {mission.has_synthesis_tasks && (
                  <span className="text-agent">Has synthesis tasks</span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {/* Max concurrent override */}
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-muted-foreground whitespace-nowrap">Parallel:</span>
                <Select
                  value={maxConcurrentOverride ?? String(mission.max_concurrent)}
                  onValueChange={(v) => setMaxConcurrentOverride(v)}
                >
                  <SelectTrigger className="h-7 w-16 text-xs !rounded-md">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">1</SelectItem>
                    <SelectItem value="2">2</SelectItem>
                    <SelectItem value="3">3</SelectItem>
                  </SelectContent>
                </Select>
              </div>
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
                className="bg-success hover:bg-success/80 text-white"
                disabled={approveMutation.isLoading}
                onClick={() => {
                  const hasModifications =
                    Object.keys(planModifications.task_overrides).length > 0 ||
                    Object.keys(planModifications.agent_overrides).length > 0 ||
                    planModifications.notes.length > 0

                  const overrideVal = maxConcurrentOverride != null
                    ? parseInt(maxConcurrentOverride, 10)
                    : undefined
                  const hasOverride = overrideVal != null && overrideVal !== mission.max_concurrent

                  approveMutation.mutate(
                    {
                      id: missionId,
                      body: hasModifications || hasOverride
                        ? {
                            ...(hasModifications ? { modifications: planModifications } : {}),
                            ...(hasOverride ? { max_concurrent_override: overrideVal } : {}),
                          }
                        : undefined,
                    },
                    {
                      onSuccess: () => {
                        toast.success('Plan approved — mission is running')
                        clearPlanModifications()
                        setMaxConcurrentOverride(null)
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

      {/* Main content — explicit min-height keeps the DAG/right-panel row
          from collapsing when a tab renders a short empty state (e.g.
          Field with no patterns yet). The parent chain has no concrete
          height (MainLayout's <main> is auto), so we anchor on vh. */}
      <div className="flex-1" style={{ minHeight: '60vh' }}>
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

          {/* Right panel: Results / Review / Activity Feed */}
          <ResizablePanel defaultSize={showReview || isReviewable || isTerminal ? 50 : 40} minSize={25}>
            {(showReview || isReviewable) ? (
              <HumanReviewPanel
                missionId={missionId}
                tasks={mission.tasks}
                className="h-full"
              />
            ) : isTerminal ? (
              <div className="h-full flex flex-col overflow-hidden">
                <div className="p-3 border-b border-border flex items-center gap-1 shrink-0">
                  <button
                    onClick={() => setRightTab('activity')}
                    className={cn(
                      'px-2.5 py-1 text-xs font-medium rounded-md transition-colors',
                      rightTab === 'activity'
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    Results
                  </button>
                  <button
                    onClick={() => setRightTab('field')}
                    className={cn(
                      'px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1',
                      rightTab === 'field'
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    <Brain className="w-3 h-3" />
                    Field
                  </button>
                </div>
                {rightTab === 'field' ? (
                  <MissionFieldPanel missionId={missionId} className="flex-1 min-h-0 overflow-y-auto" />
                ) : (
                  <MissionResultsPanel
                    mission={mission}
                    className="flex-1 min-h-0 overflow-y-auto"
                  />
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col overflow-hidden">
                <div className="p-3 border-b border-border flex items-center gap-1 shrink-0">
                  <button
                    onClick={() => setRightTab('activity')}
                    className={cn(
                      'px-2.5 py-1 text-xs font-medium rounded-md transition-colors',
                      rightTab === 'activity'
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    Activity
                  </button>
                  <button
                    onClick={() => setRightTab('field')}
                    className={cn(
                      'px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1',
                      rightTab === 'field'
                        ? 'bg-muted text-foreground'
                        : 'text-muted-foreground hover:text-foreground',
                    )}
                  >
                    <Brain className="w-3 h-3" />
                    Field
                  </button>
                </div>
                {rightTab === 'field' ? (
                  <MissionFieldPanel missionId={missionId} className="flex-1 min-h-0 overflow-y-auto" />
                ) : (
                  <MissionActivityFeed
                    events={mission.recent_events}
                    className="flex-1 min-h-0 overflow-y-auto"
                  />
                )}
              </div>
            )}
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  )
}
