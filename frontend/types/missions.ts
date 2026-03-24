/**
 * Mission Control Types — PRD-82A Sequential Mission Coordinator
 *
 * Aligned 1:1 with backend Pydantic models in orchestrator/api/missions.py
 * and enums in orchestrator/core/models/orchestration_enums.py.
 */

// ── Enums (match orchestration_enums.py exactly) ──────────────

export type RunState =
  | 'pending'
  | 'planning'
  | 'awaiting_approval'
  | 'running'
  | 'paused'
  | 'verifying'
  | 'awaiting_human'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type TaskState =
  | 'pending'
  | 'queued'
  | 'assigned'
  | 'running'
  | 'completed'     // NOT terminal — awaiting verification
  | 'verifying'
  | 'verified'      // BLOCKED not terminal — human can reject
  | 'failed'
  | 'skipped'
  | 'stalled'
  | 'retrying'

export type StateType = 'initial' | 'active' | 'blocked' | 'terminal'

// ── API Response Types (match Pydantic models) ────────────────

export interface MissionResponse {
  id: string
  workspace_id: string
  goal: string
  state: RunState
  state_type: StateType
  plan: Record<string, unknown> | null
  config: Record<string, unknown> | null
  output_summary: Record<string, unknown> | null
  token_budget_estimate: number | null
  tokens_used: number
  max_retries: number
  created_by: string
  started_at: string | null
  completed_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface MissionDetailResponse extends MissionResponse {
  tasks: TaskResponse[]
  recent_events: EventResponse[]
}

export interface TaskResponse {
  id: string
  title: string
  description: string | null
  task_type: string | null
  sequence_number: number
  agent_role: string | null
  state: TaskState
  state_type: StateType
  assigned_agent_id: number | null
  attempt_number: number
  tokens_used: number
  failure_reason_code: string | null
  failure_detail: string | null
  output_excerpt: string | null
  output: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string | null
}

export interface EventResponse {
  id: string
  event_type: string
  actor_type: string
  actor_id: string | null
  old_state: string | null
  new_state: string | null
  task_id: string | null
  created_at: string | null
}

export interface MissionListResponse {
  missions: MissionResponse[]
  total: number
  limit: number
  offset: number
}

// ── Request Types ─────────────────────────────────────────────

export interface MissionCreateRequest {
  goal: string
  config?: Record<string, unknown>
}

export interface MissionApproveRequest {
  modifications?: {
    task_overrides?: Record<string, Record<string, unknown>>
    notes?: string
    agent_overrides?: Record<string, number>
  }
}

export interface MissionRejectRequest {
  reason: string
}

export interface MissionReviewRequest {
  verdict: 'accept' | 'reject'
  task_feedback?: Record<string, string>
  feedback?: string
}

export interface SaveAsRoutineRequest {
  name: string
  description?: string
  tags?: string[]
}

export interface SaveAsRoutineResponse {
  template_id: string
  name: string
  task_count: number
}

// ── Frontend Display Helpers ──────────────────────────────────

export const DONE_TASK_STATES: readonly TaskState[] = ['verified', 'failed', 'skipped'] as const
export const ACTIVE_TASK_STATES: readonly TaskState[] = ['assigned', 'running', 'completed', 'verifying', 'retrying'] as const

export const TERMINAL_RUN_STATES: readonly RunState[] = ['completed', 'failed', 'cancelled'] as const

export interface MissionStats {
  taskCount: number
  tasksDone: number
  tasksActive: number
  tasksFailed: number
  tokensUsed: number
  elapsedMs: number
}

export function computeMissionStats(mission: MissionDetailResponse): MissionStats {
  const tasks = mission.tasks

  return {
    taskCount: tasks.length,
    tasksDone: tasks.filter(t => (DONE_TASK_STATES as readonly string[]).includes(t.state)).length,
    tasksActive: tasks.filter(t => (ACTIVE_TASK_STATES as readonly string[]).includes(t.state)).length,
    tasksFailed: tasks.filter(t => t.state === 'failed').length,
    tokensUsed: tasks.reduce((sum, t) => sum + t.tokens_used, 0),
    elapsedMs: mission.started_at
      ? Date.now() - new Date(mission.started_at).getTime()
      : 0,
  }
}

// ── RunState display config ───────────────────────────────────

export const RUN_STATE_CONFIG: Record<RunState, {
  label: string
  color: string
  bgClass: string
  pulse?: boolean
}> = {
  pending: {
    label: 'Pending',
    color: 'text-muted-foreground',
    bgClass: 'bg-muted',
  },
  planning: {
    label: 'Planning',
    color: 'text-[hsl(var(--info))]',
    bgClass: 'bg-[hsl(var(--info))]/10 border-[hsl(var(--info))]/20',
  },
  awaiting_approval: {
    label: 'Needs Approval',
    color: 'text-[hsl(var(--warning))]',
    bgClass: 'bg-[hsl(var(--warning))]/10 border-[hsl(var(--warning))]/20',
  },
  running: {
    label: 'Running',
    color: 'text-primary',
    bgClass: 'bg-primary/10 border-primary/20',
    pulse: true,
  },
  paused: {
    label: 'Paused',
    color: 'text-muted-foreground',
    bgClass: 'bg-muted',
  },
  verifying: {
    label: 'Verifying',
    color: 'text-[hsl(var(--info))]',
    bgClass: 'bg-[hsl(var(--info))]/10 border-[hsl(var(--info))]/20',
    pulse: true,
  },
  awaiting_human: {
    label: 'Needs Review',
    color: 'text-[hsl(var(--warning))]',
    bgClass: 'bg-[hsl(var(--warning))]/10 border-[hsl(var(--warning))]/20',
    pulse: true,
  },
  completed: {
    label: 'Completed',
    color: 'text-[hsl(var(--success))]',
    bgClass: 'bg-[hsl(var(--success))]/10 border-[hsl(var(--success))]/20',
  },
  failed: {
    label: 'Failed',
    color: 'text-destructive',
    bgClass: 'bg-destructive/10 border-destructive/20',
  },
  cancelled: {
    label: 'Cancelled',
    color: 'text-muted-foreground',
    bgClass: 'bg-muted',
  },
}

// ── TaskState display config (for DAG nodes) ──────────────────

export const TASK_STATE_CONFIG: Record<TaskState, {
  label: string
  color: string
  borderClass: string
  bgClass: string
  icon: string        // Lucide icon name
  animate?: string    // Tailwind animation class
}> = {
  pending: {
    label: 'Pending',
    color: 'text-muted-foreground',
    borderClass: 'border-muted-foreground/20',
    bgClass: 'bg-muted/5',
    icon: 'Circle',
  },
  queued: {
    label: 'Queued',
    color: 'text-muted-foreground',
    borderClass: 'border-muted-foreground/30',
    bgClass: 'bg-muted/5',
    icon: 'Clock',
  },
  assigned: {
    label: 'Assigned',
    color: 'text-[hsl(var(--info))]',
    borderClass: 'border-[hsl(var(--info))]/40',
    bgClass: 'bg-[hsl(var(--info))]/5',
    icon: 'UserCheck',
  },
  running: {
    label: 'Running',
    color: 'text-primary',
    borderClass: 'border-primary/60',
    bgClass: 'bg-primary/5',
    icon: 'Loader2',
    animate: 'animate-spin',
  },
  completed: {
    label: 'Completed',
    color: 'text-[hsl(var(--info))]',
    borderClass: 'border-[hsl(var(--info))]/40',
    bgClass: 'bg-[hsl(var(--info))]/5',
    icon: 'CheckCircle2',
  },
  verifying: {
    label: 'Verifying',
    color: 'text-[hsl(var(--info))]',
    borderClass: 'border-[hsl(var(--info))]/40',
    bgClass: 'bg-[hsl(var(--info))]/5',
    icon: 'Search',
    animate: 'animate-pulse',
  },
  verified: {
    label: 'Verified',
    color: 'text-[hsl(var(--success))]',
    borderClass: 'border-[hsl(var(--success))]/60',
    bgClass: 'bg-[hsl(var(--success))]/10',
    icon: 'ShieldCheck',
  },
  failed: {
    label: 'Failed',
    color: 'text-destructive',
    borderClass: 'border-destructive/40',
    bgClass: 'bg-destructive/5',
    icon: 'XCircle',
  },
  skipped: {
    label: 'Skipped',
    color: 'text-muted-foreground/50',
    borderClass: 'border-muted/20',
    bgClass: 'bg-muted/5',
    icon: 'SkipForward',
  },
  stalled: {
    label: 'Stalled',
    color: 'text-[hsl(var(--warning))]',
    borderClass: 'border-[hsl(var(--warning))]/40',
    bgClass: 'bg-[hsl(var(--warning))]/5',
    icon: 'AlertTriangle',
  },
  retrying: {
    label: 'Retrying',
    color: 'text-[hsl(var(--warning))]',
    borderClass: 'border-[hsl(var(--warning))]/40',
    bgClass: 'bg-[hsl(var(--warning))]/5',
    icon: 'RefreshCw',
    animate: 'animate-spin',
  },
}
