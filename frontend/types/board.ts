/**
 * Board types for the Kanban task management view (PRD-72 v2).
 * "Task" in UI = "Recipe" in backend. No backend rename.
 */

export type BoardStatus = 'inbox' | 'assigned' | 'in_progress' | 'review' | 'blocked' | 'done' | 'failed'

export type TaskPriority = 'urgent' | 'high' | 'medium' | 'low'

export type ReviewMode = 'human' | 'llm' | 'auto'

export interface BoardTask {
  id: string
  type: 'routine' | 'playbook' | 'task' | 'mission'
  name: string
  description?: string
  status: BoardStatus
  priority: TaskPriority
  tags: string[]
  assignee?: {
    agent_id: number
    agent_name: string
    agent_icon?: string | null
  }
  creator?: string
  due_date?: string
  review_mode: ReviewMode
  started_at?: string
  completed_at?: string
  duration_ms?: number
  step_progress?: { current: number; total: number }
  error_message?: string
  // PRD-161: dispatch retry counter. >0 means a prior attempt's lease expired
  // (worker crashed/hung) and the task was requeued → render an "unresponsive"
  // badge so the user sees the agent missed its ack deadline.
  attempts?: number
  report_id?: string
  source_id: string
  project_id?: number
  /** Full mission run UUID — set when the task was spawned by a mission.
   *  Used to cascade kanban approval back to the owning mission's
   *  awaiting_approval gate so the user only approves once. */
  mission_id?: string
  mission_name?: string
  parent_task_id?: string
  child_count?: number
  sla_deadline?: string
  blocked_at?: string
  blocked_reason?: string
  planning_data?: { playbook_id?: number; execution_id?: string; step_progress?: { current: number; total: number }; approval_action?: { type: string; post_id?: string; [key: string]: any } }
  result?: any
}

export interface BoardColumn {
  status: BoardStatus
  label: string
  count: number
  tasks: BoardTask[]
}

export interface BoardAgent {
  id: number
  name: string
  role?: string
  premium_icon?: string | null
  status: 'working' | 'idle' | 'offline'
  badge?: string // LEAD, INT, SPC
}

export const BOARD_COLUMNS: { status: BoardStatus; label: string }[] = [
  { status: 'inbox', label: 'Inbox' },
  { status: 'assigned', label: 'Assigned' },
  { status: 'in_progress', label: 'In Progress' },
  { status: 'review', label: 'Review' },
  { status: 'blocked', label: 'Blocked' },
  { status: 'done', label: 'Done' },
  { status: 'failed', label: 'Failed' },
]

export const PRIORITY_CONFIG: Record<TaskPriority, { label: string; color: string; cssVar: string }> = {
  urgent: { label: 'Urgent', color: 'hsl(var(--destructive))', cssVar: '--destructive' },
  high: { label: 'High', color: 'hsl(var(--warning))', cssVar: '--warning' },
  medium: { label: 'Medium', color: 'hsl(var(--info))', cssVar: '--info' },
  low: { label: 'Low', color: 'hsl(var(--muted-foreground))', cssVar: '--muted-foreground' },
}

export const STATUS_CONFIG: Record<BoardStatus, { label: string; dotColor: string; cssVar: string }> = {
  inbox: { label: 'Inbox', dotColor: 'bg-muted-foreground', cssVar: '--muted-foreground' },
  assigned: { label: 'Assigned', dotColor: 'bg-[hsl(var(--agent))]', cssVar: '--agent' },
  in_progress: { label: 'In Progress', dotColor: 'bg-[hsl(var(--info))]', cssVar: '--info' },
  review: { label: 'Review', dotColor: 'bg-[hsl(var(--warning))]', cssVar: '--warning' },
  blocked: { label: 'Blocked', dotColor: 'bg-[hsl(var(--destructive))]', cssVar: '--destructive' },
  done: { label: 'Done', dotColor: 'bg-[hsl(var(--success))]', cssVar: '--success' },
  failed: { label: 'Failed', dotColor: 'bg-[hsl(var(--destructive))]', cssVar: '--destructive' },
}
