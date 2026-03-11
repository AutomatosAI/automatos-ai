/**
 * Board types for the Kanban task management view (PRD-72 v2).
 * "Task" in UI = "Recipe" in backend. No backend rename.
 */

export type BoardStatus = 'inbox' | 'assigned' | 'in_progress' | 'review' | 'done'

export type TaskPriority = 'urgent' | 'high' | 'medium' | 'low'

export type ReviewMode = 'human' | 'llm' | 'auto'

export interface BoardTask {
  id: string
  type: 'routine' | 'recipe'
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
  report_id?: string
  source_id: string
  project_id?: number
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
  { status: 'done', label: 'Done' },
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
  done: { label: 'Done', dotColor: 'bg-[hsl(var(--success))]', cssVar: '--success' },
}
