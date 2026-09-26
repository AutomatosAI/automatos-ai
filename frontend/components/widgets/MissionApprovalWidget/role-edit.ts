import type { MissionTaskEdit } from '@/types/missions'
import type { MissionApprovalTask } from '../types'

/**
 * F162 (c): the plan edit that re-staffs one task on the approval card. Tasks
 * that run side by side share a step number, and the server refuses an edit by
 * step number then (81e8f37a had seven at step 1), so a task the card knows by
 * its temp_id is named by it. A card built before the tool result carried
 * temp_ids falls back to the step number.
 */
export function roleEdit(task: MissionApprovalTask, sequence: number, agentRole: string): MissionTaskEdit {
  return task.temp_id
    ? { temp_id: task.temp_id, agent_role: agentRole }
    : { sequence_number: sequence, agent_role: agentRole }
}

/**
 * F162 (b): the card's tasks after an edit, from the plan the server answered
 * with. The server ranks an edited task's agent again, and the card shows its
 * role and "who would run it" from that task's own plan entry (by temp_id).
 * A card without temp_ids keeps its tasks as they were.
 */
export function withPlanPicks(
  tasks: MissionApprovalTask[],
  plan: Record<string, unknown> | null | undefined,
): MissionApprovalTask[] {
  const planTasks = Array.isArray(plan?.tasks) ? (plan?.tasks as Array<Record<string, unknown>>) : []
  const byTemp = new Map(planTasks.filter((p) => typeof p.temp_id === 'string').map((p) => [p.temp_id as string, p]))
  return tasks.map((task) => {
    const entry = task.temp_id ? byTemp.get(task.temp_id) : undefined
    if (!entry) return task
    return {
      ...task,
      agent_role: typeof entry.agent_role === 'string' ? entry.agent_role : task.agent_role,
      match_agent: typeof entry.match_agent === 'string' ? entry.match_agent : task.match_agent,
      match_reason: typeof entry.match_reason === 'string' ? entry.match_reason : task.match_reason,
      match_is_override: Boolean(entry.match_is_override),
    }
  })
}
