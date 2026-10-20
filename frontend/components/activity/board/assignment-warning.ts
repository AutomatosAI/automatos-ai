/**
 * PRD-256 F430 — a ticket given to a CLI-session agent says what the agent's sessions cannot
 * call. `POST /api/v1/tasks` and `PATCH /api/v1/tasks/{id}` (with `assigned_agent_id`) answer
 * with `cannot_call` and `warning` when the ticket names a platform tool the agent's sessions
 * do not have; the owner sees the warning beside the success, and the assignment stands.
 */
import { toast } from 'sonner'

/** Long enough to read a sentence that names tools and a fix. */
export const ASSIGNMENT_WARNING_DURATION_MS = 9000

/** The answer's `warning`, when it carries one; null otherwise. */
export function assignmentWarning(answer: unknown): string | null {
  const warning = (answer as { warning?: unknown } | null | undefined)?.warning
  return typeof warning === 'string' && warning.trim() ? warning.trim() : null
}

/** Show the answer's warning, if any, as a warning toast. */
export function toastAssignmentWarning(answer: unknown): void {
  const warning = assignmentWarning(answer)
  if (warning) toast.warning(warning, { duration: ASSIGNMENT_WARNING_DURATION_MS })
}
