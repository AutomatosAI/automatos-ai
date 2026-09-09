/**
 * PRD-239 S7 v2 — the Runtime Canvas: where a session agent (runtime: cli) lives.
 *
 * Picking the agent in a chat, or opening its ticket from the board, lands on ONE
 * view — explorer + the agent's Claude Code session in a terminal, opened for you,
 * covering the page. No chat lane, no card, no command to copy.
 */
import type { CodingCanvasWidgetData } from '@/components/widgets/types'

export const WORKSPACE_ROOT_FALLBACK = '.'

export interface RuntimeSession {
  task_id: number | string
  explorer_root?: string | null
  cwd?: string | null
  agent_name?: string | null
}

/** The folder the explorer opens on: the session's browsable root, else the workspace root. */
export function runtimeCanvasRoot(explorerRoot: string | null | undefined): string {
  const root = (explorerRoot ?? '').trim()
  return root && root !== '/' ? root : WORKSPACE_ROOT_FALLBACK
}

/** `/chat?ticket=…&repo=…&runtime=1` — the board's way into the same view. */
export function sessionCanvasHref(taskId: number | string, explorerRoot: string | null | undefined): string {
  return `/chat?repo=${encodeURIComponent(runtimeCanvasRoot(explorerRoot))}&ticket=${encodeURIComponent(String(taskId))}&runtime=1`
}

/** The widget data a Runtime Canvas opens with. */
export function runtimeCanvasData(workspaceId: string, session: RuntimeSession): CodingCanvasWidgetData {
  return {
    workspaceId,
    rootPath: runtimeCanvasRoot(session.explorer_root),
    taskId: String(session.task_id),
    runtime: true,
    openFullscreen: true,
  }
}

/** The Canvas title: the agent's name when known, else the ticket. */
export function runtimeCanvasTitle(session: RuntimeSession): string {
  return session.agent_name ? `${session.agent_name} · session` : `Session · ticket #${session.task_id}`
}
