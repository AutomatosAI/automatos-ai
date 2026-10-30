'use client'

/**
 * PRD-268 §6.4, K11 (US-W02) — the day's brief, waiting when a person opens Automatos on the web.
 *
 * Mounted once, inside WorkspaceProvider (components/providers.tsx). When a signed-in workspace loads, it opens the
 * day's brief (POST /api/proactive/brief/open, hooks/use-brief.ts `useOpenBrief`) with the browser's local date, time
 * zone and language (lib/proactive/brief-day.ts `browserDay`). The server keeps one brief per person, workspace and
 * local date, so the client only avoids needless calls: once an open has answered where the day's brief is, the tab
 * remembers it (localStorage, per workspace and date, lib/proactive/brief-day.ts) and a reload asks nothing; the next
 * date asks again. A workspace and date are asked about at most once per load, so nothing here ever loops: a refusal
 * (a 4xx) or a failure is logged once and asked again only on the next load, as is a pending answer (another open
 * still building the brief), which names no message, and a date a tab left open past midnight. With storage blocked
 * it asks once per load. Without a workspace it asks nothing. Each open's failure is caught on its own promise, so a
 * workspace switched while an open is in flight still logs that open's failure.
 *
 * The brief on open runs only where the workspace's proactive mode is on (PRD-268 §10 rollout): elsewhere the open is
 * refused with 404 `brief_off`, which is the expected answer, not a fault, so it logs nothing and, like any refusal,
 * is not asked again in that load.
 *
 * The answer opens nothing by itself: no navigation, no sound, nothing drawn here. The brief draws where it was
 * posted, as its card in the Auto thread (thread-cards.tsx), and under Auto's Read on the Command Center (useBrief).
 */
import { useEffect, useRef } from 'react'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useOpenBrief } from '@/hooks/use-brief'
import { browserDay, rememberedBrief } from '@/lib/proactive/brief-day'
import { refusalOf } from '@/lib/proactive/wire'

const LOG_PREFIX = '[brief]'
const NOT_OPENED = "The day's brief wasn't opened"
/** The open's refusal where the workspace's proactive mode is off (api/proactive_brief.py BRIEF_OFF). */
const BRIEF_OFF = 'brief_off'

type Asked = ReadonlySet<string>

const dayKey = (workspaceId: string, localDate: string): string => `${workspaceId}:${localDate}`

/**
 * Logs why the open didn't answer: its status and code, once per workspace and date in a load. `brief_off` (the
 * workspace's proactive mode is off) is the expected answer there, so it logs nothing.
 */
function logNotOpened(error: unknown): void {
  const refusal = refusalOf(error)
  if (refusal?.code === BRIEF_OFF) return
  console.warn(LOG_PREFIX, NOT_OPENED, refusal?.status ?? null, refusal?.code ?? refusal?.detail ?? String(error))
}

/** Opens the day's brief once per local date when a workspace loads; it draws nothing. */
export function BriefOnOpen(): null {
  const workspaceId = useWorkspaceOptional()?.workspace?.id ?? null
  const { mutateAsync } = useOpenBrief()
  const asked = useRef<Asked>(new Set())

  useEffect(() => {
    if (workspaceId === null) return
    const day = browserDay()
    const key = dayKey(workspaceId, day.local_date)
    if (asked.current.has(key) || rememberedBrief(workspaceId, day.local_date) !== null) return
    asked.current = new Set([...Array.from(asked.current), key])
    mutateAsync(day).catch(logNotOpened)
  }, [workspaceId, mutateAsync])

  return null
}
