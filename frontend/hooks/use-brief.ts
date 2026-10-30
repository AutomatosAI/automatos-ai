'use client'

/**
 * PRD-268 §6.4, §6.11.2, K11 (US-C21b) — the day's brief: open it, and read it.
 *
 * `useOpenBrief` posts POST /api/proactive/brief/open with the browser's day (lib/proactive/brief-day.ts).
 * The server keeps one brief per person, workspace and local date, shared by every device and tab, and
 * answers where it is: the message this open posted, or the day's message another open or the
 * scheduled digest posted, or pending while that one is still being built. The tab remembers where
 * the day's brief is, per workspace and date.
 *
 * `useBrief` reads the day's brief message: the Brief@1 in its `brief@1` part, from
 * GET /api/chat/{chat_id}/messages (the person's own Auto thread). It reads only what an open
 * answered; it never opens the brief itself, so reading it costs no build. Until an open has
 * answered, and on a message without a brief@1 part, its data is null.
 */
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { useWorkspaceOptional } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import { browserDay, forgetBrief, localDateOf, rememberBrief, rememberedBrief } from '@/lib/proactive/brief-day'
import { briefOfMessages, briefRefOf, openedOf, refusalOf } from '@/lib/proactive/wire'
import type { Brief, BriefDay, BriefMessageRef, BriefOpened } from '@/lib/proactive/types'

const OPEN_PATH = '/api/proactive/brief/open'
/** The day's message is gone (404) or no longer the caller's (403): where it was is forgotten. */
const GONE_STATUSES: ReadonlySet<number> = new Set([403, 404])

export const briefQueryKeys = {
  all: ['proactive', 'brief'] as const,
  /** Where the workspace's brief for a local date is. */
  where: (workspaceId: string | null, localDate: string) => ['proactive', 'brief', 'where', workspaceId, localDate] as const,
  /** The brief that message carries. */
  message: (ref: BriefMessageRef | null) =>
    ['proactive', 'brief', 'message', ref?.chat_id ?? null, ref?.message_id ?? null] as const,
}

function useWorkspaceId(): string | null {
  return useWorkspaceOptional()?.workspace?.id ?? null
}

interface BriefPlace {
  readonly queryClient: QueryClient
  readonly workspaceId: string | null
  readonly localDate: string
  readonly ref: BriefMessageRef
}

/** The brief in the day's message. A message that is gone forgets where it was, so the next open is asked again. */
async function readBrief({ queryClient, workspaceId, localDate, ref }: BriefPlace): Promise<Brief | null> {
  try {
    return briefOfMessages(await apiClient.request<unknown>(`/api/chat/${ref.chat_id}/messages`), ref)
  } catch (failure) {
    const status = refusalOf(failure)?.status
    if (status === null || status === undefined || !GONE_STATUSES.has(status)) throw failure
    forgetBrief(workspaceId, localDate)
    queryClient.setQueryData(briefQueryKeys.where(workspaceId, localDate), null)
    return null
  }
}

/** Remember where the day's brief is, for this tab's reads and the next load's. */
function recordOpened(queryClient: QueryClient, workspaceId: string | null, day: BriefDay, opened: BriefOpened): void {
  const ref = briefRefOf(opened)
  if (ref === null) return
  rememberBrief(workspaceId, day.local_date, ref)
  queryClient.setQueryData(briefQueryKeys.where(workspaceId, day.local_date), ref)
}

/** The day's brief, or null while no open has answered where it is. */
export function useBrief() {
  const queryClient = useQueryClient()
  const workspaceId = useWorkspaceId()
  const localDate = localDateOf(new Date())

  const where = useQuery<BriefMessageRef | null>({
    queryKey: briefQueryKeys.where(workspaceId, localDate),
    queryFn: () => rememberedBrief(workspaceId, localDate),
    staleTime: Infinity,
  })
  const ref = where.data ?? null

  return useQuery<Brief | null>({
    queryKey: briefQueryKeys.message(ref),
    queryFn: () => (ref === null ? null : readBrief({ queryClient, workspaceId, localDate, ref })),
    enabled: ref !== null,
    // A posted brief never changes: its rows' presses are drawn in place by the card.
    staleTime: Infinity,
  })
}

/** Open the day's brief: `{chat_id, message_id, posted}`, or `{posted: false, pending: true}`. */
export function useOpenBrief() {
  const queryClient = useQueryClient()
  const workspaceId = useWorkspaceId()
  return useMutation<BriefOpened, Error, BriefDay | void>({
    mutationFn: async (asked) => {
      const day = typeof asked === 'object' && asked !== null ? asked : browserDay()
      const opened = openedOf(await apiClient.post<unknown>(OPEN_PATH, day))
      recordOpened(queryClient, workspaceId, day, opened)
      return opened
    },
  })
}
