'use client'

/**
 * PRD-268 §6.3 'In the thread', K8 (US-W03b) — the caller's follow-ups in one chat, and Cancel.
 *
 * The follow-up line under the message that scheduled one (components/proactive/follow-up-line.tsx) reads this
 * list: GET /api/proactive/follow-ups?chat_id=&status=all (US-C11e), the caller's own, pending, fired, cancelled
 * and missed alike, so the line can say where each stands. Every line of a chat shares the one query. It is asked
 * again when a chat changes (the board stream's `automatos:chat-changed`: a follow-up that spoke posts into the
 * chat), and once the soonest pending one is due, since a follow-up with nothing to say posts nothing to announce
 * it fired. A missed one (its time passed and the scheduler never ran it) will never fire, so it is never waited on:
 * a chat whose follow-ups are all missed, fired or cancelled asks no more. Cancel posts …/{id}/cancel; the
 * cancelled row the server answers goes into the list at once, and the list is asked again however the cancel ended
 * (a 409 means it fired, was cancelled elsewhere or was missed). A chat id that
 * isn't a UUID asks nothing, and an id that isn't a follow-up's never reaches a request path. The server answers at
 * most its LIST_LIMIT (100) rows, pending first: a chat's oldest fired or cancelled follow-ups can fall off it, and
 * their lines then draw nothing.
 */
import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { cancelledOf, followUpsOf, isFollowUpId, type FollowUp } from '@/lib/proactive/follow-ups'
import { isProactiveId } from '@/lib/proactive/wire'

const FOLLOW_UPS_PATH = '/api/proactive/follow-ups'
/** The window event after which the list is asked for again. */
export const FOLLOW_UPS_REFRESH_EVENT = 'automatos:chat-changed'
/** Once a pending follow-up is due, the list is asked again this often until it has fired. */
export const FOLLOW_UPS_POLL_MS = 60_000
/** The longest wait before the list is asked again, well inside a timer's range (2^31 - 1 ms). */
export const FOLLOW_UPS_MAX_WAIT_MS = 6 * 60 * 60_000

export const followUpQueryKeys = {
  chat: (chatId: string) => ['proactive', 'follow-ups', chatId] as const,
}

/** The list's path for one chat: every status, so a fired or cancelled follow-up still has its line. */
export function followUpsPath(chatId: string): string {
  return `${FOLLOW_UPS_PATH}?chat_id=${encodeURIComponent(chatId)}&status=all`
}

function cancelPath(id: number): string {
  if (!isFollowUpId(id)) throw new Error('not a follow-up id')
  return `${FOLLOW_UPS_PATH}/${id}/cancel`
}

/** How long until the list is asked again: when the soonest pending follow-up is due, at least a minute and at
 * most FOLLOW_UPS_MAX_WAIT_MS; never once none is pending (a missed one is not: the server stops calling it
 * pending once its time is past the scheduler's grace). */
export function nextAskIn(rows: readonly FollowUp[] | undefined, now: number = Date.now()): number | false {
  const due = (rows ?? []).filter((row) => row.status === 'pending').map((row) => (row.at ? Date.parse(row.at) : now))
  if (due.length === 0) return false
  return Math.min(FOLLOW_UPS_MAX_WAIT_MS, Math.max(FOLLOW_UPS_POLL_MS, Math.min(...due) - now))
}

function askAgain(queryClient: QueryClient, chatId: string): void {
  // Each line of the chat listens; one refetch in flight is enough.
  void queryClient.invalidateQueries({ queryKey: followUpQueryKeys.chat(chatId) }, { cancelRefetch: false })
}

/** The list with the cancelled row in its place. Never mutates. */
function withCancelled(rows: FollowUp[] | undefined, cancelled: FollowUp): FollowUp[] | undefined {
  return rows?.map((row) => (row.id === cancelled.id ? cancelled : row))
}

/** Put the cancelled row in the list, first dropping a fetch already in flight: it may have read the row before the
 * cancel committed, and would put it back as pending. */
async function holdCancelled(queryClient: QueryClient, chatId: string, cancelled: FollowUp): Promise<void> {
  const key = followUpQueryKeys.chat(chatId)
  await queryClient.cancelQueries({ queryKey: key })
  queryClient.setQueryData<FollowUp[]>(key, (rows) => withCancelled(rows, cancelled))
}

/** The caller's follow-ups in this chat, newest first. */
export function useFollowUps(chatId: string) {
  const queryClient = useQueryClient()
  const enabled = isProactiveId(chatId)

  const query = useQuery<FollowUp[]>({
    queryKey: followUpQueryKeys.chat(chatId),
    queryFn: async () => followUpsOf(await apiClient.request<unknown>(followUpsPath(chatId))),
    enabled,
    refetchInterval: (rows) => nextAskIn(rows),
  })

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return undefined
    const onChanged = () => askAgain(queryClient, chatId)
    window.addEventListener(FOLLOW_UPS_REFRESH_EVENT, onChanged)
    return () => window.removeEventListener(FOLLOW_UPS_REFRESH_EVENT, onChanged)
  }, [queryClient, chatId, enabled])

  return query
}

/** Cancel one of the caller's pending follow-ups in this chat; the list is asked again however it ends. */
export function useCancelFollowUp(chatId: string) {
  const queryClient = useQueryClient()
  return useMutation<FollowUp | null, Error, number>({
    mutationFn: async (id) => cancelledOf(await apiClient.post<unknown>(cancelPath(id), {})),
    onSuccess: async (cancelled) => {
      if (cancelled) await holdCancelled(queryClient, chatId, cancelled)
    },
    onSettled: () => {
      // A fetch that started before the cancel answered is dropped and asked again (not awaited: the cancel is done).
      void queryClient.invalidateQueries({ queryKey: followUpQueryKeys.chat(chatId) })
    },
  })
}
