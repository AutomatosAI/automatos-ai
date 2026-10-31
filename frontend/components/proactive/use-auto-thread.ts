'use client'

/**
 * PRD-268 §6.5.1 (US-W01b) — whether a chat is the Auto thread, and whose.
 *
 * The composer gets only the chat's id, so this reads the chat (GET /api/chat/{id}, PRD-205's `kind`) once per id
 * and keeps the answer: a chat's kind and owner never change. A chat the server doesn't have yet (a new
 * conversation) or one it refuses answers null, as does an id that isn't a chat's, which never reaches the path.
 * A failure that isn't a refusal (the network, a server error) is tried once more, and again on the next focus.
 */
import { useQuery } from '@tanstack/react-query'
import { getChat } from '@/lib/chat/api'
import { isProactiveId, refusalOf } from '@/lib/proactive/wire'

const AUTO_KIND = 'auto'
const RETRIES = 1
/** From here up, a status is the server's failure, worth one more try; below it, the server's answer. */
const SERVER_ERROR_STATUS = 500

export const autoThreadQueryKey = (chatId: string) => ['chat-kind', chatId] as const

/** One more try after a transient failure; none after a refusal (a chat not found, one not the caller's, a 401). */
export function retriesAfter(failures: number, error: unknown): boolean {
  const status = refusalOf(error)?.status ?? null
  const refused = status !== null && status < SERVER_ERROR_STATUS
  const unauthorized = error instanceof Error && error.message.startsWith('HTTP 401')
  return failures < RETRIES && !refused && !unauthorized
}

/** The Auto thread's owner (their users.id, as text) when `chatId` is an Auto thread; else null. */
export function useAutoThreadOwner(chatId: string): string | null {
  const { data } = useQuery({
    queryKey: autoThreadQueryKey(chatId),
    queryFn: () => getChat(chatId),
    enabled: isProactiveId(chatId),
    retry: retriesAfter,
    staleTime: Infinity,
  })
  if (data?.kind !== AUTO_KIND || data.userId === null || data.userId === undefined) return null
  return String(data.userId)
}
