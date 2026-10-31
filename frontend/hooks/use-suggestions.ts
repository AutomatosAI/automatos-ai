'use client'

/**
 * PRD-268 §6.2.5, §6.5.1 (US-C21b) — the open suggestions, and Start and Hide.
 *
 * One list backs every surface (the Command Center's Suggested list, the card pinned above the
 * composer, the thread's suggestion@1 cards, a theme through the SDK's proactive area), so they
 * cannot disagree. It is GET /api/proactive/suggestions?status=open (W1's US-C06a: the caller's,
 * or the workspace's for an owner or admin, in any mode), polled every minute and asked again at
 * once when a suggestion, a chat or the board changes (the board stream's window events).
 *
 * Start and Hide post W1's routes and ask for the list again once the server has decided. A
 * refused Start (the day's spend ceiling, a Mission W2 can't start, a suggestion already decided)
 * leaves the list as it is, so the card stays put and says the server's reason
 * (`refusalOf(start.error)`, lib/proactive/wire.ts). An id that isn't a suggestion's never reaches
 * a request path.
 */
import { useEffect } from 'react'
import { useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { isProactiveId, suggestionsOf } from '@/lib/proactive/wire'
import type { HideAnswer, HideReason, StartAnswer, Suggestion, SuggestionVia } from '@/lib/proactive/types'

const SUGGESTIONS_PATH = '/api/proactive/suggestions'
/** How often the open list is asked for again. */
export const SUGGESTIONS_POLL_MS = 60_000
const SUGGESTIONS_STALE_MS = 30_000

/**
 * The window events after which the list is asked for again: a suggestion's own change (the board
 * stream's suggestion_changed frame, US-W01b), a chat's (a post in the Auto thread) and the board's
 * (a started suggestion's ticket).
 */
export const SUGGESTION_REFRESH_EVENTS = [
  'automatos:suggestion-changed',
  'automatos:chat-changed',
  'automatos:board-changed',
] as const

export const suggestionQueryKeys = {
  all: ['proactive', 'suggestions'] as const,
  open: ['proactive', 'suggestions', 'open'] as const,
}

export interface StartSuggestionVars {
  id: string
  via?: SuggestionVia
}

export interface HideSuggestionVars {
  id: string
  /** One of the reason menu's three; left out, the suggestion is hidden with no reason. */
  reason?: HideReason
  via?: SuggestionVia
}

function decisionPath(id: string, decision: 'start' | 'hide'): string {
  if (!isProactiveId(id)) throw new Error('not a suggestion id')
  return `${SUGGESTIONS_PATH}/${id}/${decision}`
}

function askAgain(queryClient: QueryClient): void {
  // Several surfaces mount the list; one refetch in flight is enough.
  void queryClient.invalidateQueries({ queryKey: suggestionQueryKeys.all }, { cancelRefetch: false })
}

/** The open suggestions, newest first. */
export function useSuggestions() {
  const queryClient = useQueryClient()

  const query = useQuery<Suggestion[]>({
    queryKey: suggestionQueryKeys.open,
    queryFn: async () => suggestionsOf(await apiClient.request<unknown>(`${SUGGESTIONS_PATH}?status=open`)),
    refetchInterval: SUGGESTIONS_POLL_MS,
    staleTime: SUGGESTIONS_STALE_MS,
  })

  useEffect(() => {
    if (typeof window === 'undefined') return undefined
    const onChanged = () => askAgain(queryClient)
    SUGGESTION_REFRESH_EVENTS.forEach((name) => window.addEventListener(name, onChanged))
    return () => SUGGESTION_REFRESH_EVENTS.forEach((name) => window.removeEventListener(name, onChanged))
  }, [queryClient])

  return query
}

/** Start: `{suggestion, opens}`, where `opens` is the ticket it filed or the chat to open. */
export function useStartSuggestion() {
  const queryClient = useQueryClient()
  return useMutation<StartAnswer, Error, StartSuggestionVars>({
    mutationFn: ({ id, via }) => apiClient.post<StartAnswer>(decisionPath(id, 'start'), via ? { via } : {}),
    onSuccess: () => askAgain(queryClient),
  })
}

/** Hide, with the reason the person picked from the menu. */
export function useHideSuggestion() {
  const queryClient = useQueryClient()
  return useMutation<HideAnswer, Error, HideSuggestionVars>({
    mutationFn: ({ id, reason, via }) =>
      apiClient.post<HideAnswer>(decisionPath(id, 'hide'), { ...(reason ? { reason } : {}), ...(via ? { via } : {}) }),
    onSuccess: () => askAgain(queryClient),
  })
}
