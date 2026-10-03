/**
 * PRD-252 R5 — one "Needs you" number.
 *
 * Five counters had five definitions: the Board tab badge counted every open
 * ticket, ATTENTION counted a grant-blocked ticket twice, the Needs you widget
 * summed four lists, and Auto's pill read a super-admin-only endpoint. Now one
 * endpoint serves the number and the rows behind it
 * (orchestrator/services/needs_you.py). Every counter reads this hook, so for
 * one period they share one cached response and cannot disagree.
 */
import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/** A ticket in Review, or one that failed in the period. */
export interface NeedsYouTicketRow {
  ticket_id: number
  /** PRD-252 R4: #0042 */
  number?: string | null
  title: string | null
  agent_name: string | null
  /** Set on a mission's own card: the row opens the mission. */
  mission_id: string | null
  at: string | null
}

/** An open question, an approval grant, or a mission waiting for its plan's approval. */
export interface NeedsYouAskRow {
  source: 'grant' | 'mission'
  id: string
  title: string | null
  ticket_id: number | null
  /** PRD-252 R4: the number of the ticket it opens in */
  ticket_number?: string | null
  agent_name: string | null
  at: string | null
}

export interface NeedsYouCounts {
  review: number
  question: number
  approval: number
  failed: number
}

export interface NeedsYou {
  period: string
  total: number
  counts: NeedsYouCounts
  rows: {
    review: NeedsYouTicketRow[]
    question: NeedsYouAskRow[]
    approval: NeedsYouAskRow[]
    failed: NeedsYouTicketRow[]
  }
}

export const needsYouQueryKeys = {
  all: ['activity', 'needs-you'] as const,
  period: (period: string) => ['activity', 'needs-you', period] as const,
}

/**
 * The Needs-you number and its rows for `period` ('1d', '7d', '30d', '90d').
 * A pushed board change (`automatos:board-changed`, from the board stream)
 * refreshes it at once: a ticket entering Review changes every counter together.
 */
export function useNeedsYou(period: string = '1d') {
  const queryClient = useQueryClient()

  const query = useQuery<NeedsYou>({
    queryKey: needsYouQueryKeys.period(period),
    queryFn: () =>
      apiClient.request<NeedsYou>(`/api/activity/needs-you?period=${encodeURIComponent(period)}`),
    refetchInterval: 60000,
    staleTime: 30000,
  })

  useEffect(() => {
    if (typeof window === 'undefined') return
    // Several counters mount this hook; one refetch in flight is enough.
    const onBoardChanged = () => {
      void queryClient.invalidateQueries({ queryKey: needsYouQueryKeys.all }, { cancelRefetch: false })
    }
    window.addEventListener('automatos:board-changed', onBoardChanged)
    return () => window.removeEventListener('automatos:board-changed', onBoardChanged)
  }, [queryClient])

  return query
}
