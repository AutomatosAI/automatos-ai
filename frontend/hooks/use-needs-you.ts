/**
 * PRD-252 R5 — one "Needs you" number.
 *
 * Five counters had five definitions: the Board tab badge counted every open
 * ticket, ATTENTION counted a grant-blocked ticket twice, the Needs you widget
 * summed four lists, and Auto's pill read a super-admin-only endpoint. Now one
 * endpoint serves the number and the rows behind it
 * (orchestrator/services/needs_you.py). Every counter reads this hook, so they
 * share one cached response and cannot disagree.
 */
import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/** F246: why a ticket is stuck (services/needs_you.py STUCK_*). */
export type StuckWhy = 'no_host' | 'no_agent' | 'not_picked_up' | 'mission_ended'

/** A ticket in Review, a stuck one, or one that failed. */
export interface NeedsYouTicketRow {
  ticket_id: number
  /** PRD-252 R4: #0042 (a mission step's: #0051.3) */
  number?: string | null
  title: string | null
  agent_name: string | null
  /**
   * Set when the decision is the mission's: on a mission's own card, and (F274) on
   * a stuck step whose mission ended. The row opens the mission.
   */
  mission_id: string | null
  /** F274: on a stuck step whose mission ended, its mission card's number and title. */
  mission_number?: string | null
  mission_title?: string | null
  at: string | null
  /** Set on a stuck ticket: why nothing will move it. */
  why?: StuckWhy
}

/** An open question, an approval grant, or a mission waiting for its plan's approval. */
export interface NeedsYouAskRow {
  source: 'grant' | 'mission'
  id: string
  /** A question's words; an approval's ticket title (F274), or what it is for. */
  title: string | null
  ticket_id: number | null
  /** PRD-252 R4: the number of the ticket it opens in, named as on every row (F274) */
  number?: string | null
  agent_name: string | null
  at: string | null
}

export interface NeedsYouCounts {
  review: number
  question: number
  approval: number
  stuck: number
  failed: number
}

export interface NeedsYou {
  total: number
  counts: NeedsYouCounts
  rows: {
    review: NeedsYouTicketRow[]
    question: NeedsYouAskRow[]
    approval: NeedsYouAskRow[]
    stuck: NeedsYouTicketRow[]
    failed: NeedsYouTicketRow[]
  }
}

export const needsYouQueryKeys = {
  all: ['activity', 'needs-you'] as const,
}

/**
 * The Needs-you number and its rows. It has no period (F246): a failure or a
 * stuck ticket waits until the owner deals with it. A pushed board change
 * (`automatos:board-changed`, from the board stream) refreshes it at once: a
 * ticket entering Review changes every counter together.
 */
export function useNeedsYou() {
  const queryClient = useQueryClient()

  const query = useQuery<NeedsYou>({
    queryKey: needsYouQueryKeys.all,
    queryFn: () => apiClient.request<NeedsYou>('/api/activity/needs-you'),
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
