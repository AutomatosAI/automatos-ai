'use client'

/**
 * PRD-244 D5 — "Auto now": the floor's live objects, read from the hooks the
 * Command Centre already polls (no new endpoint, each read keeps its own
 * cadence). One snapshot for the chat rail (both styles) and the header pill.
 */
import { useEffect, useMemo, useState } from 'react'
import { useActivityStats, type ActivityStats } from '@/hooks/use-activity-api'
import { useFleetState } from '@/hooks/use-agent-api'
import { useQuestions } from '@/hooks/use-approval-grants'
import { useWatches } from '@/hooks/use-watches-api'
import { useDecisionsNeeded } from '@/hooks/use-kpi-api'
import type { ApprovalGrant, FleetAgentRow, WatchRow } from '@/lib/api-client'

export const AUTO_NOW_ROWS = 3
/** Below this width the rail yields to the header pill (D5). */
export const AUTO_NOW_RAIL_MIN_WIDTH = 1280

export interface AutoNowSnapshot {
  stats: ActivityStats | undefined
  /** Agents with work in hand right now, most recent first. */
  working: FleetAgentRow[]
  /** Open question-kind asks, newest first. */
  questions: ApprovalGrant[]
  questionCount: number
  /** Live watches; `nextWatch` is the one due soonest. */
  watches: WatchRow[]
  nextWatch: WatchRow | null
  decisionsTotal: number
  loading: boolean
}

const ts = (iso: string | null | undefined) => (iso ? Date.parse(iso) || 0 : 0)

export function useAutoNow(): AutoNowSnapshot {
  const stats = useActivityStats('1d')
  const fleet = useFleetState()
  const questions = useQuestions()
  const watches = useWatches()
  const decisions = useDecisionsNeeded()

  return useMemo(() => {
    const working = [...(fleet.data?.agents ?? [])]
      .filter((a) => a.current !== null)
      .sort((a, b) => ts(b.current?.since) - ts(a.current?.since))
    const grants = [...(questions.data?.grants ?? [])].sort(
      (a, b) => ts(b.requested_at) - ts(a.requested_at),
    )
    const live = watches.data?.watches ?? []
    const nextWatch = [...live]
      .filter((w) => w.next_check_at)
      .sort((a, b) => ts(a.next_check_at) - ts(b.next_check_at))[0] ?? null
    return {
      stats: stats.data,
      working,
      questions: grants,
      questionCount: grants.length,
      watches: live,
      nextWatch,
      decisionsTotal: decisions.data?.total ?? 0,
      loading: stats.isLoading || fleet.isLoading || questions.isLoading || watches.isLoading || decisions.isLoading,
    }
  }, [stats.data, stats.isLoading, fleet.data, fleet.isLoading, questions.data, questions.isLoading, watches.data, watches.isLoading, decisions.data, decisions.isLoading])
}

/**
 * Whether the rail is open. Persisted per browser; a browser with no stored
 * choice opens it on wide screens and yields to the pill below
 * AUTO_NOW_RAIL_MIN_WIDTH.
 */
export function useAutoNowOpen(storageKey: string): [boolean, (next: boolean) => void] {
  const [open, setOpenState] = useState(false)

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(storageKey)
      if (stored === '1' || stored === '0') {
        setOpenState(stored === '1')
        return
      }
    } catch {
      // Storage blocked: fall through to the width default.
    }
    setOpenState(window.innerWidth >= AUTO_NOW_RAIL_MIN_WIDTH)
  }, [storageKey])

  const setOpen = (next: boolean) => {
    setOpenState(next)
    try {
      window.localStorage.setItem(storageKey, next ? '1' : '0')
    } catch {
      // Storage blocked: the choice lasts for this page only.
    }
  }
  return [open, setOpen]
}
