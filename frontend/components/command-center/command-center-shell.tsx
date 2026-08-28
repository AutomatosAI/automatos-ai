'use client'

/**
 * CommandCenterShell — page frame for the Studio /command-center route.
 *
 * Layout (top to bottom):
 *   1. Editorial top: eyebrow + h1 + sub + actions cluster
 *   2. StatsStrip (auto-swaps to prose when everything's zero)
 *   3. Tab strip (Summary · Board · Calendar · Activity) with live counts
 *   4. Tab body
 *
 * The active tab is driven by `?tab=` in the URL. Counts on tabs come from
 * the same data hooks that the tabs use, so they stay in sync as backend
 * state changes.
 */

import { useMemo } from 'react'
import { useRouter, useSearchParams, usePathname } from 'next/navigation'
import { RotateCw } from 'lucide-react'

import { useActivityStats, useActivityFeed } from '@/hooks/use-activity-api'
import { useBoardTasks } from '@/hooks/use-board-tasks'
import { useBoardEventStream } from '@/hooks/use-board-event-stream'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import { useDecisionsNeeded } from '@/hooks/use-kpi-api'
import { useWatches } from '@/hooks/use-watches-api'
import { useQuestions } from '@/hooks/use-approval-grants'

import { TrialBalancePill } from '@/components/onboarding/trial-balance-pill'
import { SetupChecklistCard } from '@/components/onboarding/setup-checklist-card'

import { StatsStrip } from './stats-strip'
import { IsItWorkingStrip } from './is-it-working-strip'
import { SummaryTab } from './summary-tab'
import { BoardTab } from './board-tab'
import { CalendarTab } from './calendar-tab'
import { ActivityTab } from './activity-tab'
import { WatchlistTab } from './watchlist-tab'
import { GovernanceTab } from './governance-tab'
import { QuestionsTab } from './questions-tab'

type TabKey =
  | 'summary'
  | 'board'
  | 'calendar'
  | 'activity'
  | 'watchlist'
  | 'questions'
  | 'governance'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'summary', label: 'Summary' },
  { key: 'board', label: 'Board' },
  { key: 'calendar', label: 'Calendar' },
  { key: 'activity', label: 'Activity' },
  // PRD-204 S11: the watchlist -- work Auto is supervising to a verdict.
  { key: 'watchlist', label: 'Watchlist' },
  // PRD-225 (G2): the Questions tab -- every open agent ask, with the blocked
  // cascade behind it. The badge is a live count of open questions.
  { key: 'questions', label: 'Questions' },
  // PRD-196 (P2-15): the governance pillar (Approvals · Audit · Policy ·
  // Compliance), ws-admin-only. The human surface for the policy plane.
  { key: 'governance', label: 'Governance' },
]

const VALID_TABS = new Set<TabKey>(TABS.map((t) => t.key))

function todayDateline(): string {
  const now = new Date()
  const date = now.toLocaleDateString('en-GB', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  })
  const time = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  const tz =
    Intl.DateTimeFormat().resolvedOptions().timeZone.split('/').pop() ?? ''
  return `${date} · ${time} ${tz}`
}

export function CommandCenterShell() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()

  const rawTab = (searchParams?.get('tab') ?? 'summary') as TabKey
  const activeTab: TabKey = VALID_TABS.has(rawTab) ? rawTab : 'summary'

  const { data: stats } = useActivityStats('1d')
  const { columns } = useBoardTasks()
  const { data: schedule } = useActivitySchedule('7d')
  const { data: feed } = useActivityFeed({ limit: 200 })
  const { data: decisions } = useDecisionsNeeded(10)
  // PRD-204 S11: live watches only (the default list) -- the tab badge is
  // "how many things is Auto supervising right now".
  const { data: watchlist } = useWatches()
  // PRD-225: open (pending) question-kind asks — the Questions tab badge.
  const { data: questions } = useQuestions()

  // PRD-180 S1 (F090): real-time board push. Subscribes to the LISTEN/NOTIFY
  // SSE and invalidates the board cache on each pushed event — this is what
  // makes "Streaming live" honest (the board no longer polls on an interval).
  useBoardEventStream(true)

  const dateline = useMemo(todayDateline, [])

  const tabCounts: Record<TabKey, number> = useMemo(
    () => ({
      summary: decisions?.total ?? 0,
      board: columns.reduce(
        (sum, c) => (c.status === 'done' ? sum : sum + c.tasks.length),
        0,
      ),
      calendar: schedule?.scheduled?.length ?? 0,
      activity: feed?.total ?? feed?.items?.length ?? 0,
      watchlist: watchlist?.total ?? 0,
      // PRD-225: the count of open questions — the wave's most user-visible
      // surface earns a live badge (a member without ws-admin gets 0).
      questions: questions?.grants?.length ?? 0,
      // No live badge on Governance — the pending-approvals count lives inside
      // the ws-admin-gated pane, not fetched for every member on the shell.
      governance: 0,
    }),
    [decisions, columns, schedule, feed, watchlist, questions],
  )

  const working = stats?.working_now ?? 0
  const attn = stats?.needs_attention ?? 0
  const isQuiet = working === 0 && attn === 0

  const lede = isQuiet ? (
    <>
      A quiet hour. <span className="num">{stats?.agents_active ?? 0}</span> agents
      working,{' '}
      <span className="num">{stats?.tasks_in_queue ?? 0}</span> in queue. Routines
      keep the lights on while you focus on something else.
    </>
  ) : (
    <>
      <span className="num">{working}</span> agent{working === 1 ? '' : 's'} working,{' '}
      <span className="num">{stats?.tasks_in_queue ?? 0}</span> in queue
      {attn > 0 && (
        <>
          ,{' '}
          <span style={{ color: 'hsl(var(--accent))' }}>
            <span className="num">{attn}</span> need
            {attn === 1 ? 's' : ''} your eyes
          </span>
        </>
      )}
      . Streaming live; switch tabs to drill in.
    </>
  )

  const setTab = (k: TabKey) => {
    const params = new URLSearchParams(searchParams?.toString() ?? '')
    params.set('tab', k)
    router.push(`${pathname}?${params.toString()}` as any, { scroll: false })
  }

  return (
    <div className="cc-page">
      <div className="cc-headrow">
        <div className="cc-head">
          <p className="cc-eyebrow">Operations · {dateline}</p>
          <h1 className="cc-h1">Command Centre</h1>
          <p className="cc-sub">{lede}</p>
        </div>
        <div className="cc-actions">
          {/* PRD-222 US-014: trial balance, honest on the Command Center too —
              same snapshot the chat pill reads; self-hides once converted. */}
          <TrialBalancePill />
          <button
            type="button"
            className="cc-btn"
            onClick={() => router.refresh()}
            title="Refresh"
          >
            <RotateCw style={{ width: 12, height: 12 }} />
            Refresh
          </button>
        </div>
      </div>

      <StatsStrip />
      <IsItWorkingStrip />

      {/* PRD-222 US-020: the post-setup checklist, dual-surfaced here from the
          same server read-model the chat card reads (self-hides off the
          powerup/completed stages or once dismissed). */}
      <SetupChecklistCard className="my-3" />

      <nav className="cc-tabs" aria-label="Command Centre sections">
        {TABS.map((t) => {
          const isActive = t.key === activeTab
          const count = tabCounts[t.key]
          return (
            <button
              key={t.key}
              type="button"
              className={`cc-tab${isActive ? ' active' : ''}`}
              aria-current={isActive ? 'page' : undefined}
              onClick={() => setTab(t.key)}
            >
              <span>{t.label}</span>
              {count > 0 && <span className="cc-tab-ct">{count}</span>}
            </button>
          )
        })}
      </nav>

      <div className="cc-body">
        {activeTab === 'summary' && <SummaryTab />}
        {activeTab === 'board' && <BoardTab />}
        {activeTab === 'calendar' && <CalendarTab />}
        {activeTab === 'activity' && <ActivityTab />}
        {activeTab === 'watchlist' && <WatchlistTab />}
        {activeTab === 'questions' && <QuestionsTab />}
        {activeTab === 'governance' && <GovernanceTab />}
      </div>
    </div>
  )
}
