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
 * PRD-252 R8: the page scrolls as one. The head and the stats scroll away, the
 * tab strip sticks at the top, and the strip with the tab body (`.cc-work`) is
 * at least the visible height, so a work surface can take the whole screen.
 * Board and Calendar fill it exactly and keep their own inner scroll; the other
 * tabs grow with their content under the stuck strip (globals.css).
 *
 * The active tab is driven by `?tab=` in the URL. Counts on tabs come from
 * the same data hooks that the tabs use, so they stay in sync as backend
 * state changes. PRD-252 R5: the Board tab's badge and the lede's "need your
 * eyes" are the Needs-you number, the one the Needs you widget lists.
 */

import { useEffect, useMemo, useState } from 'react'
import { useRouter, useSearchParams, usePathname } from 'next/navigation'
import { RotateCw } from 'lucide-react'

import { useTabStripScroll } from '@/hooks/use-tab-strip-scroll'
import { useActivityStats, useActivityFeed } from '@/hooks/use-activity-api'
import { useBoardEventStream } from '@/hooks/use-board-event-stream'
import { useActivitySchedule } from '@/hooks/use-activity-api'
import { useNeedsYou } from '@/hooks/use-needs-you'
import { useWatches } from '@/hooks/use-watches-api'
import { useQuestions } from '@/hooks/use-approval-grants'

import { TrialBalancePill } from '@/components/onboarding/trial-balance-pill'
import { SetupChecklistCard } from '@/components/onboarding/setup-checklist-card'

import { StatsStrip } from './stats-strip'
import { PeriodSelect, type Period } from './period-select'
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
// PRD-252 R8: the work surfaces that take the whole height under the tab strip.
const FILL_TABS = new Set<TabKey>(['board', 'calendar'])

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

  // PRD-244 W1: the legacy page's period selector, kept — one period drives
  // the stats, the Summary tab's read and the Activity stream.
  const [period, setPeriod] = useState<Period>('1d')
  const { data: stats } = useActivityStats(period)
  const { data: needsYou } = useNeedsYou()
  const { data: schedule } = useActivitySchedule('7d')
  // The backend caps `limit` at 100 (api/activity.py) — 200 was a 422 and an
  // empty Activity count; PR #397 found the same on the tab (harvested here).
  const { data: feed } = useActivityFeed({ limit: 100 })
  // PRD-204 S11: live watches only (the default list) -- the tab badge is
  // "how many things is Auto supervising right now".
  const { data: watchlist } = useWatches()
  // PRD-225: open (pending) question-kind asks — the Questions tab badge.
  const { data: questions } = useQuestions()

  // PRD-180 S1 (F090): real-time board push. Subscribes to the LISTEN/NOTIFY
  // SSE and invalidates the board cache on each pushed event — this is what
  // makes "Streaming live" honest (the board no longer polls on an interval).
  useBoardEventStream(true)

  // F219: the clock is the reader's, so it is written after mount. Rendered on
  // the server it carried the server's time and zone, and React logged #418.
  const [dateline, setDateline] = useState('')
  useEffect(() => setDateline(todayDateline()), [])
  // Seven tabs are wider than a phone, so the active one is scrolled into
  // view on a compact viewport (PRD-246 US-002).
  const tabStrip = useTabStripScroll(activeTab)

  const tabCounts: Record<TabKey, number> = useMemo(
    () => ({
      // PRD-252 R5: the one Needs-you number sits on Board; it counted every open ticket.
      summary: 0,
      board: needsYou?.total ?? 0,
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
    [needsYou, schedule, feed, watchlist, questions],
  )

  const working = stats?.working_now ?? 0
  const attn = needsYou?.total ?? 0
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
          <p className="cc-eyebrow">{dateline ? `Operations · ${dateline}` : 'Operations'}</p>
          <h1 className="cc-h1">Command Center</h1>
          <p className="cc-sub">{lede}</p>
        </div>
        <div className="cc-actions">
          <PeriodSelect value={period} onChange={setPeriod} />
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

      <div className={`cc-work${FILL_TABS.has(activeTab) ? ' fill' : ''}`}>
        <nav className="cc-tabs" aria-label="Command Center sections" ref={tabStrip}>
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
          {activeTab === 'summary' && <SummaryTab period={period} />}
          {activeTab === 'board' && <BoardTab />}
          {activeTab === 'calendar' && <CalendarTab />}
          {activeTab === 'activity' && <ActivityTab period={period} />}
          {activeTab === 'watchlist' && <WatchlistTab />}
          {activeTab === 'questions' && <QuestionsTab />}
          {activeTab === 'governance' && <GovernanceTab />}
        </div>
      </div>
    </div>
  )
}
