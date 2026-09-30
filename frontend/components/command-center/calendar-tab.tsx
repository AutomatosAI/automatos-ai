'use client'

/**
 * CalendarTab — Studio calendar for scheduled work.
 *
 * Three real view modes (all wired):
 *  - Day: single column for the selected day, hour-by-hour
 *  - Week: 7 columns (Sun–Sat), hour-by-hour (default)
 *  - Month: classic 6×7 grid with event chips per day
 *
 * ONE data source: `useActivitySchedule` (GET /api/activity/schedule), the
 * DB-first feed PRD-162 made identical on every worker. It carries six kinds
 * of item (see ScheduleItemType): heartbeat routines with a structured
 * recurrence, cron playbooks, agent-scheduled tasks, mission SLA deadlines,
 * board-task SLA deadlines and scheduled social posts (PRD-251 US-307: shown in
 * the post's own timezone, rescheduled by drag or from the menu). The always-on band and the grid's routine rows are
 * built from the routine items' `recurrence.interval_minutes` — NOT from
 * /api/heartbeat/workspace, which sits behind the PRD-143 router-wide
 * super-admin lock (a 403 for every other user, polled every 30s) and read
 * `next_run_at` from the one worker that hosts APScheduler (null on the others,
 * so routines anchored at "now" and shifted on every refresh).
 *
 * Events are actionable: clicking one opens a small menu — open the agent /
 * playbook / mission / board card, pause a routine, pause or cancel a
 * scheduled task. Every action rides an endpoint that already exists (see
 * calendar-actions.ts). Deadline items carry a DUE tag; overdue ones go amber.
 * Colour is by KIND (heartbeat / playbook / scheduled task / mission SLA /
 * task deadline — calendar-kinds.ts); the agent is the small dot. Legend chips
 * in the toolbar hide a kind. Heartbeats at the band cadence or faster stay in
 * the 24/7 band and off the grid.
 *
 * Prev / Today / Next actually move the visible window. No speculative
 * "Schedule task" / "Filter" / "Export" CTAs — those don't belong on a
 * monitoring surface.
 *
 * Mounted by BOTH the studio CommandCenterShell and the classic ActivityPage
 * (dark / light / matte themes). Every `cc-cal-*` rule in globals.css is
 * scoped `:is(.studio, .cc-cal-root)`, so the `.cc-cal-root` wrapper below is
 * what styles the calendar when no `.studio` ancestor exists — drop it and
 * the classic Command Centre renders the grid as a stack of unstyled divs.
 *
 * This file holds CalendarTab's state; the model (calendar-model.ts), the view's
 * derived data (calendar-view.ts), the toolbar, the panels, the week and month
 * grids and the social post's drag and dialog (calendar-social*.ts) live beside it.
 */

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'

import { useActivitySchedule, useSchedulerHealth } from '@/hooks/use-activity-api'
import { useIsMobile } from '@/hooks/use-mobile'
import { useToggleHeartbeat } from '@/hooks/use-heartbeats-api'
import { useUpdateScheduledTaskStatus } from '@/hooks/use-scheduled-tasks-api'
import type { EventActionDeps } from './calendar-actions'
import type { ScheduleItemType } from './calendar-kinds'
import type { ViewMode } from './calendar-model'
import { MonthGrid } from './calendar-month-grid'
import { AlwaysOnBand, HealthBanner, LoadError, NextUp } from './calendar-panels'
import { useSocialReschedule } from './calendar-social-reschedule'
import { CalendarToolbar } from './calendar-toolbar'
import { shiftedAnchor, titleFor, useCalendarView } from './calendar-view'
import { WeekGrid } from './calendar-week-grid'

function useHiddenKinds() {
  // Legend chips hide a kind everywhere (grid, band, Next Up) for this view.
  const [hiddenKinds, setHiddenKinds] = useState<ReadonlySet<ScheduleItemType>>(() => new Set())
  const toggleKind = (kind: ScheduleItemType) =>
    setHiddenKinds((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  return { hiddenKinds, toggleKind }
}

export function CalendarTab() {
  const router = useRouter()
  // A week is `60px repeat(7, 1fr)` — 47px per day at 390px, which is not a
  // calendar. On a phone the choice is Day or Month and Week is not offered
  // (PRD-246 US-002); a Week preference set on a desktop reads as Day there.
  const isPhone = useIsMobile()
  const [preferred, setPreferred] = useState<ViewMode>('week')
  const mode: ViewMode = isPhone && preferred === 'week' ? 'day' : preferred
  const [anchor, setAnchor] = useState<Date>(() => new Date())

  const range = mode === 'month' ? '30d' : '7d'
  const { data: schedule, isLoading, isError, refetch } = useActivitySchedule(range)
  const { data: health } = useSchedulerHealth()
  const { mutate: toggleHeartbeat } = useToggleHeartbeat()
  const { mutate: updateScheduledTask } = useUpdateScheduledTaskStatus()
  const social = useSocialReschedule(() => void refetch())
  const { hiddenKinds, toggleKind } = useHiddenKinds()

  const actionDeps = useMemo<EventActionDeps>(
    () => ({
      navigate: (href) => router.push(href as any),
      pauseRoutine: (agentId) => toggleHeartbeat(agentId),
      setScheduledTaskStatus: (taskId, status) => updateScheduledTask({ taskId, status }),
      rescheduleSocialPost: social.open,
    }),
    [router, toggleHeartbeat, updateScheduledTask, social.open],
  )

  const items = useMemo(() => schedule?.scheduled ?? [], [schedule])
  const view = useCalendarView(items, hiddenKinds, mode, anchor)
  const nothing = view.events.length === 0
  const empty = isLoading && nothing ? 'loading' : !isLoading && !isError && nothing && view.alwaysOn.length === 0 ? 'none' : null

  return (
    <div className="cc-cal-root">
      <CalendarToolbar
        mode={mode}
        isPhone={isPhone}
        title={titleFor(mode, anchor, view.week)}
        hiddenKinds={hiddenKinds}
        onMode={setPreferred}
        onShift={(direction) => setAnchor((a) => shiftedAnchor(a, mode, direction))}
        onToggleKind={toggleKind}
      />
      {health?.healthy === false && <HealthBanner />}
      {isError && <LoadError onRetry={() => refetch()} />}
      {mode !== 'month' && view.alwaysOn.length > 0 && <AlwaysOnBand items={view.alwaysOn} actionDeps={actionDeps} />}
      {view.nextUp.length > 0 && <NextUp items={view.nextUp} />}
      {mode === 'month' ? (
        <MonthGrid cells={view.monthCells} events={view.events} anchorMonth={anchor.getMonth()} actionDeps={actionDeps} social={social} />
      ) : (
        <WeekGrid mode={mode} days={view.visibleDays} events={view.events} actionDeps={actionDeps} social={social} empty={empty} />
      )}
      {social.dialog}
    </div>
  )
}
