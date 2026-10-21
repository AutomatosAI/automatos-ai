'use client'

/**
 * The Month grid (split out of calendar-tab.tsx, PRD-251 US-307): a 6×7 grid with
 * up to three event chips a day. A social post's chip is draggable, and a cell
 * takes the drop at the post's time of day (US-307).
 *
 * PRD-251B US-B108: the Socials calendar reuses this grid. `social` is optional (no
 * drag without it) and `renderEvent`, when given, draws each event in place of the chip.
 */
import { Fragment, type ReactNode } from 'react'

import { buildEventActions, isPausedItem, type EventActionDeps } from './calendar-actions'
import { EventMenu } from './calendar-event-menu'
import { PAUSED_OPACITY, timeLabel, type CalEvent, type DayCell } from './calendar-model'
import type { SocialReschedule } from './calendar-social-reschedule'
import { DueTag, PausedTag, eventAccent } from './calendar-week-grid'

const WEEKDAYS = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
const CHIPS_PER_DAY = 3

interface MonthCellProps {
  day: DayCell
  events: CalEvent[]
  anchorMonth: number
  todayStart: number
  actionDeps: EventActionDeps
  social?: SocialReschedule
  renderEvent?: (evt: CalEvent) => ReactNode
}

function MonthChip({ evt, past, actionDeps, social }: { evt: CalEvent; past: boolean; actionDeps: EventActionDeps; social?: SocialReschedule }) {
  const overdue = Boolean(evt.due) && evt.date.getTime() < Date.now()
  const paused = isPausedItem(evt.item)
  // A past day's chip fades; a paused one (PRD-266 D27) is at most its own dimness.
  const opacity = Math.min(past && !evt.due ? 0.5 : 1, paused ? PAUSED_OPACITY : 1)
  return (
    <EventMenu actions={buildEventActions(evt.item, actionDeps)}>
      <button
        type="button"
        {...social?.dragProps(evt.item)}
        className="cc-cal-month-evt"
        data-kind={evt.item.type}
        data-paused={paused || undefined}
        style={{ borderLeftColor: eventAccent(evt, overdue), opacity }}
        title={evt.name}
      >
        <span className="t">{timeLabel(evt)}</span> {evt.due && <DueTag overdue={overdue} />}{' '}
        {paused && <PausedTag />}
        <span className="n2">{evt.name}</span>
      </button>
    </EventMenu>
  )
}

function MonthCell({ day, events, anchorMonth, todayStart, actionDeps, social, renderEvent }: MonthCellProps) {
  const dayEvents = events.filter((e) => e.dayKey === day.date.toDateString())
  const outside = day.month !== anchorMonth
  const past = day.date.getTime() < todayStart
  return (
    <div
      {...social?.dropOnDayProps(day.date)}
      className={`cc-cal-month-cell${outside ? ' outside' : ''}${day.today ? ' today' : ''}${past ? ' past' : ''}`}
    >
      <div className="n">{day.today ? <span>{day.n}</span> : day.n}</div>
      <div className="evts">
        {dayEvents.slice(0, CHIPS_PER_DAY).map((evt) =>
          renderEvent ? (
            <Fragment key={evt.id}>{renderEvent(evt)}</Fragment>
          ) : (
            <MonthChip key={evt.id} evt={evt} past={past} actionDeps={actionDeps} social={social} />
          ),
        )}
        {dayEvents.length > CHIPS_PER_DAY && <div className="cc-cal-month-more">+{dayEvents.length - CHIPS_PER_DAY} more</div>}
      </div>
    </div>
  )
}

export function MonthGrid({
  cells,
  events,
  anchorMonth,
  actionDeps,
  social,
  renderEvent,
}: {
  cells: DayCell[]
  events: CalEvent[]
  anchorMonth: number
  actionDeps: EventActionDeps
  social?: SocialReschedule
  renderEvent?: (evt: CalEvent) => ReactNode
}) {
  const todayStart = new Date()
  todayStart.setHours(0, 0, 0, 0)
  return (
    <div className="cc-cal-month">
      <div className="cc-cal-month-head">
        {WEEKDAYS.map((d) => (
          <div key={d} className="h">
            {d}
          </div>
        ))}
      </div>
      <div className="cc-cal-month-grid">
        {cells.map((d) => (
          <MonthCell
            key={d.iso}
            day={d}
            events={events}
            anchorMonth={anchorMonth}
            todayStart={todayStart.getTime()}
            actionDeps={actionDeps}
            social={social}
            renderEvent={renderEvent}
          />
        ))}
      </div>
    </div>
  )
}
