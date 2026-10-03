'use client'

/**
 * The Day and Week grid (split out of calendar-tab.tsx, PRD-251 US-307): hour rows,
 * a column per day, each event at its time with overlapping events sharing the
 * column, a crowded slot as one stacked card, deadlines with a DUE tag. A social
 * post's event is draggable and a day column takes the drop (US-307).
 *
 * PRD-251B US-B108: the Socials calendar reuses this grid. `social` is optional (no
 * drag without it) and `renderEvent`, when given, draws each single event in its box
 * (`eventBox`); a crowded slot stays one stacked card.
 */
import { Fragment, type CSSProperties, type ReactNode } from 'react'

import { toneFor } from './agent-tones'
import { buildEventActions, type EventActionDeps } from './calendar-actions'
import { EventMenu } from './calendar-event-menu'
import { KIND_META, collapseCrowded, kindTone, layoutLanes } from './calendar-kinds'
import {
  HOURS,
  HOUR_PX,
  MAX_LANES,
  MIN_EVENT_PX,
  OVERDUE_TONE,
  START_HR,
  eventSpan,
  hhmm,
  timeLabel,
  type CalEvent,
  type DayCell,
  type ViewMode,
} from './calendar-model'
import type { SocialReschedule } from './calendar-social-reschedule'

export function DueTag({ overdue }: { overdue: boolean }) {
  const tone = overdue ? 'hsl(var(--destructive))' : OVERDUE_TONE
  return (
    <span
      className="due"
      style={{
        marginLeft: 6,
        padding: '0 4px',
        borderRadius: 3,
        fontWeight: 700,
        letterSpacing: 0.4,
        background: overdue ? 'hsl(var(--destructive) / 0.15)' : 'hsl(45 80% 55% / 0.18)',
        color: tone,
      }}
    >
      {overdue ? 'OVERDUE' : 'DUE'}
    </span>
  )
}

/** A slot with more overlapping events than the column has lanes for: one
 *  card, the members' agent dots, and a submenu per member. */
function GroupCard({ members, start, end, actionDeps }: { members: CalEvent[]; start: number; end: number; actionDeps: EventActionDeps }) {
  const first = members[0]
  const kinds = new Set(members.map((m) => m.item.type))
  const kind = kinds.size === 1 ? first.item.type : null
  const what = kind ? KIND_META[kind].plural : 'items'
  const names = members.map((m) => (m.item.type === 'routine' && m.agent ? m.agent : m.name))
  const top = (first.hour - START_HR) * HOUR_PX + (first.min / 60) * HOUR_PX
  const height = Math.max(((end - start) / 3_600_000) * HOUR_PX, MIN_EVENT_PX)
  const groups = members.map((m) => ({ label: `${hhmm(m)} ${m.name}`, actions: buildEventActions(m.item, actionDeps) }))
  return (
    <EventMenu groups={groups}>
      <button
        type="button"
        className="cc-cal-event cc-cal-group"
        data-kind={kind ?? 'mixed'}
        data-group-size={members.length}
        style={{
          top,
          height,
          left: 'calc(0% + 3px)',
          width: 'calc(100% - 6px)',
          right: 'auto',
          borderLeftColor: kind ? kindTone(kind) : 'hsl(var(--muted-foreground))',
          background: 'hsl(var(--secondary))',
        }}
        title={`${members.length} ${what}: ${names.join(', ')} — click for actions`}
      >
        <div className="nm">
          {hhmm(first)} · {members.length} {what}
          <span className="agents">
            {members.slice(0, 8).map((m) => (
              <span key={m.id} className="agent-dot" style={{ background: toneFor(m.agent).bg }} />
            ))}
          </span>
        </div>
        <div className="ttl">{names.join(', ')}</div>
      </button>
    </EventMenu>
  )
}

interface GridEventProps {
  evt: CalEvent
  lane: number
  lanes: number
  actionDeps: EventActionDeps
  social?: SocialReschedule
}

/** Where an event's box sits in its day column: its time, its length, its lane. */
export function eventBox(evt: CalEvent, lane: number, lanes: number): CSSProperties {
  return {
    top: (evt.hour - START_HR) * HOUR_PX + (evt.min / 60) * HOUR_PX,
    height: Math.max((evt.durMin / 60) * HOUR_PX, MIN_EVENT_PX),
    // overlapping events share the column instead of stacking
    left: `calc(${(lane / lanes) * 100}% + 3px)`,
    width: `calc(${100 / lanes}% - 6px)`,
    right: 'auto',
  }
}

function GridEvent({ evt, lane, lanes, actionDeps, social }: GridEventProps) {
  const tone = toneFor(evt.agent)
  const overdue = Boolean(evt.due) && evt.date.getTime() < Date.now()
  const kind = KIND_META[evt.item.type]?.label ?? evt.item.type
  return (
    <EventMenu actions={buildEventActions(evt.item, actionDeps)}>
      <button
        type="button"
        {...social?.dragProps(evt.item)}
        className="cc-cal-event"
        data-kind={evt.item.type}
        data-lane={lane}
        data-lanes={lanes}
        style={{
          ...eventBox(evt, lane, lanes),
          borderLeftColor: overdue ? OVERDUE_TONE : kindTone(evt.item.type),
          background: 'hsl(var(--secondary))',
        }}
        title={`${kind}: ${evt.name} — click for actions`}
      >
        <div className="nm">
          {timeLabel(evt)}
          {evt.agent && (
            <span style={{ marginLeft: 6, opacity: 0.85 }}>
              <span className="agent-dot" style={{ background: tone.bg }} />
              {evt.agent}
            </span>
          )}
          {evt.due && <DueTag overdue={overdue} />}
        </div>
        <div className="ttl">{evt.name}</div>
      </button>
    </EventMenu>
  )
}

interface DayColumnProps {
  day: DayCell
  column: number
  events: CalEvent[]
  mode: ViewMode
  nowHourPos: number
  actionDeps: EventActionDeps
  social?: SocialReschedule
  renderEvent?: (evt: CalEvent, lane: number, lanes: number) => ReactNode
}

function DayColumn({ day, column, events, mode, nowHourPos, actionDeps, social, renderEvent }: DayColumnProps) {
  const dayEvents = events.filter((e) => e.dayKey === day.date.toDateString())
  const placed = collapseCrowded(layoutLanes(dayEvents, eventSpan), MAX_LANES[mode], eventSpan)
  return (
    <div
      {...social?.dropProps(day.date, START_HR, HOUR_PX)}
      className={`cc-cal-daycol${day.today ? ' today' : ''}`}
      style={{ gridColumn: column + 2, gridRow: `1 / span ${HOURS.length}`, height: HOURS.length * HOUR_PX }}
    >
      {placed.map((p) =>
        p.kind === 'group' ? (
          <GroupCard key={`group-${p.start}`} members={p.members} start={p.start} end={p.end} actionDeps={actionDeps} />
        ) : renderEvent ? (
          <Fragment key={p.evt.id}>{renderEvent(p.evt, p.lane, p.lanes)}</Fragment>
        ) : (
          <GridEvent key={p.evt.id} evt={p.evt} lane={p.lane} lanes={p.lanes} actionDeps={actionDeps} social={social} />
        ),
      )}
      {day.today && <div className="cc-cal-nowline" style={{ top: nowHourPos }} />}
    </div>
  )
}

function HourLabels() {
  return (
    <div style={{ gridColumn: 1, gridRow: `1 / span ${HOURS.length}`, borderRight: '1px solid hsl(var(--border))' }}>
      {HOURS.map((h) => (
        <div key={h} className="cc-cal-hourlabel" style={{ height: HOUR_PX }}>
          {String(h).padStart(2, '0')}:00
        </div>
      ))}
    </div>
  )
}

interface WeekGridProps {
  mode: ViewMode
  days: DayCell[]
  events: CalEvent[]
  actionDeps: EventActionDeps
  social?: SocialReschedule
  /** Nothing to show: loading, or an empty window (and no 24/7 band). */
  empty: 'loading' | 'none' | null
  renderEvent?: (evt: CalEvent, lane: number, lanes: number) => ReactNode
}

export function WeekGrid({ mode, days, events, actionDeps, social, empty, renderEvent }: WeekGridProps) {
  const columns = mode === 'day' ? '60px 1fr' : '60px repeat(7, 1fr)'
  const now = new Date()
  const nowHourPos = (now.getHours() - START_HR) * HOUR_PX + (now.getMinutes() / 60) * HOUR_PX
  return (
    <div className="cc-cal-frame">
      <div className="cc-cal-head" style={{ gridTemplateColumns: columns }}>
        <div className="h" />
        {days.map((d) => (
          <div key={d.iso} className={`h${d.today ? ' today' : ''}`}>
            <div className="d">{d.short}</div>
            <div className="n">{d.today ? <span>{d.n}</span> : d.n}</div>
          </div>
        ))}
      </div>
      <div
        className="cc-cal-grid"
        style={{
          gridTemplateColumns: columns,
          gridTemplateRows: `repeat(${HOURS.length}, ${HOUR_PX}px)`,
          backgroundSize: `100% ${HOUR_PX}px`,
          backgroundImage: `linear-gradient(to bottom, transparent 0, transparent calc(${HOUR_PX}px - 1px), hsl(var(--border)) calc(${HOUR_PX}px - 1px), hsl(var(--border)) ${HOUR_PX}px)`,
        }}
      >
        <HourLabels />
        {days.map((d, di) => (
          <DayColumn
            key={d.iso}
            day={d}
            column={di}
            events={events}
            mode={mode}
            nowHourPos={nowHourPos}
            actionDeps={actionDeps}
            social={social}
            renderEvent={renderEvent}
          />
        ))}
      </div>
      {empty === 'loading' && <div className="cc-panel-empty">Loading schedule…</div>}
      {empty === 'none' && (
        <div className="cc-panel-empty">
          No scheduled work in this window. Enable a heartbeat in /agents
          to populate the calendar.
        </div>
      )}
    </div>
  )
}
