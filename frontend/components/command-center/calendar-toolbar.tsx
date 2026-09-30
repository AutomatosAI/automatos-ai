'use client'

/**
 * The calendar's toolbar (split out of calendar-tab.tsx, PRD-251 US-307): Day /
 * Week / Month (no Week on a phone, PRD-246 US-002), Prev / Today / Next, the
 * window's title and the legend, whose chips hide a kind.
 */
import { ChevronLeft, ChevronRight } from 'lucide-react'

import { KIND_META, KIND_ORDER, type ScheduleItemType } from './calendar-kinds'
import type { ViewMode } from './calendar-model'

interface CalendarToolbarProps {
  mode: ViewMode
  isPhone: boolean
  title: string
  hiddenKinds: ReadonlySet<ScheduleItemType>
  onMode: (mode: ViewMode) => void
  onShift: (direction: -1 | 0 | 1) => void
  onToggleKind: (kind: ScheduleItemType) => void
}

function ModeSwitch({ mode, isPhone, onMode }: Pick<CalendarToolbarProps, 'mode' | 'isPhone' | 'onMode'>) {
  const modes: Array<[ViewMode, string]> = isPhone
    ? [['day', 'Day'], ['month', 'Month']]
    : [['day', 'Day'], ['week', 'Week'], ['month', 'Month']]
  return (
    <div className="cc-seg" role="group" aria-label="Calendar mode">
      {modes.map(([value, label]) => (
        <button key={value} type="button" className={mode === value ? 'on' : ''} onClick={() => onMode(value)}>
          {label}
        </button>
      ))}
    </div>
  )
}

function Navigation({ onShift }: Pick<CalendarToolbarProps, 'onShift'>) {
  return (
    <div style={{ display: 'inline-flex', gap: 4 }}>
      <button type="button" className="cc-btn" style={{ width: 30, padding: 0 }} onClick={() => onShift(-1)} aria-label="Previous">
        <ChevronLeft style={{ width: 13, height: 13 }} />
      </button>
      <button type="button" className="cc-btn" style={{ fontSize: 11.5, padding: '0 10px' }} onClick={() => onShift(0)}>
        Today
      </button>
      <button type="button" className="cc-btn" style={{ width: 30, padding: 0 }} onClick={() => onShift(1)} aria-label="Next">
        <ChevronRight style={{ width: 13, height: 13 }} />
      </button>
    </div>
  )
}

function Legend({ hiddenKinds, onToggleKind }: Pick<CalendarToolbarProps, 'hiddenKinds' | 'onToggleKind'>) {
  return (
    <div className="cc-cal-legend" role="group" aria-label="Show on calendar">
      {KIND_ORDER.map((kind) => {
        const hidden = hiddenKinds.has(kind)
        const { label, tone } = KIND_META[kind]
        return (
          <button
            key={kind}
            type="button"
            className={hidden ? 'off' : ''}
            aria-pressed={!hidden}
            onClick={() => onToggleKind(kind)}
            title={hidden ? `Show ${label}` : `Hide ${label}`}
          >
            <span className="sw" style={{ background: tone }} />
            {label}
          </button>
        )
      })}
    </div>
  )
}

export function CalendarToolbar(props: CalendarToolbarProps) {
  return (
    <div className="cc-cal-toolbar">
      <ModeSwitch mode={props.mode} isPhone={props.isPhone} onMode={props.onMode} />
      <Navigation onShift={props.onShift} />
      <span
        style={{
          fontFamily: 'var(--font-newsreader, serif)',
          fontSize: 17,
          fontWeight: 500,
          color: 'hsl(var(--foreground))',
        }}
      >
        {props.title}
      </span>
      <Legend hiddenKinds={props.hiddenKinds} onToggleKind={props.onToggleKind} />
    </div>
  )
}
