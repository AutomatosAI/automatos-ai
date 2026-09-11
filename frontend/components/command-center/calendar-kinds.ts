/**
 * Calendar kinds — what an item IS decides its colour; the agent is the dot.
 *
 * The feed (GET /api/activity/schedule) carries five kinds. Colouring by agent
 * (the old scheme) made a heartbeat, a playbook run and a deadline for the same
 * agent indistinguishable, and a calendar full of WATCHTOWER grey said nothing
 * about what was actually scheduled. The legend in the calendar toolbar uses
 * the same table; each chip hides its kind.
 */
import type { ScheduleItemType } from '@/hooks/use-activity-api'

export type { ScheduleItemType } from '@/hooks/use-activity-api'

export interface KindMeta {
  label: string
  tone: string
}

export const KIND_META: Record<ScheduleItemType, KindMeta> = {
  routine: { label: 'Heartbeat', tone: 'hsl(201 44% 40%)' },
  recipe: { label: 'Playbook', tone: 'hsl(272 30% 45%)' },
  task: { label: 'Scheduled task', tone: 'hsl(158 44% 35%)' },
  mission: { label: 'Mission SLA', tone: 'hsl(28 70% 45%)' },
  task_due: { label: 'Task deadline', tone: 'hsl(345 60% 45%)' },
}

/** Legend order: recurring things first, deadlines last. */
export const KIND_ORDER: readonly ScheduleItemType[] = ['routine', 'recipe', 'task', 'mission', 'task_due']

export function kindTone(type: ScheduleItemType): string {
  return (KIND_META[type] ?? KIND_META.routine).tone
}

export interface Laned<T> {
  evt: T
  /** 0-based column inside the overlap cluster */
  lane: number
  /** how many columns the cluster needs — every member gets the same */
  lanes: number
}

/**
 * Same-column overlap layout. Events that overlap in time share the column
 * width instead of stacking on top of each other; a cluster is a run of
 * transitively overlapping events, and every member takes 1/lanes of the
 * width. Returns the events sorted by start.
 */
export function layoutLanes<T>(
  events: readonly T[],
  span: (evt: T) => readonly [number, number],
): Laned<T>[] {
  const sorted = events
    .map((evt) => {
      const [start, end] = span(evt)
      return { evt, start, end }
    })
    .sort((a, b) => a.start - b.start || b.end - a.end)

  const out: Laned<T>[] = []
  let cluster: Array<{ evt: T; lane: number }> = []
  let laneEnds: number[] = []
  let clusterEnd = Number.NEGATIVE_INFINITY

  const flush = () => {
    const lanes = Math.max(laneEnds.length, 1)
    cluster.forEach((member) => out.push({ evt: member.evt, lane: member.lane, lanes }))
    cluster = []
    laneEnds = []
    clusterEnd = Number.NEGATIVE_INFINITY
  }

  for (const { evt, start, end } of sorted) {
    if (start >= clusterEnd) flush()
    const free = laneEnds.findIndex((laneEnd) => laneEnd <= start)
    const lane = free === -1 ? laneEnds.length : free
    laneEnds = free === -1 ? [...laneEnds, end] : laneEnds.map((v, i) => (i === lane ? end : v))
    cluster = [...cluster, { evt, lane }]
    clusterEnd = Math.max(clusterEnd, end)
  }
  flush()
  return out
}
