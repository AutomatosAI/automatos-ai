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
  /** for a stacked card: "5 heartbeats" */
  plural: string
  tone: string
}

export const KIND_META: Record<ScheduleItemType, KindMeta> = {
  routine: { label: 'Heartbeat', plural: 'heartbeats', tone: 'hsl(201 44% 40%)' },
  recipe: { label: 'Playbook', plural: 'playbooks', tone: 'hsl(272 30% 45%)' },
  task: { label: 'Scheduled task', plural: 'scheduled tasks', tone: 'hsl(158 44% 35%)' },
  mission: { label: 'Mission SLA', plural: 'mission SLAs', tone: 'hsl(28 70% 45%)' },
  task_due: { label: 'Task deadline', plural: 'deadlines', tone: 'hsl(345 60% 45%)' },
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
  /** which run of overlapping events this belongs to (0-based, by start) */
  cluster: number
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
  let clusterId = 0

  const flush = () => {
    if (cluster.length === 0) return
    const lanes = laneEnds.length
    cluster.forEach((member) =>
      out.push({ evt: member.evt, lane: member.lane, lanes, cluster: clusterId }),
    )
    clusterId += 1
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

export type Placed<T> =
  | { kind: 'single'; evt: T; lane: number; lanes: number }
  | { kind: 'group'; members: T[]; start: number; end: number }

/**
 * A cluster that needs more lanes than the column can show legibly becomes
 * one stacked card (five daily heartbeats at 10:00 were five unreadable
 * slivers in the week view); every other event keeps its lane.
 */
export function collapseCrowded<T>(
  laid: readonly Laned<T>[],
  maxLanes: number,
  span: (evt: T) => readonly [number, number],
): Placed<T>[] {
  const crowded = new Set(laid.filter((l) => l.lanes > maxLanes).map((l) => l.cluster))
  const folded = new Set<number>()
  const out: Placed<T>[] = []
  for (const l of laid) {
    if (!crowded.has(l.cluster)) {
      out.push({ kind: 'single', evt: l.evt, lane: l.lane, lanes: l.lanes })
      continue
    }
    if (folded.has(l.cluster)) continue
    folded.add(l.cluster)
    const members = laid.filter((m) => m.cluster === l.cluster).map((m) => m.evt)
    const spans = members.map(span)
    out.push({
      kind: 'group',
      members,
      start: Math.min(...spans.map((s) => s[0])),
      end: Math.max(...spans.map((s) => s[1])),
    })
  }
  return out
}
