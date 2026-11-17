/**
 * PRD-251 Wave 3, US-307 (S3.1b, D10) — a coming social post in the Command
 * Center calendar: the `social` kind renders with its label and colour, its time
 * shows in the post's own timezone, it opens the post, and Reschedule… and a drop
 * call PUT /api/socials/posts/{id}/slot with the new slot and the post's
 * timezone (the slot moves, an approval stands). No npm dependency: native HTML5
 * drag events.
 *
 * 16 Nov 2026: the feed carries every post still to go out, not only a scheduled one. A
 * post that is not scheduled yet says where it stands, one waiting for approval opens in
 * the Socials Queue, and a move goes through the one route that moves a post of any status.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, createEvent, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { readFileSync } from 'fs'
import path from 'path'
import { toast } from 'sonner'
import type { ScheduleItem } from '@/hooks/use-activity-api'

const feed = vi.hoisted(() => ({ items: [] as unknown[] }))
const nav = vi.hoisted(() => ({ push: vi.fn() }))
const api = vi.hoisted(() => ({ scheduleSocialPost: vi.fn(), setSocialPostSlot: vi.fn() }))
const socials = vi.hoisted(() => ({ changed: vi.fn(async () => []) }))

vi.mock('next/navigation', () => ({ useRouter: () => nav }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))
vi.mock('@/lib/api-client', () => ({ apiClient: api, default: api }))
vi.mock('@/hooks/use-activity-api', () => ({
  activityQueryKeys: { all: ['activity'] },
  useActivitySchedule: () => ({ data: { scheduled: feed.items }, isLoading: false, isError: false, refetch: vi.fn() }),
  useSchedulerHealth: () => ({ data: { healthy: null, last_fired_at: null } }),
}))
vi.mock('@/hooks/use-socials-api', () => ({ useInvalidateSocials: () => socials.changed }))
vi.mock('@/hooks/use-heartbeats-api', () => ({ useToggleHeartbeat: () => ({ mutate: vi.fn() }) }))
vi.mock('@/hooks/use-scheduled-tasks-api', () => ({ useUpdateScheduledTaskStatus: () => ({ mutate: vi.fn() }) }))

import { CalendarTab } from '../calendar-tab'
import { KIND_META, KIND_ORDER, kindTone } from '../calendar-kinds'
import { buildEventActions, socialPostHref } from '../calendar-actions'
import { PUBLISHES_THERE_MESSAGE, RESCHEDULED_MESSAGE } from '../calendar-social-reschedule'
import { SOCIAL_DRAG_TYPE, slotFromDrop, slotOnDay, socialChipLabel, socialTimeLabel, wallInZone, zonedWallToIso } from '../calendar-social'

/** Today at 10:00 local: inside the visible week, far from a day boundary. */
function todayAt(hours: number): Date {
  const d = new Date()
  d.setHours(hours, 0, 0, 0)
  return d
}

function socialItem(slot: Date, timezone = 'Asia/Tokyo', status?: string): ScheduleItem {
  return {
    status,
    id: 'social-p1',
    post_id: 'p1',
    type: 'social',
    name: 'Harvest Club opens',
    next_run_at: slot.toISOString(),
    timezone,
    frequency: 'One-off',
    agent_name: null,
    agent_id: null,
  }
}

function renderCalendar() {
  return render(<CalendarTab />)
}

/** A DataTransfer stand-in: jsdom has none. */
function dataTransfer() {
  const store = new Map<string, string>()
  return {
    setData: (type: string, value: string) => store.set(type, value),
    getData: (type: string) => store.get(type) ?? '',
    get types() {
      return Array.from(store.keys())
    },
    effectAllowed: 'none',
    dropEffect: 'none',
  }
}

beforeEach(() => {
  api.scheduleSocialPost.mockReset()
  api.setSocialPostSlot.mockReset().mockResolvedValue({ id: 'p1', status: 'scheduled' })
  socials.changed.mockClear()
  vi.mocked(toast.success).mockClear()
  nav.push.mockReset()
})
afterEach(() => cleanup())

describe('the social kind', () => {
  it('has its own label and colour in the legend', () => {
    expect(KIND_ORDER).toContain('social')
    expect(KIND_META.social.label).toBe('Social post')
    const tones = KIND_ORDER.map((k) => KIND_META[k].tone)
    expect(new Set(tones).size).toBe(tones.length)
    expect(kindTone('social')).toBe(KIND_META.social.tone)
  })

  it('shows the time in the post\'s own timezone', () => {
    const slot = new Date(Date.UTC(2026, 10, 9, 9, 0))
    expect(socialTimeLabel({ next_run_at: slot.toISOString(), timezone: 'Asia/Tokyo' })).toBe('18:00 GMT+9')
    expect(socialTimeLabel({ next_run_at: slot.toISOString(), timezone: 'UTC' })).toBe('09:00 UTC')
  })

  it('opens the post, and offers Reschedule', () => {
    const deps = { navigate: vi.fn(), pauseRoutine: vi.fn(), setScheduledTaskStatus: vi.fn(), rescheduleSocialPost: vi.fn() }
    const item = socialItem(todayAt(10))
    const actions = buildEventActions(item, deps)
    expect(actions.map((a) => a.label)).toEqual(['Open post', 'Reschedule…'])
    actions[0].run()
    expect(deps.navigate).toHaveBeenCalledWith('/deliverables?tab=socials&post=p1')
    expect(socialPostHref('p1')).toBe('/deliverables?tab=socials&post=p1')
    actions[1].run()
    expect(deps.rescheduleSocialPost).toHaveBeenCalledWith(item)
  })

  it('says where a post stands until it is scheduled', () => {
    const slot = new Date(Date.UTC(2026, 10, 9, 9, 0)).toISOString()
    const at = { next_run_at: slot, timezone: 'UTC' }
    expect(socialChipLabel({ ...at, status: 'needs_approval' })).toBe('09:00 UTC · Needs approval')
    expect(socialChipLabel({ ...at, status: 'draft' })).toBe('09:00 UTC · Draft')
    expect(socialChipLabel({ ...at, status: 'approved' })).toBe('09:00 UTC · Approved')
    expect(socialChipLabel({ ...at, status: 'scheduled' })).toBe('09:00 UTC')
    expect(socialChipLabel(at)).toBe('09:00 UTC')
  })

  it('opens a post waiting for approval in the Socials Queue, where it is approved', () => {
    const deps = { navigate: vi.fn(), pauseRoutine: vi.fn(), setScheduledTaskStatus: vi.fn(), rescheduleSocialPost: vi.fn() }
    buildEventActions(socialItem(todayAt(10), 'UTC', 'needs_approval'), deps)[0].run()
    expect(deps.navigate).toHaveBeenCalledWith('/deliverables?tab=socials&view=queue&post=p1')
    expect(socialPostHref('p1', 'draft')).toBe('/deliverables?tab=socials&post=p1')
  })
})

describe('the calendar with a scheduled post', () => {
  it('renders the post as a social event with the post\'s timezone and its colour', () => {
    const slot = todayAt(10)
    feed.items = [socialItem(slot)]
    renderCalendar()
    const event = document.querySelector('.cc-cal-event[data-kind="social"]') as HTMLElement
    expect(event).toBeTruthy()
    expect(event.textContent).toContain('Harvest Club opens')
    expect(event.textContent).toContain(socialTimeLabel(socialItem(slot)))
    expect(event.getAttribute('draggable')).toBe('true')
    expect(screen.getAllByText('Social post').length).toBeGreaterThan(0) // the legend chip
  })

  it('renders a post waiting for approval at its planned slot, saying so, and it can be moved', async () => {
    api.setSocialPostSlot.mockResolvedValue({ id: 'p1', status: 'needs_approval' })
    feed.items = [socialItem(todayAt(10), 'UTC', 'needs_approval')]
    renderCalendar()
    const event = document.querySelector('.cc-cal-event[data-kind="social"]') as HTMLElement
    expect(event.textContent).toContain('Needs approval')
    expect(event.getAttribute('draggable')).toBe('true')

    const transfer = dataTransfer()
    fireEvent.dragStart(event, { dataTransfer: transfer })
    fireEvent.drop(event.closest('.cc-cal-daycol') as HTMLElement, { dataTransfer: transfer })
    await waitFor(() => expect(api.setSocialPostSlot).toHaveBeenCalledTimes(1))
    // moved, and still waiting: nothing publishes
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(RESCHEDULED_MESSAGE))
  })

  it('a drop on a day column moves the slot with the post\'s timezone', async () => {
    const slot = todayAt(10)
    feed.items = [socialItem(slot)]
    renderCalendar()
    const event = document.querySelector('.cc-cal-event[data-kind="social"]') as HTMLElement
    const column = event.closest('.cc-cal-daycol') as HTMLElement
    const transfer = dataTransfer()
    fireEvent.dragStart(event, { dataTransfer: transfer })
    expect(JSON.parse(transfer.getData(SOCIAL_DRAG_TYPE))).toMatchObject({ postId: 'p1', timezone: 'Asia/Tokyo' })
    fireEvent.dragOver(column, { dataTransfer: transfer })
    const drop = createEvent.drop(column, { dataTransfer: transfer })
    Object.defineProperty(drop, 'clientY', { value: 44 * 14.5 }) // jsdom's drop event has no pointer position
    fireEvent(column, drop)

    await waitFor(() => expect(api.setSocialPostSlot).toHaveBeenCalledTimes(1))
    const expected = slotFromDrop(slot, 44 * 14.5, 0, 44) // 14:30 that day
    expect(expected.getHours()).toBe(14)
    expect(api.setSocialPostSlot).toHaveBeenCalledWith('p1', expected.toISOString(), 'Asia/Tokyo')
    expect(api.scheduleSocialPost).not.toHaveBeenCalled() // PUT /slot moves a post of any status
    // The move is a change in Socials too: its calendar and this one both read the post again.
    await waitFor(() => expect(socials.changed).toHaveBeenCalledTimes(1))
    // the answer says it is scheduled
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(PUBLISHES_THERE_MESSAGE))
  })

  it('Reschedule… sends the new wall time in the post\'s timezone', async () => {
    const slot = todayAt(10)
    feed.items = [socialItem(slot, 'Europe/Lisbon')]
    renderCalendar()
    const event = document.querySelector('.cc-cal-event[data-kind="social"]') as HTMLElement
    fireEvent.keyDown(event, { key: 'Enter' })
    fireEvent.click(await screen.findByText('Reschedule…'))
    const input = (await screen.findByLabelText('New slot')) as HTMLInputElement
    expect(input.value).toBe(wallInZone(slot.toISOString(), 'Europe/Lisbon'))
    fireEvent.change(input, { target: { value: '2026-11-09T09:30' } })
    fireEvent.click(screen.getByRole('button', { name: 'Reschedule' }))

    await waitFor(() => expect(api.setSocialPostSlot).toHaveBeenCalledTimes(1))
    expect(api.setSocialPostSlot).toHaveBeenCalledWith('p1', '2026-11-09T09:30:00.000Z', 'Europe/Lisbon') // Lisbon is UTC+0 in November
  })
})

describe('time zones without a library', () => {
  it('turns a wall time in a zone into the instant it names, and back', () => {
    expect(zonedWallToIso('2026-07-01T09:00', 'Europe/Lisbon')).toBe('2026-07-01T08:00:00.000Z') // summer: UTC+1
    expect(zonedWallToIso('2026-11-09T18:00', 'Asia/Tokyo')).toBe('2026-11-09T09:00:00.000Z')
    expect(wallInZone('2026-11-09T09:00:00.000Z', 'Asia/Tokyo')).toBe('2026-11-09T18:00')
  })

  it('a Month-view drop keeps the post\'s own time of day, in its timezone, on the new date', () => {
    // 20:00 UTC on 9 Nov is 05:00 on 10 Nov in Tokyo: the post's time is 05:00, Tokyo.
    const moved = slotOnDay(new Date(2026, 10, 20), '2026-11-09T20:00:00Z', 'Asia/Tokyo')
    expect(wallInZone(moved.toISOString(), 'Asia/Tokyo')).toBe('2026-11-20T05:00')
    const noSlot = slotOnDay(new Date(2026, 10, 20), null, 'Europe/Lisbon')
    expect(wallInZone(noSlot.toISOString(), 'Europe/Lisbon')).toBe('2026-11-20T12:00')
  })

  it('snaps a drop to the quarter hour', () => {
    const day = new Date(2026, 10, 9)
    expect(slotFromDrop(day, 44 * 9 + 10, 0, 44).getMinutes()).toBe(15)
    expect(slotFromDrop(day, 0, 0, 44).getHours()).toBe(0)
  })
})

describe('no new dependency', () => {
  it('drag uses native HTML5 events: the calendar imports no drag library', () => {
    for (const file of ['calendar-social-reschedule.tsx', 'calendar-social.ts', 'calendar-tab.tsx']) {
      const src = readFileSync(path.resolve(__dirname, '..', file), 'utf8')
      expect(src).not.toMatch(/from ['"][^'"]*(dnd|drag|sortable)[^'"]*['"]/i)
    }
    expect(readFileSync(path.resolve(__dirname, '..', 'calendar-social-reschedule.tsx'), 'utf8')).toContain('onDragStart')
  })
})
