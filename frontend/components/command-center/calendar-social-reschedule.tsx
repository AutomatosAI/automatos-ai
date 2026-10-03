'use client'

/**
 * PRD-251 US-307 (D10) — rescheduling a social post from the calendar: the
 * Reschedule… dialog (a wall time in the post's own timezone) and drag and drop
 * (native HTML5 drag events: a social item is draggable, a day column takes the
 * drop at the quarter hour under the pointer). Both call
 * POST /api/socials/posts/{id}/schedule with the post's timezone: the slot moves,
 * the approval stands. calendar-tab.tsx only wires these in.
 */
import { useState, type DragEvent, type HTMLAttributes, type ReactNode } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import type { ScheduleItem } from '@/hooks/use-activity-api'
import { apiClient } from '@/lib/api-client'
import { SOCIAL_DRAG_TYPE, isSocialItem, slotFromDrop, slotOnDay, wallInZone, zonedWallToIso } from './calendar-social'

const DEFAULT_TZ = 'UTC'
const HTTP_CONFLICT = 409
export const RESCHEDULED_MESSAGE = 'Rescheduled. The approval stands.'
export const POST_CHANGED_MESSAGE = 'This post changed since the calendar loaded it. The calendar has been refreshed.'

type Reschedule = (postId: string, scheduledFor: string, timezone: string) => Promise<void>

/** PRD-251B US-B108: how a move is written, and what it says. The Command Center's
 * default is POST /schedule (a scheduled post); the Socials calendar moves any post's
 * slot through PUT /slot, which keeps an approval and schedules an approved post. */
export interface RescheduleOptions {
  move?: (postId: string, slot: string, timezone: string) => Promise<unknown>
  movedMessage?: string
}

/** POST /schedule (or the given move), then refresh the calendar however it went: a 409 says the post changed. */
function useRescheduleCall(onChanged: () => void, options: RescheduleOptions): Reschedule {
  const move = options.move ?? ((postId, slot, timezone) => apiClient.scheduleSocialPost(postId, slot, timezone))
  return async (postId, scheduledFor, timezone) => {
    try {
      await move(postId, scheduledFor, timezone)
      toast.success(options.movedMessage ?? RESCHEDULED_MESSAGE)
    } catch (error) {
      const status = (error as { status?: unknown } | null)?.status
      toast.error(status === HTTP_CONFLICT ? POST_CHANGED_MESSAGE : (error as Error)?.message || 'The post could not be rescheduled')
    } finally {
      onChanged()
    }
  }
}

export interface SocialReschedule {
  /** Open the dialog for a social item (the Reschedule… menu item). */
  open: (item: ScheduleItem) => void
  /** A social item's drag props; nothing for any other kind. */
  dragProps: (item: ScheduleItem) => HTMLAttributes<HTMLElement>
  /** A day column's drop props: `startHour` and `hourPx` are the grid's. */
  dropProps: (day: Date, startHour: number, hourPx: number) => HTMLAttributes<HTMLElement>
  /** A month cell's drop props: the post keeps its time of day, on `day`. */
  dropOnDayProps: (day: Date) => HTMLAttributes<HTMLElement>
  dialog: ReactNode
}

/** What a dragged social item carries: the post, and the timezone its slot keeps. */
interface DragPayload {
  postId: string
  timezone: string
  /** The slot it is dragged from (ISO): a month cell keeps its time of day. */
  slot: string | null
}

function readPayload(raw: string): DragPayload | null {
  try {
    const parsed = JSON.parse(raw) as Partial<DragPayload>
    return parsed.postId ? { postId: parsed.postId, timezone: parsed.timezone || DEFAULT_TZ, slot: parsed.slot ?? null } : null
  } catch {
    return null
  }
}

function RescheduleDialog({ item, onClose, reschedule }: { item: ScheduleItem | null; onClose: () => void; reschedule: Reschedule }) {
  const tz = item?.timezone || DEFAULT_TZ
  const [wall, setWall] = useState('')
  const [busy, setBusy] = useState(false)
  const current = item?.next_run_at ? wallInZone(item.next_run_at, tz) : ''
  const value = wall || current
  const submit = async () => {
    if (!item?.post_id || !value) return
    setBusy(true)
    await reschedule(item.post_id, zonedWallToIso(value, tz), tz)
    setBusy(false)
    setWall('')
    onClose()
  }
  return (
    <Dialog open={item !== null} onOpenChange={(openNow) => !openNow && (setWall(''), onClose())}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reschedule {item?.name}</DialogTitle>
          <DialogDescription>The new slot is in the post&apos;s timezone, {tz}. The approval stands.</DialogDescription>
        </DialogHeader>
        <Input type="datetime-local" aria-label="New slot" value={value} onChange={(e) => setWall(e.target.value)} />
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={submit} disabled={!value || busy}>Reschedule</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** ``onChanged`` refreshes the calendar after a reschedule (its feed's refetch). */
export function useSocialReschedule(onChanged: () => void, options: RescheduleOptions = {}): SocialReschedule {
  const [editing, setEditing] = useState<ScheduleItem | null>(null)
  const reschedule = useRescheduleCall(onChanged, options)

  const dragProps = (item: ScheduleItem): HTMLAttributes<HTMLElement> => {
    if (!isSocialItem(item)) return {}
    return {
      draggable: true,
      onDragStart: (e: DragEvent<HTMLElement>) => {
        const payload: DragPayload = {
          postId: item.post_id as string,
          timezone: item.timezone || DEFAULT_TZ,
          slot: item.next_run_at,
        }
        e.dataTransfer.setData(SOCIAL_DRAG_TYPE, JSON.stringify(payload))
        e.dataTransfer.effectAllowed = 'move'
      },
    }
  }

  const dropTarget = (slotFor: (e: DragEvent<HTMLElement>, payload: DragPayload) => Date): HTMLAttributes<HTMLElement> => ({
    onDragOver: (e: DragEvent<HTMLElement>) => {
      if (Array.from(e.dataTransfer.types).includes(SOCIAL_DRAG_TYPE)) e.preventDefault()
    },
    onDrop: (e: DragEvent<HTMLElement>) => {
      const payload = readPayload(e.dataTransfer.getData(SOCIAL_DRAG_TYPE))
      if (!payload) return
      e.preventDefault()
      const slot = slotFor(e, payload)
      void reschedule(payload.postId, slot.toISOString(), payload.timezone)
    },
  })
  const dropProps = (day: Date, startHour: number, hourPx: number) =>
    dropTarget((e) => slotFromDrop(day, (e.clientY || 0) - e.currentTarget.getBoundingClientRect().top, startHour, hourPx))
  const dropOnDayProps = (day: Date) => dropTarget((_e, payload) => slotOnDay(day, payload.slot, payload.timezone))

  return {
    open: setEditing,
    dragProps,
    dropProps,
    dropOnDayProps,
    dialog: <RescheduleDialog item={editing} onClose={() => setEditing(null)} reschedule={reschedule} />,
  }
}
