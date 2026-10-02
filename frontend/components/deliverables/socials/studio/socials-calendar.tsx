'use client'

/**
 * PRD-251B US-B108 — the Socials calendar (Main.dc.html): Month · Week · List over the
 * workspace's posts, on THE COMMAND CENTER'S GRID (MonthGrid, WeekGrid and their model),
 * each post drawn as a Socials chip, with the Today rail beside it. A post sits at its
 * planned slot (B11; a scheduled or published one at its schedule) in its own timezone.
 *
 * Drag a movable post to another day (Month keeps its time of day; Week takes the quarter
 * hour under the pointer): the move is PUT /slot (US-B105), which before approval sets the
 * slot alone and on an approved or scheduled post reschedules it through the one schedule
 * path, so the planned slot and the schedule never part. List is the post list (List |
 * Board). A click opens the post; a post waiting for approval opens in the Queue.
 */
import { useMemo, useState, type CSSProperties } from 'react'
import { useRouter } from 'next/navigation'

import type { Workspace } from '@/components/workspace-provider'
import { apiClient, type SocialPost } from '@/lib/api-client'
import { useInvalidateSocials } from '@/hooks/use-socials-api'
import { useSocialChannels } from '@/hooks/use-socials-composer'
import type { EventActionDeps } from '@/components/command-center/calendar-actions'
import { buildMonthGrid, buildWeek, type CalEvent } from '@/components/command-center/calendar-model'
import { MonthGrid } from '@/components/command-center/calendar-month-grid'
import { useSocialReschedule, type SocialReschedule } from '@/components/command-center/calendar-social-reschedule'
import { shiftedAnchor, titleFor, windowSpanFor } from '@/components/command-center/calendar-view'
import { WeekGrid, eventBox } from '@/components/command-center/calendar-week-grid'
import { SocialsPostList } from '../socials-post-list'
import { SocialsCalendarChip } from './socials-calendar-chip'
import { SocialsCalendarHeader } from './socials-calendar-header'
import { ALL_CHANNELS, isMovable, matchesFilter, opensInQueue, postEvents, type ChannelFilter } from './socials-calendar-model'
import { SocialsTodayRail } from './socials-today-rail'
import { NEW_POST, type GoTo, type SocialsRoute } from './studio-route'

export const MOVED_MESSAGE = 'Moved to the new slot.'
export const DRAG_HINT = 'Drag a post to another day to move it. Click a post to open it.'
const LIST_TITLE = 'All posts'

interface SocialsCalendarProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  route: SocialsRoute
  go: GoTo
}

function useCalendarMoves(): SocialReschedule {
  const invalidate = useInvalidateSocials()
  return useSocialReschedule(() => void invalidate(), {
    move: (postId, slot, timezone) => apiClient.setSocialPostSlot(postId, slot, timezone),
    movedMessage: MOVED_MESSAGE,
  })
}

/** What the grid's own menus (a crowded slot's stacked card) can do with a post. */
function useActionDeps(social: SocialReschedule, openId: (postId: string) => void): EventActionDeps {
  const router = useRouter()
  return useMemo(
    () => ({
      navigate: (href: string) => {
        const postId = new URLSearchParams(href.split('?')[1] ?? '').get('post')
        if (postId) openId(postId)
        else router.push(href as any)
      },
      pauseRoutine: () => undefined,
      setScheduledTaskStatus: () => undefined,
      rescheduleSocialPost: social.open,
    }),
    [router, social.open, openId],
  )
}

export function SocialsCalendar({ role, posts, route, go }: SocialsCalendarProps) {
  const [anchor, setAnchor] = useState(() => new Date())
  const [filter, setFilter] = useState<ChannelFilter>(ALL_CHANNELS)
  const { data: channels } = useSocialChannels()
  const social = useCalendarMoves()
  const layout = route.post && route.post !== NEW_POST ? 'list' : route.cal
  const mode = layout === 'week' ? 'week' : 'month'

  const shown = useMemo(() => posts.filter((post) => matchesFilter(post, filter)), [posts, filter])
  const byId = useMemo(() => new Map(posts.map((post) => [post.id, post])), [posts])
  const week = useMemo(() => buildWeek(anchor), [anchor])
  const cells = useMemo(() => buildMonthGrid(anchor), [anchor])
  const events = useMemo(() => postEvents(shown, windowSpanFor(mode, anchor)), [shown, mode, anchor])

  const open = (post: SocialPost) => go(opensInQueue(post) ? { view: 'queue', post: post.id } : { post: post.id })
  const openId = (postId: string) => {
    const post = byId.get(postId)
    if (post) open(post)
  }
  const actionDeps = useActionDeps(social, openId)

  const chip = (evt: CalEvent, box?: CSSProperties) => {
    const post = evt.item.post_id ? byId.get(evt.item.post_id) : undefined
    if (!post || !evt.item.next_run_at) return null
    const drag = isMovable(post) ? social.dragProps(evt.item) : undefined
    const style = box ? { ...box, position: 'absolute' as const, overflow: 'hidden' } : undefined
    return <SocialsCalendarChip post={post} slot={evt.item.next_run_at} onOpen={() => open(post)} dragProps={drag} style={style} />
  }

  const title = layout === 'list' ? LIST_TITLE : titleFor(mode, anchor, week)
  return (
    <div className="cc-cal-root socials-calendar">
      <SocialsCalendarHeader
        title={title}
        layout={layout}
        channels={channels ?? []}
        filter={filter}
        onLayout={(next) => go({ cal: next, post: null })}
        onShift={(direction) => setAnchor((current) => shiftedAnchor(current, mode, direction))}
        onFilter={setFilter}
      />
      {layout === 'list' ? (
        <SocialsPostList role={role} focusPostId={route.post} />
      ) : (
        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
          <section aria-label={title} className="socials-calendar-grid overflow-x-auto">
            <div className="min-w-[840px]">
              {mode === 'month' ? (
                <MonthGrid cells={cells} events={events} anchorMonth={anchor.getMonth()} actionDeps={actionDeps} social={social} renderEvent={(evt) => chip(evt)} />
              ) : (
                <WeekGrid mode="week" days={week} events={events} actionDeps={actionDeps} social={social} empty={null} renderEvent={(evt, lane, lanes) => chip(evt, eventBox(evt, lane, lanes))} />
              )}
            </div>
          </section>
          <SocialsTodayRail posts={shown} onOpen={open} onReview={() => go({ view: 'queue', post: null })} />
        </div>
      )}
      {layout !== 'list' && <p className="text-[12.5px] leading-[1.45] text-muted-foreground">{DRAG_HINT}</p>}
      {social.dialog}
    </div>
  )
}
