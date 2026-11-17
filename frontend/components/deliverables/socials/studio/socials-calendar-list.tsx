'use client'

/**
 * PRD-251D US-D202 — the calendar's List: every post, grouped by status in the order that puts
 * what needs a person first (socials-status.ts `groupPostsByStatus`), each group with its count
 * and its posts newest first. It is the one view that lists an archived post and a post with no
 * slot, which the Month and Week grids leave out. A row says where and when the post goes, its
 * title and its format; a click opens the post as a calendar chip does (the post page, or the
 * Queue for a post that waits for approval). It replaces the older post list, with its Board,
 * its List | Board toggle and its second detail pane.
 *
 * The posts are the ones the calendar was given (the Studio's one posts query); the list reads
 * the same query only for whether it is still loading or failed. S1.1c: this month's render
 * minutes show under the hint, their one place in the Studio. Nothing here drags or moves a
 * post: a post changes only through the actions on its own page.
 */
import { useMemo } from 'react'
import { formatDistanceToNow } from 'date-fns'
import { Loader2, Share2 } from 'lucide-react'

import { cn } from '@/lib/utils'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { slotDateLabel } from '@/lib/social-time'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { SocialsRenderMinutes } from '../socials-render-minutes'
import { anyRendering, canAuthorPosts, groupPostsByStatus, type StatusGroup } from '../socials-status'
import { CHANNEL_BADGE, CHIP_KEY } from './socials-calendar-chip'
import { formatLabel, postBadges, slotOf } from './socials-calendar-model'
import { Hint } from './editor-ui'
import { NO_SLOT } from './queue-model'

export const LIST_HINT = 'Every post is approved one by one. An edit after approval sends it back for approval.'
export const NO_POSTS = 'No posts yet'
export const START_ONE = 'Start one with New post.'
export const TEAM_DRAFTS = 'Posts your team drafts will show here.'
export const LOADING_POSTS = 'Loading posts…'
export const LOAD_FAILED = 'Could not load posts:'
const UNKNOWN_ERROR = 'unknown error'
const GROUP_LABEL = 'm-0 text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground'

interface SocialsCalendarListProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  /** Opens a post as a calendar chip does. */
  onOpen: (post: SocialPost) => void
}

/** "Mon 9 Nov 2026, 18:00 GMT+9" in the post's own timezone, or "No slot". */
function whenLabel(post: SocialPost): string {
  const slot = slotOf(post)
  return slot ? slotDateLabel(slot, post.timezone) : NO_SLOT
}

/** "Updated 3 days ago"; nothing for a time the browser cannot read. */
function updatedLabel(post: SocialPost): string | null {
  const at = new Date(post.updated_at || post.created_at)
  return Number.isNaN(at.getTime()) ? null : `Updated ${formatDistanceToNow(at, { addSuffix: true })}`
}

function ListRow({ post, onOpen }: { post: SocialPost; onOpen: () => void }) {
  const badges = postBadges(post)
  const when = whenLabel(post)
  // Read aloud as one sentence, the title first, as a calendar chip is.
  const label = [post.title, when, formatLabel(post), ...badges].join(', ')
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={label}
      title={label}
      className="flex w-full min-w-0 flex-col gap-1.5 rounded-xl border border-border bg-card px-3.5 py-3 text-left transition-colors hover:border-muted-foreground"
    >
      <span className="flex min-w-0 items-center gap-1.5">
        {badges.map((badge) => (
          <span key={badge} className={CHANNEL_BADGE}>{badge}</span>
        ))}
        <span className={cn(CHIP_KEY, 'truncate')}>{when}</span>
      </span>
      <span className="break-words text-[15px] font-semibold text-foreground">{post.title}</span>
      <span className="text-[12.5px] text-muted-foreground">{[formatLabel(post), updatedLabel(post)].filter(Boolean).join(' · ')}</span>
    </button>
  )
}

function ListGroup({ group, onOpen }: { group: StatusGroup; onOpen: (post: SocialPost) => void }) {
  return (
    <section aria-label={`${group.label} posts`} className="flex flex-col gap-2.5">
      <h2 className={GROUP_LABEL}>
        {group.label}{' '}
        <span className="font-mono" data-testid="socials-status-count">{group.posts.length}</span>
      </h2>
      <ul className="m-0 grid list-none gap-2.5 p-0 md:grid-cols-2 xl:grid-cols-3">
        {group.posts.map((post) => (
          <li key={post.id} className="min-w-0">
            <ListRow post={post} onOpen={() => onOpen(post)} />
          </li>
        ))}
      </ul>
    </section>
  )
}

function EmptyList({ canAuthor }: { canAuthor: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border px-6 py-10 text-center">
      <Share2 className="h-7 w-7 text-muted-foreground" aria-hidden />
      <p className="m-0 text-sm font-medium text-foreground">{NO_POSTS}</p>
      <p className="m-0 text-sm text-muted-foreground">{canAuthor ? START_ONE : TEAM_DRAFTS}</p>
    </div>
  )
}

function ListBody({ role, posts, onOpen }: SocialsCalendarListProps) {
  const { isLoading, isError, error } = useSocialPosts()
  const groups = useMemo(() => groupPostsByStatus(posts), [posts])
  if (isLoading) {
    return (
      <p className="m-0 flex items-center gap-2 py-6 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
        {LOADING_POSTS}
      </p>
    )
  }
  if (isError) {
    return (
      <p role="alert" className="m-0 text-sm text-destructive">
        {LOAD_FAILED} {error instanceof Error ? error.message : UNKNOWN_ERROR}
      </p>
    )
  }
  if (groups.length === 0) return <EmptyList canAuthor={canAuthorPosts(role)} />
  return (
    <div className="flex flex-col gap-6">
      {groups.map((group) => (
        <ListGroup key={group.status} group={group} onOpen={onOpen} />
      ))}
    </div>
  )
}

export function SocialsCalendarList({ role, posts, onOpen }: SocialsCalendarListProps) {
  return (
    <div className="socials-calendar-list flex flex-col gap-5">
      <div className="flex flex-col gap-1">
        <Hint>{LIST_HINT}</Hint>
        <SocialsRenderMinutes rendering={anyRendering(posts)} />
      </div>
      <ListBody role={role} posts={posts} onOpen={onOpen} />
    </div>
  )
}
