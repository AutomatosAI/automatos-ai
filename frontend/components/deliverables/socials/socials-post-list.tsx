'use client'

/**
 * PRD-251 S0.5 — the Socials list: posts grouped by status with counts, newest
 * first, with an empty state; "New post" (the composer, US-207) and "Blank draft";
 * and the selected post's detail with
 * the actions the caller's role allows. S1.1c adds this month's render minutes
 * under the heading; S1.3 opens the brand kit (D5) from here for a role that
 * edits it.
 *
 * PRD-251B US-B107: this is the Studio's calendar view until US-B108 (then its
 * List); New post, the brand kit and the campaigns (Plans) moved to the Studio's
 * header (studio/studio-shell.tsx).
 *
 * S2.1 (US-205): a Board beside the List (the List | Board toggle). Both views
 * read the one posts query, so they show the same posts and the same counts,
 * and the board is read-only: a post moves only through its detail's actions.
 * Below 1024 px (useIsTabletOrBelow), and from the board, a post's detail is a
 * full-width view with a back control; on a wide screen the list keeps it
 * beside. `focusPostId` (the page's `?post=`, where global search links) opens
 * that post.
 */
import { useEffect, useMemo, useState } from 'react'
import { ArrowLeft, FilePlus, Loader2, Share2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { SocialsBoard } from './socials-board'
import { SocialsNewDraft } from './socials-new-draft'
import { SocialsPostDetail } from './socials-post-detail'
import { SocialsRenderMinutes } from './socials-render-minutes'
import { SocialsStatusList } from './socials-status-list'
import { SocialsViewToggle, type SocialsView } from './socials-view-toggle'
import { anyRendering, canAuthorPosts, groupPostsByStatus, type StatusGroup } from './socials-status'

interface SocialsPostsBodyProps {
  posts: SocialPost[]
  groups: StatusGroup[]
  role: Workspace['role']
  view: SocialsView
  selectedId: string | null
  onSelect: (postId: string | null) => void
}

/** The list (with the selected post beside it on a wide screen) or the board; a
 * post opened below 1024 px, or from the board, takes their place until "back". */
function SocialsPostsBody({ posts, groups, role, view, selectedId, onSelect }: SocialsPostsBodyProps) {
  const compact = useIsTabletOrBelow()
  const selected = posts.find((post) => post.id === selectedId) ?? null

  if (selected && (compact || view === 'board')) {
    return (
      <div className="space-y-3">
        <Button type="button" size="sm" variant="ghost" className="socials-back -ml-2" onClick={() => onSelect(null)}>
          <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden />
          {view === 'board' ? 'Back to board' : 'Back to posts'}
        </Button>
        <SocialsPostDetail post={selected} role={role} />
      </div>
    )
  }
  if (view === 'board') return <SocialsBoard posts={posts} selectedId={selectedId} onSelect={onSelect} />

  const list = <SocialsStatusList groups={groups} selectedId={selectedId} onSelect={onSelect} />
  if (compact) return list
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
      {list}
      <div>
        {selected ? (
          <SocialsPostDetail post={selected} role={role} />
        ) : (
          <p className="text-sm text-muted-foreground">Select a post to see its status and actions.</p>
        )}
      </div>
    </div>
  )
}

interface SocialsPostListProps {
  role: Workspace['role']
  /** A post to open: the page's `?post=`, which global search links to. */
  focusPostId?: string | null
}

export function SocialsPostList({ role, focusPostId = null }: SocialsPostListProps) {
  const { data, isLoading, isError, error } = useSocialPosts()
  // "Blank draft": the bare form. "New post" (the composer) is in the Studio's header.
  const [creating, setCreating] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(focusPostId)
  const [view, setView] = useState<SocialsView>('list')

  // A new ?post= (another search result) opens that post.
  useEffect(() => {
    if (focusPostId) setSelectedId(focusPostId)
  }, [focusPostId])

  const posts = useMemo(() => data?.posts ?? [], [data])
  const groups = useMemo(() => groupPostsByStatus(posts), [posts])
  const canAuthor = canAuthorPosts(role)

  const handleDraftDone = (post: SocialPost | null) => {
    setCreating(false)
    if (post) setSelectedId(post.id)
  }

  return (
    <div className="socials-tab space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Posts</h2>
          <p className="text-sm text-muted-foreground">
            Every post is approved one by one. An edit after approval sends it back for approval.
          </p>
          <SocialsRenderMinutes rendering={anyRendering(posts)} />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <SocialsViewToggle value={view} onChange={setView} />
          {canAuthor && !creating && (
            <Button size="sm" variant="outline" onClick={() => setCreating(true)}>
              <FilePlus className="mr-1.5 h-4 w-4" aria-hidden />
              Blank draft
            </Button>
          )}
        </div>
      </div>

      {creating && <SocialsNewDraft onDone={handleDraftDone} />}

      {isLoading ? (
        <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Loading posts…
        </div>
      ) : isError ? (
        <p className="text-sm text-destructive" role="alert">
          Could not load posts: {error instanceof Error ? error.message : 'unknown error'}
        </p>
      ) : groups.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/60 bg-card/20 px-6 py-10 text-center">
          <Share2 className="h-7 w-7 text-muted-foreground" aria-hidden />
          <p className="text-sm font-medium text-foreground">No posts yet</p>
          <p className="text-sm text-muted-foreground">
            {canAuthor ? 'Start one with New post.' : 'Posts your team drafts will show here.'}
          </p>
        </div>
      ) : (
        <SocialsPostsBody
          posts={posts}
          groups={groups}
          role={role}
          view={view}
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
      )}
    </div>
  )
}
