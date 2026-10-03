'use client'

/**
 * PRD-251B US-B109 — what ?post= opens (global search, notifications and the calendar link
 * there): the editor for a post the caller drafts and can still change; the post's own
 * view (its status, receipts, history and the approver's actions) for one being made,
 * posted or archived, or for a role that only reads; New post opens an empty editor.
 */
import { ArrowLeft, Loader2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsPostDetail } from '../socials-post-detail'
import { canAuthorPosts } from '../socials-status'
import { SocialsEditor } from './socials-editor'
import { NEW_POST, type GoTo } from './studio-route'

/** The statuses a post can still be edited in (modules/socials/service.py EDITABLE_STATUSES). */
export const EDITOR_STATUSES: ReadonlySet<string> = new Set([
  'draft', 'needs_approval', 'changes_requested', 'approved', 'scheduled', 'missed', 'failed',
])
export const POST_NOT_FOUND = 'This post is not in this workspace any more.'

interface SocialsPostPageProps {
  role: Workspace['role']
  posts: ReadonlyArray<SocialPost>
  postId: string
  loading: boolean
  go: GoTo
}

function Back({ go }: { go: GoTo }) {
  return (
    <Button type="button" variant="ghost" size="sm" className="socials-back -ml-2 w-fit text-muted-foreground" onClick={() => go({ view: 'calendar', post: null })}>
      <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden />
      Back to calendar
    </Button>
  )
}

export function SocialsPostPage({ role, posts, postId, loading, go }: SocialsPostPageProps) {
  if (postId === NEW_POST) return <SocialsEditor role={role} post={null} go={go} />
  const post = posts.find((p) => p.id === postId)
  if (!post) {
    return (
      <div className="flex flex-col gap-3">
        <Back go={go} />
        {loading ? (
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" aria-label="Loading the post" />
        ) : (
          <p className="text-sm text-muted-foreground">{POST_NOT_FOUND}</p>
        )}
      </div>
    )
  }
  if (!canAuthorPosts(role) || !EDITOR_STATUSES.has(post.status)) {
    return (
      <div className="flex flex-col gap-3">
        <Back go={go} />
        <SocialsPostDetail post={post} role={role} />
      </div>
    )
  }
  return <SocialsEditor key={post.id} role={role} post={post} go={go} />
}
