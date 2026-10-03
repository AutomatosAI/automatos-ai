'use client'

/**
 * 3 Oct 2026 — Delete a post: an owner or admin, for a post that has not gone out on a
 * channel (api/socials_delete.py). It asks first, since it cannot be undone. Used by the
 * editor's header and the Queue's pane.
 */
import { useState } from 'react'
import { Trash2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useDeleteSocialPost } from '@/hooks/use-socials-api'
import { canDeletePost } from '../socials-status'

export const DELETE_CONFIRM = 'Delete this post? Its files go with it, and it cannot be undone.'
const DESTRUCTIVE_OUTLINE = 'text-destructive hover:bg-destructive/10 hover:text-destructive'

interface DeletePostButtonProps {
  post: SocialPost
  role: Workspace['role']
  onDeleted: () => void
}

export function DeletePostButton({ post, role, onDeleted }: DeletePostButtonProps) {
  const [asking, setAsking] = useState(false)
  const remove = useDeleteSocialPost()
  if (!canDeletePost(role, post)) return null
  if (!asking) {
    return (
      <Button type="button" variant="outline" size="sm" className={DESTRUCTIVE_OUTLINE} onClick={() => setAsking(true)}>
        <Trash2 className="mr-1.5 h-4 w-4" aria-hidden />
        Delete
      </Button>
    )
  }
  return (
    <div role="alertdialog" aria-label="Delete post" className="flex flex-wrap items-center gap-2 rounded-lg border border-destructive/40 px-3 py-2">
      <span className="text-sm text-foreground">{DELETE_CONFIRM}</span>
      <Button type="button" size="sm" variant="destructive" disabled={remove.isLoading}
        onClick={() => remove.mutate({ postId: post.id }, { onSuccess: onDeleted })}>
        Delete post
      </Button>
      <Button type="button" size="sm" variant="ghost" onClick={() => setAsking(false)}>Keep it</Button>
    </div>
  )
}
