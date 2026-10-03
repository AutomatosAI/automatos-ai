'use client'

/**
 * 3 Oct 2026 — Delete a post: an owner or admin, for a post that has not gone out on a
 * channel (api/socials_delete.py). It asks first, since it cannot be undone. Used by the
 * editor's header and the Queue's pane.
 */
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useDeleteSocialPost } from '@/hooks/use-socials-api'
import { canDeletePost } from '../socials-status'
import { DeleteAskedFirst } from './delete-asked-first'

export const DELETE_CONFIRM = 'Delete this post? Its files go with it, and it cannot be undone.'

interface DeletePostButtonProps {
  post: SocialPost
  role: Workspace['role']
  onDeleted: () => void
}

export function DeletePostButton({ post, role, onDeleted }: DeletePostButtonProps) {
  const remove = useDeleteSocialPost()
  if (!canDeletePost(role, post)) return null
  return (
    <DeleteAskedFirst noun="post" question={DELETE_CONFIRM} busy={remove.isLoading}
      onConfirm={() => remove.mutate({ postId: post.id }, { onSuccess: onDeleted })} />
  )
}
