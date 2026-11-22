'use client'

/**
 * PRD-251D (E3, US-D101) — approve where you are: the editor shows the Queue's own approve
 * controls (socials-post-review.tsx: the one Approve, labelled for what it does, "Approve · posts
 * HH:MM" or "Approve & post now to X", Request changes, which sends the post back to Auto as the
 * Queue does, Reject, and the unsourced-claims confirmation) for a post that waits for approval,
 * to a person who may approve (the Queue's role rule, `postActions(...).review`). Approval binds
 * to the stored content (D6), so while the
 * editor holds unsaved edits Approve is off and the card says to save first. A post whose file was
 * made before its last change to what a render reads (its `file_current`, US-D108) shows the
 * Queue's own sentence in place of a working Approve, with Render again (socials-post-review.tsx).
 */
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSendBackToAuto } from '@/hooks/use-socials-queue'
import { SocialsPostReview, type SendBack } from '../socials-post-review'
import { postActions } from '../socials-status'
import { SEND_BACK_HINT } from './queue-pane'
import { EditorCard, Hint } from './editor-ui'

export const APPROVE_CARD = 'Approve'
export const SAVE_BEFORE_APPROVING = 'You have unsaved edits. Save draft first: Approve takes the post exactly as it is saved.'
// The Queue's own wording for sending a post back to Auto (studio/queue-pane.tsx).
export const SEND_BACK_LABEL = 'What should change?'
export const SEND_BACK_SUBMIT = 'Send back to Auto'

interface EditorApproveCardProps {
  post: SocialPost
  role: Workspace['role']
  /** The editor holds edits the server does not have yet. */
  dirty: boolean
}

/** Request changes as the Queue sends it: back to Auto with what should change. */
function useSendBack(postId: string): SendBack {
  const sendBack = useSendBackToAuto()
  return {
    label: SEND_BACK_LABEL,
    submit: SEND_BACK_SUBMIT,
    hint: SEND_BACK_HINT,
    busy: sendBack.isLoading,
    onSend: (comment) => sendBack.mutate({ postId, comment }),
  }
}

export function EditorApproveCard({ post, role, dirty }: EditorApproveCardProps) {
  const sendBack = useSendBack(post.id)
  if (!postActions(role, post.status).review) return null
  return (
    <EditorCard label={APPROVE_CARD}>
      {dirty && <Hint>{SAVE_BEFORE_APPROVING}</Hint>}
      <SocialsPostReview post={post} dirty={dirty} sendBack={sendBack} />
    </EditorCard>
  )
}
