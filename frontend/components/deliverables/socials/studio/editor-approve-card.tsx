'use client'

/**
 * PRD-251D (E3, US-D101) — approve where you are: the editor shows the Queue's own approve
 * controls (socials-post-review.tsx: "Approve · publishes HH:MM", Approve & post now, Request
 * changes, which sends the post back to Auto as the Queue does, Reject, and the unsourced-claims
 * confirmation) for a post that waits for approval, to a person who may approve (the Queue's
 * role rule, `postActions(...).review`). Approval binds to the stored content (D6), so while the
 * editor holds unsaved edits Approve is off and the card says to save first. After a saved
 * change to what the file shows, the file is older than the words: Approve stays off and the
 * card says to make the file again (editor-submit.ts `saveOutdatesFile`).
 */
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSendBackToAuto } from '@/hooks/use-socials-queue'
import { SocialsPostReview, type SendBack } from '../socials-post-review'
import { postActions } from '../socials-status'
import { approveLabel } from './queue-model'
import { SEND_BACK_HINT } from './queue-pane'
import { FILE_OLDER_THAN_WORDS } from './editor-submit'
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
  /** A change saved in this sitting left the post's file older than its words. */
  fileOutdated?: boolean
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

export function EditorApproveCard({ post, role, dirty, fileOutdated = false }: EditorApproveCardProps) {
  const sendBack = useSendBack(post.id)
  if (!postActions(role, post.status).review) return null
  return (
    <EditorCard label={APPROVE_CARD}>
      {dirty && <Hint>{SAVE_BEFORE_APPROVING}</Hint>}
      {fileOutdated && <Hint>{FILE_OLDER_THAN_WORDS}</Hint>}
      <SocialsPostReview post={post} dirty={dirty || fileOutdated} approveLabel={approveLabel(post)} sendBack={sendBack} />
    </EditorCard>
  )
}
