'use client'

/**
 * PRD-251B US-B109 — what has happened to a saved post, under its editor: an approval an
 * edit reset, each channel's receipt (with Retry), the schedule and publish controls of an
 * approved post, and its history (PRD-251's own components).
 * PRD-251D US-D203: the post's read-only view (socials-post-view.tsx) shows this card too, and
 * it carries what the older detail pane had about rendering (post-notices.tsx): since when a
 * post renders, what a waiting draft waits for, why the last render failed, and the full render
 * of a failed or waiting post, which the editor's header does not offer.
 */
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsApprovalResetBanner, SocialsPostHistory } from '../socials-post-history'
import { SocialsPostReceipts } from '../socials-post-receipts'
import { SocialsPublishControls } from '../socials-publish-controls'
import { CARD } from './editor-ui'
import { SocialsPostNotices } from './post-notices'

interface SocialsEditorActivityProps {
  post: SocialPost
  role: Workspace['role']
  /** The editor above holds unsaved edits: the full render here waits for the save. */
  dirty?: boolean
}

export function SocialsEditorActivity({ post, role, dirty = false }: SocialsEditorActivityProps) {
  return (
    <section aria-label="Activity" className={CARD}>
      <h2 className="text-[15px] font-semibold text-foreground">Activity</h2>
      <SocialsApprovalResetBanner post={post} />
      <SocialsPostNotices post={post} role={role} dirty={dirty} />
      <SocialsPostReceipts post={post} role={role} />
      <SocialsPublishControls post={post} role={role} />
      <SocialsPostHistory post={post} />
    </section>
  )
}
