'use client'

/**
 * PRD-251B US-B109 — what has happened to a saved post, under its editor: an approval an
 * edit reset, each channel's receipt (with Retry), the schedule and publish controls of an
 * approved post, and its history (PRD-251's own components).
 */
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { SocialsApprovalResetBanner, SocialsPostHistory } from '../socials-post-history'
import { SocialsPostReceipts } from '../socials-post-receipts'
import { SocialsPublishControls } from '../socials-publish-controls'
import { CARD } from './editor-ui'

export function SocialsEditorActivity({ post, role }: { post: SocialPost; role: Workspace['role'] }) {
  return (
    <section aria-label="Activity" className={CARD}>
      <h2 className="text-[15px] font-semibold text-foreground">Activity</h2>
      <SocialsApprovalResetBanner post={post} />
      <SocialsPostReceipts post={post} role={role} />
      <SocialsPublishControls post={post} role={role} />
      <SocialsPostHistory post={post} />
    </section>
  )
}
