'use client'

/**
 * PRD-251D US-D203 — the lines that say where a post's render stands, moved here from the older
 * detail pane (socials-post-detail.tsx, deleted) so every status and every role still sees them:
 *
 * - P256-FIX-SOC-5 (F437): a rendering post says when its render started and when it gives up
 *   (the server's `render`), and a plan's draft that waits for you says what it waits for (SOC-2's
 *   `needs_input`);
 * - S1.1c: a failed post says why its last render failed.
 * - An approved post with no time says nothing will post until a time is picked or it is posted
 *   now (the server schedules an approval only into a slot still ahead), and, to a role that
 *   schedules, that Schedule and Publish now are below (socials-publish-controls.tsx). The editor
 *   shows the line at its top instead, where its Approve card was (`EditorNoTimeNotice`), since its
 *   Activity card is the last on the page: it passes `hideNoTime`, so the page says it once.
 *
 * And the older pane's Render (S1.1c), the full render (POST /render), where the editor has none:
 * for a post with a template that failed or waits for approval, to a role that authors. The
 * editor's Submit for approval makes the full render of a draft or a post sent back (PRD-251D
 * US-D102), but it is not offered in these two statuses, and its Render preview of a video is the
 * half-resolution preview. This render takes the post as it is saved.
 *
 * A post's Activity card shows them (socials-editor-activity.tsx), under the editor and in the
 * post's read-only view.
 */
import { format } from 'date-fns'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost, SocialPostStatus } from '@/lib/api-client'
import { useRenderSocialPost } from '@/hooks/use-socials-api'
import { RENDER_AGAIN } from '../socials-post-review'
import { canAuthorPosts, postActions } from '../socials-status'
import { HINT, Hint } from './editor-ui'
import { approvedWithNoTime, inputAsked, waitsForInput } from './queue-model'

/** What a rendering post says when the server gives no start time. */
export const RENDERING_NOTE = 'Rendering. This can take a few minutes; the post moves to Needs approval when it is done.'
/** The start of a failed post's line; the reason the render gave follows it. */
export const RENDER_FAILED = 'The last render failed:'
export const RENDER = 'Render'
export const RENDERS_AS_SAVED = 'It renders the post as it is saved: save your edits first.'
/** An approved post with no time: nothing posts it until a person acts. */
export const APPROVED_NO_TIME_NOTE = 'Approved, but it has no time yet: nothing will post until a time is picked or it is posted now.'
/** After it, in the Activity card, for a role that schedules: the publish controls follow in the same card. */
export const CONTROLS_BELOW = 'Schedule and Publish now are below.'
/** After it, at the top of the editor: the Activity card is the last on the page. */
export const CONTROLS_IN_ACTIVITY = 'Schedule and Publish now are in Activity, further down this page.'
const NOTICE = 'm-0 rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm text-foreground'
const CLOCK = 'HH:mm'
/** The review_log action of a render that failed (modules/socials/service.py ACTION_RENDER_FAILED). */
const RENDER_FAILED_ACTION = 'render_failed'
/** The statuses a full render starts from (socials-status.ts RENDERABLE) that the editor's Submit does not cover. */
const RENDER_HERE: ReadonlySet<SocialPostStatus> = new Set<SocialPostStatus>(['failed', 'needs_approval'])

/** P256-FIX-SOC-5: "Rendering. Started 16:22; it gives up at 16:47 if it has not finished.", in local time. */
export function renderingLine(post: Pick<SocialPost, 'render'>): string {
  const started = post.render?.started_at ? new Date(post.render.started_at) : null
  const givesUp = post.render?.gives_up_at ? new Date(post.render.gives_up_at) : null
  if (!started || !givesUp || Number.isNaN(started.getTime()) || Number.isNaN(givesUp.getTime())) return RENDERING_NOTE
  return `Rendering. Started ${format(started, CLOCK)}; it gives up at ${format(givesUp, CLOCK)} if it has not finished.`
}

/** P256-FIX-SOC-5: what a plan's draft that waits for you waits for (SOC-2's `needs_input`), else null. */
export function waitingLine(post: Pick<SocialPost, 'status' | 'review_log'>): string | null {
  return waitsForInput(post) ? `Waiting for you: ${inputAsked(post)}` : null
}

/** An approved post with no time: that nothing will post, and, to a role that schedules, where the controls
 * are (the publish controls show only to such a role); else null. */
export function noTimeLine(post: Pick<SocialPost, 'status' | 'planned_for' | 'scheduled_for'>, role: Workspace['role']): string | null {
  if (!approvedWithNoTime(post)) return null
  return canAuthorPosts(role) ? `${APPROVED_NO_TIME_NOTE} ${CONTROLS_BELOW}` : APPROVED_NO_TIME_NOTE
}

/** The editor's line for an approved post with no time, where its Approve card was: the person lands there
 * after Approve, and the Activity card, with Schedule and Publish now, is the last on the page. */
export function EditorNoTimeNotice({ post }: { post: SocialPost }) {
  if (!approvedWithNoTime(post)) return null
  return <p className={NOTICE}>{`${APPROVED_NO_TIME_NOTE} ${CONTROLS_IN_ACTIVITY}`}</p>
}

/** S1.1c: "The last render failed: <the reason the render gave>", for a failed post; else null. */
export function renderFailedLine(post: Pick<SocialPost, 'status' | 'review_log'>): string | null {
  if (post.status !== 'failed') return null
  const entry = [...(post.review_log ?? [])].reverse().find((e) => e.action === RENDER_FAILED_ACTION)
  return entry?.comment ? `${RENDER_FAILED} ${entry.comment}` : null
}

/** Whether `role` gets the full render here: a post with a template that failed or waits for approval.
 * Not while its file is older than its words (US-D108): the approve controls then say so and hold the
 * one Render again, so the page never shows two. */
export function rendersHere(role: Workspace['role'], post: Pick<SocialPost, 'status' | 'template_id' | 'file_current'>): boolean {
  if (post.status === 'needs_approval' && post.file_current === false) return false
  return RENDER_HERE.has(post.status) && postActions(role, post.status, !!post.template_id).render
}

/** `dirty`: the editor above holds unsaved edits. The render takes the post as it is saved, so it
 * waits for the save, as the approve controls' Render again does. */
function RenderButton({ post, role, dirty }: { post: SocialPost; role: Workspace['role']; dirty: boolean }) {
  const startRender = useRenderSocialPost()
  if (!rendersHere(role, post)) return null
  const rendered = Object.keys(post.media ?? {}).length > 0 || post.status === 'failed'
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button size="sm" variant="outline" onClick={() => startRender.mutate({ postId: post.id })} disabled={dirty || startRender.isLoading}>
        {rendered ? RENDER_AGAIN : RENDER}
      </Button>
      <Hint>{RENDERS_AS_SAVED}</Hint>
    </div>
  )
}

interface SocialsPostNoticesProps {
  post: SocialPost
  role: Workspace['role']
  dirty?: boolean
  /** The page already says the post has no time (the editor's `EditorNoTimeNotice`): the line is left out here. */
  hideNoTime?: boolean
}

export function SocialsPostNotices({ post, role, dirty = false, hideNoTime = false }: SocialsPostNoticesProps) {
  const waiting = waitingLine(post)
  const failed = renderFailedLine(post)
  const noTime = hideNoTime ? null : noTimeLine(post, role)
  return (
    <>
      {post.status === 'rendering' && <p role="status" className={cn(HINT, 'm-0')}>{renderingLine(post)}</p>}
      {waiting && <p role="alert" className={NOTICE}>{waiting}</p>}
      {noTime && <p role="status" className={NOTICE}>{noTime}</p>}
      {failed && (
        <p role="alert" className="m-0 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive">{failed}</p>
      )}
      <RenderButton post={post} role={role} dirty={dirty} />
    </>
  )
}
