'use client'

/**
 * PRD-251 S0.5 — one post: its status, its copy, and the actions the caller's
 * role allows in that status (S0.3). Saving a copy change on an approved or
 * scheduled post voids the approval (D6) — the server moves it back to Needs
 * approval, and a banner says the approval was reset (US-206).
 * S1.1c: a post with a template renders; the render ends it in Needs approval,
 * or in Failed with the reason shown here and in the history.
 * S1.5: a video post's voice is chosen here (SocialsVoicePicker).
 * S2.3 (US-206): the approval view — each channel's copy, the exact rendered
 * media, every claim's source (SocialsPostEvidence) — and the approver's
 * actions (SocialsPostReview).
 * Wave 3 (US-308): scheduling and publishing (SocialsPublishControls) and each
 * channel's receipt, with Retry for the ones that failed (SocialsPostReceipts).
 */
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { badgeVariants } from '@/components/ui/badge'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useRenderSocialPost, useSocialPostAction, useUpdateSocialPost } from '@/hooks/use-socials-api'
import { SOCIAL_STATUS_LABELS, canAuthorPosts, postActions, speaksAScript } from './socials-status'
import { SocialsVoicePicker } from './socials-voice-picker'
import { SocialsPostEvidence } from './socials-post-evidence'
import { SocialsApprovalResetBanner, SocialsPostHistory } from './socials-post-history'
import { SocialsPostReview } from './socials-post-review'
import { SocialsPostReceipts } from './socials-post-receipts'
import { SocialsPublishControls } from './socials-publish-controls'

interface SocialsPostDetailProps {
  post: SocialPost
  role: Workspace['role']
}

/** The reason the last render gave for failing, if the post is failed. */
function lastRenderFailure(post: SocialPost): string | null {
  if (post.status !== 'failed') return null
  const entry = [...(post.review_log ?? [])].reverse().find((e) => e.action === 'render_failed')
  return entry?.comment ?? null
}

function RenderNotices({ post }: { post: SocialPost }) {
  const renderFailure = lastRenderFailure(post)
  return (
    <>
      {post.status === 'rendering' && (
        <p className="text-sm text-muted-foreground" role="status">
          Rendering. This can take a few minutes; the post moves to Needs approval when it is done.
        </p>
      )}
      {renderFailure && (
        <p className="rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive" role="alert">
          The last render failed: {renderFailure}
        </p>
      )}
    </>
  )
}

interface CopyFieldProps {
  post: SocialPost
  editable: boolean
  value: string
  onChange: (value: string) => void
}

function CopyField({ post, editable, value, onChange }: CopyFieldProps) {
  const approvalAtStake = post.status === 'approved' || post.status === 'scheduled' || post.status === 'missed'
  if (!editable) {
    return (
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">Copy</p>
        <p className="whitespace-pre-wrap text-sm text-foreground">{post.copy?.base || 'No copy yet.'}</p>
      </div>
    )
  }
  return (
    <div className="space-y-1.5">
      <Label htmlFor={`socials-copy-${post.id}`}>Copy</Label>
      <Textarea id={`socials-copy-${post.id}`} value={value} rows={5} onChange={(event) => onChange(event.target.value)} />
      {approvalAtStake && (
        <p className="text-xs text-muted-foreground">Saving a change sends this post back for approval.</p>
      )}
    </div>
  )
}

export function SocialsPostDetail({ post, role }: SocialsPostDetailProps) {
  const actions = postActions(role, post.status, !!post.template_id)
  const update = useUpdateSocialPost()
  const act = useSocialPostAction()
  const startRender = useRenderSocialPost()
  const savedBase = post.copy?.base ?? ''
  const [base, setBase] = useState(savedBase)

  // A refetch (after any action) carries the server's copy; follow it.
  useEffect(() => {
    setBase(savedBase)
  }, [post.id, savedBase])

  const dirty = base !== savedBase
  const busy = update.isLoading || act.isLoading || startRender.isLoading
  const rendered = Object.keys(post.media ?? {}).length > 0
  const saveCopy = () => update.mutate({ postId: post.id, changes: { copy: { ...post.copy, base } } })

  return (
    <article aria-label={`Post: ${post.title}`} className="space-y-4 rounded-xl border border-border bg-card/40 p-5">
      <header className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-base font-semibold text-foreground">{post.title}</h3>
          <span className={badgeVariants({ variant: 'secondary' })} data-testid="socials-post-status">
            {SOCIAL_STATUS_LABELS[post.status]}
          </span>
        </div>
        {post.brief && <p className="text-sm text-muted-foreground">{post.brief}</p>}
      </header>

      <SocialsApprovalResetBanner post={post} />
      <RenderNotices post={post} />
      <CopyField post={post} editable={actions.edit} value={base} onChange={setBase} />
      <SocialsPostEvidence post={post} />
      {speaksAScript(post) && <SocialsVoicePicker post={post} editable={actions.edit} />}

      <div className="flex flex-wrap gap-2">
        {actions.edit && (
          <Button size="sm" variant="outline" onClick={saveCopy} disabled={!dirty || busy}>
            Save copy
          </Button>
        )}
        {actions.render && (
          <Button size="sm" variant="outline" onClick={() => startRender.mutate({ postId: post.id })} disabled={busy || dirty}>
            {rendered || post.status === 'failed' ? 'Render again' : 'Render'}
          </Button>
        )}
        {actions.submit && (
          <Button size="sm" onClick={() => act.mutate({ postId: post.id, action: { kind: 'submit' } })} disabled={busy || dirty}>
            Submit for approval
          </Button>
        )}
      </div>
      {actions.review && <SocialsPostReview post={post} dirty={dirty} />}
      <SocialsPostReceipts post={post} role={role} />
      <SocialsPublishControls post={post} role={role} />

      {!canAuthorPosts(role) && (
        <p className="text-xs text-muted-foreground">Your role can read posts but not change them.</p>
      )}
      <SocialsPostHistory post={post} />
    </article>
  )
}
