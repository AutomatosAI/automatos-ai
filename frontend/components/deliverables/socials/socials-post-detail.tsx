'use client'

/**
 * PRD-251 S0.5 — one post: its status, its copy, and the actions the caller's
 * role allows in that status (S0.3): submit, approve, request changes (with a
 * comment) and reject. Saving a copy change on an approved or scheduled post
 * voids the approval (D6) — the server moves it back to Needs approval. Approve
 * sends the content_hash of the version shown here, so it approves only that.
 * S1.1c: a post with a template renders; the render ends it in Needs approval,
 * or in Failed with the reason shown here and in the history.
 * S1.5: a video post's voice is chosen here: Kokoro, or a voice toolkit the
 * workspace has connected in Composio (SocialsVoicePicker).
 */
import { useEffect, useState, type FormEvent } from 'react'
import { formatDistanceToNow } from 'date-fns'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { badgeVariants } from '@/components/ui/badge'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import {
  useRenderSocialPost,
  useSocialPostAction,
  useUpdateSocialPost,
  type SocialPostAction,
} from '@/hooks/use-socials-api'
import {
  REVIEW_ACTION_LABELS,
  SOCIAL_COMMENT_MAX_CHARS,
  SOCIAL_STATUS_LABELS,
  canAuthorPosts,
  postActions,
  speaksAScript,
} from './socials-status'
import { SocialsVoicePicker } from './socials-voice-picker'

interface SocialsPostDetailProps {
  post: SocialPost
  role: Workspace['role']
}

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return ''
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

/** The reason the last render gave for failing, if the post is failed. */
function lastRenderFailure(post: SocialPost): string | null {
  if (post.status !== 'failed') return null
  const entry = [...(post.review_log ?? [])].reverse().find((e) => e.action === 'render_failed')
  return entry?.comment ?? null
}

export function SocialsPostDetail({ post, role }: SocialsPostDetailProps) {
  const actions = postActions(role, post.status, !!post.template_id)
  const update = useUpdateSocialPost()
  const act = useSocialPostAction()
  const startRender = useRenderSocialPost()
  const savedBase = post.copy?.base ?? ''
  const [base, setBase] = useState(savedBase)
  const [askingChanges, setAskingChanges] = useState(false)
  const [comment, setComment] = useState('')

  // A refetch (after any action) carries the server's copy; follow it.
  useEffect(() => {
    setBase(savedBase)
  }, [post.id, savedBase])

  const dirty = base !== savedBase
  const busy = update.isLoading || act.isLoading || startRender.isLoading
  const renderFailure = lastRenderFailure(post)
  const rendered = Object.keys(post.media ?? {}).length > 0
  const approvalAtStake = post.status === 'approved' || post.status === 'scheduled'

  const run = (action: SocialPostAction) =>
    act.mutate(
      { postId: post.id, action },
      {
        onSuccess: () => {
          setAskingChanges(false)
          setComment('')
        },
      },
    )

  const saveCopy = () => update.mutate({ postId: post.id, changes: { copy: { ...post.copy, base } } })

  const sendChangeRequest = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!comment.trim()) return
    run({ kind: 'request_changes', comment: comment.trim() })
  }

  const history = [...(post.review_log ?? [])].reverse()

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

      <div className="space-y-1.5">
        {actions.edit ? (
          <>
            <Label htmlFor={`socials-copy-${post.id}`}>Copy</Label>
            <Textarea
              id={`socials-copy-${post.id}`}
              value={base}
              rows={5}
              onChange={(event) => setBase(event.target.value)}
            />
          </>
        ) : (
          <>
            <p className="text-sm font-medium text-foreground">Copy</p>
            <p className="whitespace-pre-wrap text-sm text-foreground">{savedBase || 'No copy yet.'}</p>
          </>
        )}
        {actions.edit && approvalAtStake && (
          <p className="text-xs text-muted-foreground">Saving a change sends this post back for approval.</p>
        )}
      </div>

      {speaksAScript(post) && <SocialsVoicePicker post={post} editable={actions.edit} />}

      <div className="flex flex-wrap gap-2">
        {actions.edit && (
          <Button size="sm" variant="outline" onClick={saveCopy} disabled={!dirty || busy}>
            Save copy
          </Button>
        )}
        {actions.render && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => startRender.mutate({ postId: post.id })}
            disabled={busy || dirty}
          >
            {rendered || post.status === 'failed' ? 'Render again' : 'Render'}
          </Button>
        )}
        {actions.submit && (
          <Button size="sm" onClick={() => run({ kind: 'submit' })} disabled={busy || dirty}>
            Submit for approval
          </Button>
        )}
        {actions.review && (
          <>
            <Button
              size="sm"
              onClick={() => run({ kind: 'approve', contentHash: post.content_hash })}
              disabled={busy || dirty}
            >
              Approve
            </Button>
            <Button size="sm" variant="outline" onClick={() => setAskingChanges(true)} disabled={busy}>
              Request changes
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={() => run({ kind: 'reject' })}
              disabled={busy}
            >
              Reject
            </Button>
          </>
        )}
      </div>

      {askingChanges && actions.review && (
        <form onSubmit={sendChangeRequest} aria-label="Request changes" className="space-y-2">
          <Label htmlFor={`socials-changes-${post.id}`}>What needs to change?</Label>
          <Textarea
            id={`socials-changes-${post.id}`}
            value={comment}
            rows={3}
            maxLength={SOCIAL_COMMENT_MAX_CHARS}
            onChange={(event) => setComment(event.target.value)}
          />
          <div className="flex justify-end gap-2">
            <Button type="button" size="sm" variant="ghost" onClick={() => setAskingChanges(false)}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={!comment.trim() || busy}>
              Send request
            </Button>
          </div>
        </form>
      )}

      {!canAuthorPosts(role) && (
        <p className="text-xs text-muted-foreground">Your role can read posts but not change them.</p>
      )}

      {history.length > 0 && (
        <section aria-label="History" className="space-y-2 border-t border-border/60 pt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">History</h4>
          <ol className="space-y-1.5">
            {history.map((entry, index) => (
              <li key={`${entry.at}-${index}`} className="text-sm">
                <span className="text-foreground">{REVIEW_ACTION_LABELS[entry.action] ?? entry.action}</span>
                <span className="text-xs text-muted-foreground"> · {timeAgo(entry.at)}</span>
                {entry.comment && <p className="text-sm text-muted-foreground">{entry.comment}</p>}
              </li>
            ))}
          </ol>
        </section>
      )}
    </article>
  )
}
