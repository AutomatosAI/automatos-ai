'use client'

/**
 * PRD-251 S2.3 (US-206) — the approver's actions on a post in Needs approval.
 *
 * - Approve sends the content_hash of the version on screen (D6). A 409 says the
 *   post changed while it was reviewed and reloads it.
 * - A post with unsourced claims (D7) needs a second confirmation that names them,
 *   which approves with override_unsourced. The server may name claims whose
 *   source has since gone (a 422), which asks the same way.
 * - Request changes needs a comment; reject takes an optional reason.
 */
import { useEffect, useState, type FormEvent } from 'react'

import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import type { SocialPost } from '@/lib/api-client'
import {
  SOCIAL_POST_REVIEW_STALE_MESSAGE,
  useApproveSocialPost,
  useSocialPostAction,
  type SocialPostAction,
} from '@/hooks/use-socials-api'
import { SOCIAL_COMMENT_MAX_CHARS } from './socials-status'
import { unsourcedClaims } from './socials-review'

type Asking = 'changes' | 'reject' | null

interface CommentFormProps {
  postId: string
  label: string
  heading: string
  submit: string
  required: boolean
  busy: boolean
  onSend: (text: string) => void
  onCancel: () => void
}

function CommentForm({ postId, label, heading, submit, required, busy, onSend, onCancel }: CommentFormProps) {
  const [text, setText] = useState('')
  const id = `socials-${heading.replace(/\s+/g, '-').toLowerCase()}-${postId}`
  const send = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (required && !text.trim()) return
    onSend(text.trim())
  }
  return (
    <form onSubmit={send} aria-label={heading} className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Textarea id={id} value={text} rows={3} maxLength={SOCIAL_COMMENT_MAX_CHARS} onChange={(e) => setText(e.target.value)} />
      <div className="flex justify-end gap-2">
        <Button type="button" size="sm" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" size="sm" disabled={(required && !text.trim()) || busy}>
          {submit}
        </Button>
      </div>
    </form>
  )
}

function UnsourcedConfirmation({ claims, busy, onConfirm, onCancel }: {
  claims: string[]
  busy: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div role="alertdialog" aria-label="Approve with unsourced claims" className="space-y-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2">
      <p className="text-sm text-destructive">
        These claims have no source: {claims.join(', ')}. Approve them anyway?
      </p>
      <div className="flex justify-end gap-2">
        <Button type="button" size="sm" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="button" size="sm" variant="destructive" onClick={onConfirm} disabled={busy}>
          Approve with unsourced claims
        </Button>
      </div>
    </div>
  )
}

interface SocialsPostReviewProps {
  post: SocialPost
  /** Unsaved edits on screen: approving them would approve what the server does not have. */
  dirty: boolean
}

export function SocialsPostReview({ post, dirty }: SocialsPostReviewProps) {
  const [asking, setAsking] = useState<Asking>(null)
  const [confirmClaims, setConfirmClaims] = useState<string[] | null>(null)
  const [stale, setStale] = useState(false)
  const act = useSocialPostAction()
  const approve = useApproveSocialPost({ onStale: () => setStale(true), onUnsourced: setConfirmClaims })
  const busy = act.isLoading || approve.isLoading

  // A new version on screen is the one to review: a confirmation for the old one goes.
  useEffect(() => {
    setConfirmClaims(null)
  }, [post.content_hash])

  const run = (action: SocialPostAction) => act.mutate({ postId: post.id, action }, { onSuccess: () => setAsking(null) })
  const approveShown = (overrideUnsourced = false) => {
    setStale(false)
    approve.mutate({ postId: post.id, contentHash: post.content_hash, overrideUnsourced })
  }
  const startApprove = () => {
    const claims = unsourcedClaims(post)
    if (claims.length > 0) setConfirmClaims(claims)
    else approveShown()
  }

  return (
    <div className="space-y-3">
      {stale && (
        <p role="alert" className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-foreground">
          {SOCIAL_POST_REVIEW_STALE_MESSAGE}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={startApprove} disabled={busy || dirty || confirmClaims !== null}>
          Approve
        </Button>
        <Button size="sm" variant="outline" onClick={() => setAsking('changes')} disabled={busy}>
          Request changes
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="text-destructive hover:bg-destructive/10 hover:text-destructive"
          onClick={() => setAsking('reject')}
          disabled={busy}
        >
          Reject
        </Button>
      </div>
      {confirmClaims && (
        <UnsourcedConfirmation
          claims={confirmClaims}
          busy={busy}
          onConfirm={() => approveShown(true)}
          onCancel={() => setConfirmClaims(null)}
        />
      )}
      {asking === 'changes' && (
        <CommentForm
          postId={post.id} heading="Request changes" label="What needs to change?" submit="Send request" required busy={busy}
          onSend={(comment) => run({ kind: 'request_changes', comment })} onCancel={() => setAsking(null)}
        />
      )}
      {asking === 'reject' && (
        <CommentForm
          postId={post.id} heading="Reject post" label="Reason (optional)" submit="Reject post" required={false} busy={busy}
          onSend={(reason) => run({ kind: 'reject', reason: reason || undefined })} onCancel={() => setAsking(null)}
        />
      )}
    </div>
  )
}
