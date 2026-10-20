'use client'

/**
 * PRD-251 S3.3f (US-308) — where a post goes, and what happened there.
 *
 * Each target (a channel and post kind) shows its status: waiting, publishing
 * while the list polls (use-socials-api renderPollInterval), published with its
 * permalink (opens the live post), or failed with the platform's own message; a
 * step the publish skipped (a thumbnail with no public storage) is noted. A failed
 * or partially published post offers Retry, which publishes only the failed
 * targets again (POST /retry). Before it publishes, a TikTok target shows the
 * privacy level publish will use (US-304).
 *
 * PRD-264 EUAI-05: every channel states its AI label decision (Art. 50): TikTok's
 * AI-generated label, YouTube Studio's setting, or the caption for a channel that takes
 * no AI flag. The words are the backend's (GET /ai-labels, useSocialPostAiLabels): the
 * line a publish used once one ran, else the line it will use.
 */
import { Button } from '@/components/ui/button'
import { badgeVariants } from '@/components/ui/badge'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost, SocialPostTarget } from '@/lib/api-client'
import { useRetrySocialPost } from '@/hooks/use-socials-publish-api'
import { useSocialPostAiLabels } from '@/hooks/use-socials-ai-labels'
import { canAuthorPosts, channelLabel } from './socials-status'

const TARGET_STATUS_LABELS: Record<SocialPostTarget['status'], string> = {
  pending: 'Waiting',
  uploading: 'Publishing…',
  published: 'Published',
  failed: 'Failed',
}
const RETRYABLE: ReadonlySet<SocialPost['status']> = new Set(['failed', 'partially_published'])
const BEFORE_PUBLISH: ReadonlySet<SocialPost['status']> = new Set(['approved', 'scheduled', 'missed'])
const MOST_PRIVATE = 'the most private level the account allows'

/**
 * The privacy TikTok's publish will use (US-304): the chosen level when the account
 * allows it (publish reads the creator's allowed levels), else the most private
 * level it allows. Never a public default.
 */
export function tiktokPrivacy(target: SocialPostTarget): string {
  const chosen = target.options.privacy_level
  const privacy = typeof chosen === 'string' ? `${chosen} if the account allows it, else ${MOST_PRIVATE}` : MOST_PRIVATE
  return `Privacy: ${privacy}.`
}

function Receipt({ post, target, aiLabel }: { post: SocialPost; target: SocialPostTarget; aiLabel?: string }) {
  return (
    <li className="space-y-1 rounded-lg border border-border/60 px-3 py-2" data-testid={`socials-receipt-${target.toolkit}-${target.post_kind}`}>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="font-medium text-foreground">
          {channelLabel(target.toolkit)} · {target.post_kind}
        </span>
        <span className={badgeVariants({ variant: target.status === 'failed' ? 'destructive' : 'secondary' })}>
          {TARGET_STATUS_LABELS[target.status]}
        </span>
        {target.permalink && (
          <a href={target.permalink} target="_blank" rel="noopener noreferrer" className="text-sm text-primary underline">
            View the post
          </a>
        )}
      </div>
      {target.error && <p className="text-xs text-destructive">{target.error}</p>}
      {(target.notes ?? []).map((note) => (
        <p key={note} className="text-xs text-muted-foreground">
          {note}
        </p>
      ))}
      {target.toolkit === 'tiktok' && BEFORE_PUBLISH.has(post.status) && (
        <p className="text-xs text-muted-foreground">{tiktokPrivacy(target)}</p>
      )}
      {aiLabel && (
        <p className="text-xs text-muted-foreground" data-testid={`socials-ai-label-${target.toolkit}-${target.post_kind}`}>
          {aiLabel}
        </p>
      )}
    </li>
  )
}

export function SocialsPostReceipts({ post, role }: { post: SocialPost; role: Workspace['role'] }) {
  const retry = useRetrySocialPost()
  const aiLabels = useSocialPostAiLabels(post)
  const targets = post.targets ?? []
  const labelOf = new Map((aiLabels.data?.targets ?? []).map((t) => [t.id, t.ai_label] as const))
  if (targets.length === 0) return null
  const canRetry = canAuthorPosts(role) && RETRYABLE.has(post.status) && targets.some((t) => t.status === 'failed')
  return (
    <section aria-label="Publish status" className="space-y-2">
      <p className="text-sm font-medium text-foreground">Publish status</p>
      <ul className="space-y-2">
        {targets.map((target) => (
          <Receipt key={target.id} post={post} target={target} aiLabel={labelOf.get(target.id)} />
        ))}
      </ul>
      {canRetry && (
        <Button size="sm" variant="outline" onClick={() => retry.mutate({ postId: post.id })} disabled={retry.isLoading}>
          Retry the failed channels
        </Button>
      )}
    </section>
  )
}
