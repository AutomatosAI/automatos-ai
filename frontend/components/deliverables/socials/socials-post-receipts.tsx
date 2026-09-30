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
 * privacy level and AI label publish will use (US-304).
 */
import { Button } from '@/components/ui/button'
import { badgeVariants } from '@/components/ui/badge'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost, SocialPostTarget } from '@/lib/api-client'
import { useRetrySocialPost } from '@/hooks/use-socials-publish-api'
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

/** Whether a render recorded AI-made footage for the post (D12): TikTok's AI label. */
function hasGeneratedFootage(post: SocialPost): boolean {
  return Object.values(post.footage ?? {}).some((slot) => (slot as { status?: string } | null)?.status === 'done')
}

/** What TikTok's publish will use (US-304): the chosen privacy, or the most private allowed. */
export function tiktokSettings(post: SocialPost, target: SocialPostTarget): string {
  const privacy = typeof target.options.privacy_level === 'string' ? target.options.privacy_level : MOST_PRIVATE
  const aigc = hasGeneratedFootage(post) || target.options.is_aigc === true
  return `Privacy: ${privacy}. AI-generated label: ${aigc ? 'on' : 'off'}.`
}

function Receipt({ post, target }: { post: SocialPost; target: SocialPostTarget }) {
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
        <p className="text-xs text-muted-foreground">{tiktokSettings(post, target)}</p>
      )}
    </li>
  )
}

export function SocialsPostReceipts({ post, role }: { post: SocialPost; role: Workspace['role'] }) {
  const retry = useRetrySocialPost()
  const targets = post.targets ?? []
  if (targets.length === 0) return null
  const canRetry = canAuthorPosts(role) && RETRYABLE.has(post.status) && targets.some((t) => t.status === 'failed')
  return (
    <section aria-label="Publish status" className="space-y-2">
      <p className="text-sm font-medium text-foreground">Publish status</p>
      <ul className="space-y-2">
        {targets.map((target) => (
          <Receipt key={target.id} post={post} target={target} />
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
