'use client'

/**
 * PRD-251 S0.5 — a post's history (its review_log), newest first. US-206: when
 * the newest entry voided an approval, a banner says the approval was reset.
 */
import { formatDistanceToNow } from 'date-fns'

import type { SocialPost } from '@/lib/api-client'
import { REVIEW_ACTION_LABELS } from './socials-status'
import { approvalWasVoided } from './socials-review'

function timeAgo(iso: string | null | undefined): string {
  if (!iso) return ''
  try {
    return formatDistanceToNow(new Date(iso), { addSuffix: true })
  } catch {
    return ''
  }
}

/** Shown while the newest history entry is an approval an edit voided (D6). */
export function SocialsApprovalResetBanner({ post }: { post: SocialPost }) {
  if (!approvalWasVoided(post)) return null
  return (
    <p role="status" className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-foreground">
      Approval reset: content changed
    </p>
  )
}

export function SocialsPostHistory({ post }: { post: SocialPost }) {
  const history = [...(post.review_log ?? [])].reverse()
  if (history.length === 0) return null
  return (
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
  )
}
