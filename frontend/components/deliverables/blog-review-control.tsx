'use client'

/**
 * PRD-264 EUAI-08b: an agent-written post is labelled AI-generated wherever it is
 * published (the blog widget reads `ai_generated`) until a workspace member marks it
 * reviewed and takes editorial responsibility for it (AI Act Art. 50(4) ¶2). Editing
 * the post afterwards clears the review and the label returns.
 */
import { CheckCircle, Loader2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useMarkPostReviewed } from '@/hooks/use-blogs-api'
import type { BlogPost } from '@/hooks/use-blogs-api'
import { useWorkspace } from '@/components/workspace-provider'

const AI_LABEL_STYLE = 'bg-[hsl(var(--warning))]/15 text-[hsl(var(--warning))] border-transparent'
const REVIEWED_STYLE = 'text-[hsl(var(--success))]'

interface BlogReviewControlProps {
  post: BlogPost
  /** The form holds edits not yet saved: saving them would clear a review, so save first. */
  hasUnsavedEdits: boolean
}

function reviewHint(canEdit: boolean, hasUnsavedEdits: boolean): string {
  if (!canEdit) return 'A workspace editor can mark it reviewed.'
  if (hasUnsavedEdits) return 'Save your edits first, then mark the post reviewed.'
  return 'Mark it reviewed once you have read it and take editorial responsibility for it.'
}

export function BlogReviewControl({ post, hasUnsavedEdits }: BlogReviewControlProps) {
  const { canEdit } = useWorkspace()
  const markReviewed = useMarkPostReviewed()

  if (post.reviewed_at) {
    return (
      <div className="glass-card p-3 flex items-start gap-2 text-xs text-muted-foreground" data-testid="blog-review-done">
        <CheckCircle className={`w-4 h-4 shrink-0 ${REVIEWED_STYLE}`} aria-hidden="true" />
        <p>
          <span className={`font-medium ${REVIEWED_STYLE}`}>
            Reviewed {new Date(post.reviewed_at).toLocaleDateString()}
          </span>
          {post.author_agent_id !== null && '. Editing the post clears the review, and it is labelled AI-generated again.'}
        </p>
      </div>
    )
  }
  if (!post.ai_generated) return null

  const disabled = !canEdit || hasUnsavedEdits || markReviewed.isLoading
  return (
    <div className="glass-card p-3 space-y-2" data-ai-generated="true">
      <div className="flex items-center justify-between gap-3">
        <Badge variant="outline" className={AI_LABEL_STYLE} aria-label="AI-generated">
          AI-generated
        </Badge>
        <Button size="sm" variant="outline" onClick={() => markReviewed.mutate(post.id)} disabled={disabled}>
          {markReviewed.isLoading ? (
            <Loader2 className="w-4 h-4 mr-1 animate-spin" />
          ) : (
            <CheckCircle className="w-4 h-4 mr-1" />
          )}
          Mark as reviewed
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        An agent wrote this post, so it is labelled AI-generated where it is published.{' '}
        {reviewHint(canEdit, hasUnsavedEdits)}
      </p>
    </div>
  )
}
