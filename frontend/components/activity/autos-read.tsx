'use client'

import { useState } from 'react'
import { Sparkles, ThumbsUp, ThumbsDown, AlertTriangle } from 'lucide-react'
import { useWorkspaceDigest, useSubmitDigestFeedback } from '@/hooks/use-digest-api'

// PRD-221 S13 — "Auto's read": a plain-English interpretation of the workspace
// state, mounted in BOTH Command Centre shells (classic ActivityPage + studio
// summary tab). One component, no fork.
export function AutosRead({ period = '1d' }: { period?: string }) {
  const { data, isLoading, isError } = useWorkspaceDigest(period)
  const submit = useSubmitDigestFeedback()
  const [rated, setRated] = useState<1 | -1 | null>(null)

  if (isLoading) {
    return (
      <div className="rounded-xl border border-border bg-card p-4">
        <div className="h-4 w-24 animate-pulse rounded bg-secondary/40" />
        <div className="mt-3 h-3 w-full animate-pulse rounded bg-secondary/30" />
        <div className="mt-2 h-3 w-2/3 animate-pulse rounded bg-secondary/30" />
      </div>
    )
  }

  if (isError || !data) return null

  const rate = (rating: 1 | -1) => {
    if (rated !== null) return
    setRated(rating)
    submit.mutate({ state_hash: data.state_hash, rating })
  }

  const generatedAt = data.generated_at
    ? new Date(data.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <div className="rounded-xl border border-border bg-card p-4" data-testid="autos-read">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-sm font-medium text-foreground">
          <Sparkles className="h-4 w-4 text-primary" />
          Auto&apos;s read
        </div>
        {data.needs_attention_count > 0 && (
          <span className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-xs font-medium text-warning">
            <AlertTriangle className="h-3 w-3" />
            {data.needs_attention_count} need{data.needs_attention_count === 1 ? 's' : ''} attention
          </span>
        )}
      </div>

      <p className="text-sm leading-relaxed text-muted-foreground">{data.text}</p>

      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {generatedAt ? `Updated ${generatedAt}` : ''}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            aria-label="Helpful"
            disabled={rated !== null}
            onClick={() => rate(1)}
            className={`rounded-md p-1 transition-colors hover:bg-muted disabled:opacity-60 ${
              rated === 1 ? 'text-primary' : 'text-muted-foreground'
            }`}
          >
            <ThumbsUp className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Not helpful"
            disabled={rated !== null}
            onClick={() => rate(-1)}
            className={`rounded-md p-1 transition-colors hover:bg-muted disabled:opacity-60 ${
              rated === -1 ? 'text-destructive' : 'text-muted-foreground'
            }`}
          >
            <ThumbsDown className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  )
}
