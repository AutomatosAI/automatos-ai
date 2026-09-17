'use client'

import { useEffect, useState } from 'react'
import { Sparkles, ThumbsUp, ThumbsDown, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import { useWorkspaceDigest, useSubmitDigestFeedback } from '@/hooks/use-digest-api'

/** Per-read collapse memory: keyed by the read's state hash, so a new read reappears. */
const COLLAPSE_KEY_PREFIX = 'autosRead:collapsed:'
const PREVIEW_CHARS = 120

function readCollapsed(hash: string | undefined): boolean {
  if (!hash) return false
  try {
    return window.localStorage.getItem(COLLAPSE_KEY_PREFIX + hash) === '1'
  } catch {
    return false
  }
}
function writeCollapsed(hash: string | undefined, collapsed: boolean): void {
  if (!hash) return
  try {
    window.localStorage.setItem(COLLAPSE_KEY_PREFIX + hash, collapsed ? '1' : '0')
  } catch {
    // Storage blocked: the choice lasts for this page only.
  }
}

/** The first sentence, trimmed to one line, for the collapsed state. */
export function readPreview(text: string): string {
  const first = text.split(/(?<=[.!?])\s/)[0]?.trim() ?? text.trim()
  return first.length > PREVIEW_CHARS ? `${first.slice(0, PREVIEW_CHARS - 1)}…` : first
}

// PRD-221 S13 — "Auto's read": a plain-English interpretation of the workspace
// state, mounted in BOTH Command Centre shells (classic ActivityPage + studio
// summary tab). One component, no fork. PRD-244 review (Gerard, 09-17): the
// card collapses to one line, and a thumbs-up collapses it — the read has done
// its job — until the next read arrives.
export function AutosRead({ period = '1d' }: { period?: string }) {
  const { data, isLoading, isError } = useWorkspaceDigest(period)
  const submit = useSubmitDigestFeedback()
  const [rated, setRated] = useState<1 | -1 | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const hash = data?.state_hash

  useEffect(() => {
    setCollapsed(readCollapsed(hash))
    setRated(null)
  }, [hash])

  const toggle = (next: boolean) => {
    setCollapsed(next)
    writeCollapsed(hash, next)
  }

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
    if (rating === 1) toggle(true)
  }

  const generatedAt = data.generated_at
    ? new Date(data.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null

  return (
    <div className={`rounded-xl border border-border bg-card ${collapsed ? 'px-4 py-2' : 'p-4'}`} data-testid="autos-read" data-collapsed={collapsed ? 'true' : 'false'}>
      <div className={`flex items-center justify-between gap-3 ${collapsed ? '' : 'mb-2'}`}>
        <div className="flex min-w-0 items-center gap-1.5 text-sm font-medium text-foreground">
          <Sparkles className="h-4 w-4 shrink-0 text-primary" />
          <span className="shrink-0">Auto&apos;s read</span>
          {collapsed && (
            <span className="truncate text-sm font-normal text-muted-foreground" title={data.text}>
              · {readPreview(data.text)}
            </span>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {data.needs_attention_count > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-xs font-medium text-warning">
              <AlertTriangle className="h-3 w-3" />
              {data.needs_attention_count} need{data.needs_attention_count === 1 ? 's' : ''} attention
            </span>
          )}
          <button
            type="button"
            aria-label={collapsed ? "Show Auto's read" : "Hide Auto's read"}
            aria-expanded={!collapsed}
            onClick={() => toggle(!collapsed)}
            className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            {collapsed ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>

      {!collapsed && <p className="text-sm leading-relaxed text-muted-foreground">{data.text}</p>}

      <div className={`flex items-center justify-between ${collapsed ? 'hidden' : 'mt-3'}`}>
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
