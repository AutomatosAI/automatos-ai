'use client'

/**
 * PRD-251 S0.5 — the Socials list: posts grouped by status with counts, newest
 * first, with an empty state; "New draft"; and the selected post's detail with
 * the actions the caller's role allows. S1.1c adds this month's render minutes
 * under the heading; S1.3 opens the brand kit (D5) from here for a role that
 * edits it.
 */
import { useMemo, useState } from 'react'
import { formatDistanceToNow } from 'date-fns'
import { Loader2, Palette, Plus, Share2 } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { badgeVariants } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { BrandKitDialog } from '@/components/documents/blocks/BrandKitDialog'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { SocialsNewDraft } from './socials-new-draft'
import { SocialsPostDetail } from './socials-post-detail'
import { SocialsRenderMinutes } from './socials-render-minutes'
import { anyRendering, canAuthorPosts, canEditBrandKit, groupPostsByStatus } from './socials-status'

function updatedAgo(post: SocialPost): string {
  try {
    return formatDistanceToNow(new Date(post.updated_at || post.created_at), { addSuffix: true })
  } catch {
    return ''
  }
}

export function SocialsPostList({ role }: { role: Workspace['role'] }) {
  const { data, isLoading, isError, error } = useSocialPosts()
  const [creating, setCreating] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [brandKitOpen, setBrandKitOpen] = useState(false)

  const posts = useMemo(() => data?.posts ?? [], [data])
  const groups = useMemo(() => groupPostsByStatus(posts), [posts])
  const selected = posts.find((post) => post.id === selectedId) ?? null
  const canAuthor = canAuthorPosts(role)
  const canBrand = canEditBrandKit(role)

  const handleDraftDone = (post: SocialPost | null) => {
    setCreating(false)
    if (post) setSelectedId(post.id)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">Posts</h2>
          <p className="text-sm text-muted-foreground">
            Every post is approved one by one. An edit after approval sends it back for approval.
          </p>
          <SocialsRenderMinutes rendering={anyRendering(posts)} />
        </div>
        <div className="flex flex-wrap gap-2">
          {canBrand && (
            <Button size="sm" variant="outline" onClick={() => setBrandKitOpen(true)}>
              <Palette className="mr-1.5 h-4 w-4" aria-hidden />
              Brand kit
            </Button>
          )}
          {canAuthor && !creating && (
            <Button size="sm" onClick={() => setCreating(true)}>
              <Plus className="mr-1.5 h-4 w-4" aria-hidden />
              New draft
            </Button>
          )}
        </div>
      </div>

      {canBrand && <BrandKitDialog open={brandKitOpen} onOpenChange={setBrandKitOpen} />}

      {creating && <SocialsNewDraft onDone={handleDraftDone} />}

      {isLoading ? (
        <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          Loading posts…
        </div>
      ) : isError ? (
        <p className="text-sm text-destructive" role="alert">
          Could not load posts: {error instanceof Error ? error.message : 'unknown error'}
        </p>
      ) : groups.length === 0 ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/60 bg-card/20 px-6 py-10 text-center">
          <Share2 className="h-7 w-7 text-muted-foreground" aria-hidden />
          <p className="text-sm font-medium text-foreground">No posts yet</p>
          <p className="text-sm text-muted-foreground">
            {canAuthor ? 'Start one with New draft.' : 'Posts your team drafts will show here.'}
          </p>
        </div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
          <div className="space-y-5">
            {groups.map((group) => (
              <section key={group.status} aria-label={`${group.label} posts`} className="space-y-2">
                <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  {group.label} <span className={badgeVariants({ variant: 'secondary' })}>{group.posts.length}</span>
                </h3>
                <ul className="space-y-1.5">
                  {group.posts.map((post) => (
                    <li key={post.id}>
                      <button
                        type="button"
                        onClick={() => setSelectedId(post.id)}
                        aria-pressed={post.id === selectedId}
                        className={cn(
                          'flex w-full flex-col items-start gap-0.5 rounded-lg border border-border bg-card px-3 py-2 text-left transition',
                          'hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/40',
                          post.id === selectedId && 'border-primary/60',
                        )}
                      >
                        <span className="text-sm font-medium text-foreground">{post.title}</span>
                        <span className="text-xs text-muted-foreground">Updated {updatedAgo(post)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ))}
          </div>
          <div>
            {selected ? (
              <SocialsPostDetail post={selected} role={role} />
            ) : (
              <p className="text-sm text-muted-foreground">Select a post to see its status and actions.</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
