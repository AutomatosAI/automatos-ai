'use client'

/**
 * PRD-251B US-B107 — the Socials Studio: the header (the sub-navigation and the two
 * actions, studio-nav.tsx) and the view the URL names (studio-route.ts). The calendar
 * view is the posts (US-B108 puts them on the Command Center's grid, with this list as its
 * List), the Queue holds what waits for approval, and Plans is the campaigns view (B6: a
 * plan is a campaign). Brand kit opens the brand kit dialog in place. New post opens the
 * composer at ?post=new; New plan opens the plan form at ?view=plans&plan=new.
 */
import { useMemo, useState } from 'react'
import { ArrowLeft } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { BrandKitDialog } from '@/components/documents/blocks/BrandKitDialog'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { SocialsCampaigns } from '../socials-campaigns-view'
import { SocialsComposer } from '../socials-composer'
import { SocialsPostList } from '../socials-post-list'
import { canAuthorPosts, canEditBrandKit } from '../socials-status'
import { SocialsQueue, queuedPosts } from './socials-queue'
import { SocialsStudioNav } from './studio-nav'
import { NEW_PLAN, NEW_POST, useSocialsRoute, type GoTo, type SocialsRoute } from './studio-route'

interface StudioBodyProps {
  route: SocialsRoute
  go: GoTo
  role: Workspace['role']
  posts: SocialPost[]
}

function NewPost({ go }: { go: GoTo }) {
  return (
    <div className="space-y-3">
      <Button type="button" size="sm" variant="ghost" className="socials-back -ml-2" onClick={() => go({ post: null })}>
        <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden />
        Back
      </Button>
      <SocialsComposer onDone={(post) => go({ post: post?.id ?? null })} />
    </div>
  )
}

function StudioBody({ route, go, role, posts }: StudioBodyProps) {
  if (route.post === NEW_POST) return <NewPost go={go} />
  if (route.view === 'queue') {
    return <SocialsQueue role={role} posts={posts} selectedId={route.post} onSelect={(id) => go({ post: id })} />
  }
  if (route.view === 'plans') {
    return (
      <SocialsCampaigns
        role={role}
        posts={posts}
        creating={route.plan === NEW_PLAN}
        onCreatingChange={(on) => go({ plan: on ? NEW_PLAN : null })}
      />
    )
  }
  return <SocialsPostList role={role} focusPostId={route.post} />
}

interface SocialsStudioProps {
  role: Workspace['role']
  /** The page's ?post= (global search and notifications link a post there). */
  postId?: string | null
}

export function SocialsStudio({ role, postId = null }: SocialsStudioProps) {
  const [route, go] = useSocialsRoute(postId)
  const { data } = useSocialPosts()
  const posts = useMemo(() => data?.posts ?? [], [data])
  const [brandKitOpen, setBrandKitOpen] = useState(false)
  const canBrand = canEditBrandKit(role)

  return (
    <div className="socials-studio space-y-5">
      <SocialsStudioNav
        view={route.view}
        queueCount={queuedPosts(posts).length}
        canAuthor={canAuthorPosts(role)}
        canBrand={canBrand}
        onView={(view) => go({ view, post: null, plan: null })}
        onBrandKit={() => setBrandKitOpen(true)}
        onNewPlan={() => go({ view: 'plans', plan: NEW_PLAN, post: null })}
        onNewPost={() => go({ post: NEW_POST, plan: null })}
      />
      {canBrand && <BrandKitDialog open={brandKitOpen} onOpenChange={setBrandKitOpen} />}
      <StudioBody route={route} go={go} role={role} posts={posts} />
    </div>
  )
}
