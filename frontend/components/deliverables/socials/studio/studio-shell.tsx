'use client'

/**
 * PRD-251B US-B107 — the Socials Studio: the header (the sub-navigation and the two
 * actions, studio-nav.tsx) and the view the URL names (studio-route.ts). The calendar
 * view is the posts (US-B108 puts them on the Command Center's grid, with this list as its
 * List), the Queue holds what waits for approval, and Plans is the campaigns view (B6: a
 * plan is a campaign). Brand kit opens the brand kit dialog in place. New post opens the
 * editor at ?post=new; New plan opens the plan form at ?view=plans&plan=new.
 * US-B108: the calendar view is the Socials calendar (Month · Week · List).
 * US-B109: ?post=<id> opens the post editor (studio/socials-post-page.tsx), and
 * ?view=brand (the Brand kit item's own link) opens the brand kit over the calendar.
 */
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'next/navigation'

import { BrandKitDialog } from '@/components/documents/blocks/BrandKitDialog'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { SocialsCampaigns } from '../socials-campaigns-view'
import { canAuthorPosts, canEditBrandKit } from '../socials-status'
import { SocialsCalendar } from './socials-calendar'
import { SocialsPostPage } from './socials-post-page'
import { SocialsQueue, queuedPosts } from './socials-queue'
import { SocialsStudioNav } from './studio-nav'
import { BRAND_VIEW, NEW_PLAN, NEW_POST, useSocialsRoute, type GoTo, type SocialsRoute } from './studio-route'

interface StudioBodyProps {
  route: SocialsRoute
  go: GoTo
  role: Workspace['role']
  posts: SocialPost[]
  loading: boolean
}

function StudioBody({ route, go, role, posts, loading }: StudioBodyProps) {
  if (route.post === NEW_POST) return <SocialsPostPage role={role} posts={posts} postId={NEW_POST} loading={loading} go={go} />
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
  if (route.post) return <SocialsPostPage role={role} posts={posts} postId={route.post} loading={loading} go={go} />
  return <SocialsCalendar role={role} posts={posts} route={route} go={go} />
}

interface SocialsStudioProps {
  role: Workspace['role']
  /** The page's ?post= (global search and notifications link a post there). */
  postId?: string | null
}

export function SocialsStudio({ role, postId = null }: SocialsStudioProps) {
  const [route, go] = useSocialsRoute(postId)
  const { data, isLoading, isFetching } = useSocialPosts()
  const posts = useMemo(() => data?.posts ?? [], [data])
  const [brandKitOpen, setBrandKitOpen] = useState(false)
  const canBrand = canEditBrandKit(role)
  const brandAsked = useSearchParams()?.get('view') === BRAND_VIEW

  // ?view=brand: the brand kit is a dialog over the calendar until its own view (Wave 3).
  useEffect(() => {
    if (brandAsked && canBrand) setBrandKitOpen(true)
  }, [brandAsked, canBrand])

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
      <StudioBody route={route} go={go} role={role} posts={posts} loading={isLoading || isFetching} />
    </div>
  )
}
