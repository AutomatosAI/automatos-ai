'use client'

/**
 * PRD-251B US-B107 — the Socials Studio: the header (the sub-navigation and the two
 * actions, studio-nav.tsx) and the view the URL names (studio-route.ts). The calendar
 * view is the posts (US-B108 puts them on the Command Center's grid, with this list as its
 * List), the Queue holds what waits for approval, and Plans lists the plans (B6: a plan is a
 * campaign; Wave 2) with the other campaigns below them. The brand kit is the Deliverables
 * Brand kit tab (PRD-251B US-B301; F250: no entry of its own here). New post opens the editor
 * at ?post=new; New plan opens the Plan page at
 * ?view=plans&plan=new, and ?view=plans&plan=<id> a plan's own page (US-B207).
 * US-B108: the calendar view is the Socials calendar (Month · Week · List).
 * US-B109: ?post=<id> opens the post editor (studio/socials-post-page.tsx), and the old
 * ?view=brand link goes to the Brand kit tab.
 */
import { useEffect, useMemo } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import type { Workspace } from '@/components/workspace-provider'
import { BRAND_KIT_HREF } from '@/lib/deliverables/tabs'
import type { SocialPost } from '@/lib/api-client'
import { useSocialPosts } from '@/hooks/use-socials-api'
import { SocialsPlansView } from '../plans/socials-plans-view'
import { canAuthorPosts } from '../socials-status'
import { SocialsCalendar } from './socials-calendar'
import { SocialsPostPage } from './socials-post-page'
import { queuedPosts } from './queue-model'
import { SocialsQueue } from './socials-queue'
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
  if (route.view === 'plans') return <SocialsPlansView role={role} posts={posts} planId={route.plan} go={go} />
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
  const router = useRouter()
  const brandAsked = useSearchParams()?.get('view') === BRAND_VIEW

  // ?view=brand: the brand kit is the Brand kit tab now.
  useEffect(() => {
    if (brandAsked) router.replace(BRAND_KIT_HREF as any)
  }, [brandAsked, router])

  return (
    <div className="socials-studio space-y-5">
      <SocialsStudioNav
        view={route.view}
        queueCount={queuedPosts(posts).length}
        canAuthor={canAuthorPosts(role)}
        onView={(view) => go({ view, post: null, plan: null })}
        onNewPlan={() => go({ view: 'plans', plan: NEW_PLAN, post: null })}
        onNewPost={() => go({ post: NEW_POST, plan: null })}
      />
      <StudioBody route={route} go={go} role={role} posts={posts} loading={isLoading || isFetching} />
    </div>
  )
}
