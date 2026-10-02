'use client'

/**
 * PRD-251B US-B107 — where the Socials Studio is, held in the URL with the house `tab=`
 * convention: /deliverables?tab=socials&view=calendar|queue|plans, with &post=<id>|new for
 * a post, &plan=new for the plan form and &cal=month|week|list for the calendar's layout.
 *
 * A view change is a router.push, never a replace, so the browser's back button returns to
 * the previous view. The route is mirrored in state as well: the click answers at once, and
 * the URL decides whenever it changes (back, forward, a link from search or a notification).
 * An unknown view is the calendar; the deep link ?tab=socials&post=<id> keeps working.
 */
import { useCallback, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

export const SOCIALS_VIEWS = ['calendar', 'queue', 'plans'] as const
export type SocialsStudioView = (typeof SOCIALS_VIEWS)[number]

export const CALENDAR_LAYOUTS = ['month', 'week', 'list'] as const
export type CalendarLayout = (typeof CALENDAR_LAYOUTS)[number]

/** `post=new`: the editor for a post not created yet. `plan=new`: the plan form. */
export const NEW_POST = 'new'
export const NEW_PLAN = 'new'
/** `view=brand`: the brand kit, a dialog over the calendar until it is a view of its own (Wave 3). */
export const BRAND_VIEW = 'brand'

export const DELIVERABLES_PATH = '/deliverables'
export const SOCIALS_TAB = 'socials'

export interface SocialsRoute {
  view: SocialsStudioView
  /** A post's id, NEW_POST, or null. */
  post: string | null
  /** NEW_PLAN while the plan form is open, else null. */
  plan: string | null
  cal: CalendarLayout
}

function oneOf<T extends string>(value: string | null | undefined, allowed: readonly T[], fallback: T): T {
  return value && (allowed as readonly string[]).includes(value) ? (value as T) : fallback
}

/** The route a URL's query names; anything unknown falls back (the calendar, its month). */
export function parseSocialsRoute(params: URLSearchParams | null | undefined): SocialsRoute {
  return {
    view: oneOf(params?.get('view'), SOCIALS_VIEWS, 'calendar'),
    post: params?.get('post') || null,
    plan: params?.get('plan') || null,
    cal: oneOf(params?.get('cal'), CALENDAR_LAYOUTS, 'month'),
  }
}

/** The URL of a route: the defaults are left out, so the plain tab link stays the calendar. */
export function socialsHref(route: SocialsRoute): string {
  const query = new URLSearchParams({ tab: SOCIALS_TAB, view: route.view })
  if (route.cal !== 'month') query.set('cal', route.cal)
  if (route.post) query.set('post', route.post)
  if (route.plan) query.set('plan', route.plan)
  return `${DELIVERABLES_PATH}?${query.toString()}`
}

export type GoTo = (next: Partial<SocialsRoute>) => void

/**
 * The Studio's route and how to move it. `initialPost` is the page's ?post= as a prop
 * (both shells pass it): it opens that post when the URL itself names none.
 */
export function useSocialsRoute(initialPost: string | null = null): [SocialsRoute, GoTo] {
  const router = useRouter()
  const params = useSearchParams()
  const query = params?.toString() ?? ''
  const [route, setRoute] = useState<SocialsRoute>(() => {
    const parsed = parseSocialsRoute(params)
    return { ...parsed, post: parsed.post ?? initialPost }
  })

  // The URL decides when it changes: back, forward, or a link from elsewhere.
  useEffect(() => {
    if (query) setRoute(parseSocialsRoute(new URLSearchParams(query)))
  }, [query])

  const go = useCallback<GoTo>(
    (next) => {
      const target = { ...route, ...next }
      setRoute(target)
      // typedRoutes: a computed href is cast at the push, as the Command Center's are.
      router.push(socialsHref(target) as any)
    },
    [route, router],
  )
  return [route, go]
}
