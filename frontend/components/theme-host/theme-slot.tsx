'use client'

import { Suspense, type ComponentType, type ReactNode } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'

import { MainLayout } from '@/components/layout/main-layout'
import type { SlotId } from '@/lib/theme-host/registry-types'

import { registeredComponent, useActiveThemeSlots, type ActiveThemeSlots } from './theme-slot-registrations'

/**
 * PRD-261 A-04b — the seam a theme's page or slot mounts through (it generalises PRD-260 K7's
 * PlayRouteSwitch to every theme). A route's layout (or the 404) wraps the core's own version in
 * it, and it renders the active theme's version instead when the theme registered one:
 *
 * - a **page** entry (`page="/chat"`, registerPage): on that route only, unless the link asks for
 *   a surface only the core page has (`keepCorePage` names those query parameters; `themeServes`
 *   names values of one that the theme's page still is). An installed theme's page draws full
 *   bleed inside MainLayout (the theme's frame), as its routes at /x/<route> do (A-04d); builtin
 *   Play's pages draw MainLayout themselves (ActiveThemeSlots.framesPages).
 * - a **slot** (`slot="notFound@1"`, registerSlot): wherever it is placed.
 *
 * Where the theme decides (it registered this page or slot, or an installed theme is still
 * loading and may have), nothing renders while the layout is 'pending' (the width is not known:
 * the server and hydration), so neither version mounts for a render it then loses (P260-FX-25,
 * M69). With no theme, below the theme's width ('narrow'), or with nothing registered here, the
 * core's version renders, exactly as before. The core's page files are never edited.
 *
 * The page entry adds no Suspense boundary of its own, and the core page is never a fallback.
 * React's streaming server renderer outlines a finished boundary over 500 bytes once the bytes
 * flushed before it and its own pass 12,800 (progressiveChunkSize): it streams the fallback first
 * and the content again in a hidden <div id="S:…">, which a script swaps in on a later frame. With
 * the core page as the fallback of a boundary around it (/agents and /chat are about 17 KB), the
 * server sent the page twice, and when a provider above the boundary changed before the swap,
 * React gave up hydrating the boundary and rendered the page again in the browser beside the
 * hidden copy (react/react#37014): two Studio shells.
 *
 * So the choice reads the search params with no boundary around it (PRD-260's M14 added one
 * against a static prerender). Next asks for one only where a route is prerendered, where reading
 * them bails out to the browser; none is (app/layout.tsx is force-dynamic and reads cookies()), so
 * they never suspend, and the core pages under this seam read them with no boundary of their own
 * (app/playbooks/page.tsx, app/chat/page.tsx). Were a route prerendered, `next build` would stop
 * on it (missing-suspense-with-csr-bailout) rather than ship a broken page. And the core page sits
 * at one place in the tree in every state: a style switched in place (Studio and builtin Play, an
 * installed theme and Studio), the width becoming known, an installed theme settling or failing,
 * never remounts it (M69).
 */

/** Values of a `keepCorePage` parameter that the theme's page still serves. Serializable: a server layout passes it. */
export type ThemeServedValues = Readonly<Record<string, readonly string[]>>

export interface ThemePageSlotProps {
  /** The core page's route, as THEME_PAGE_ROUTES names it ('/chat'; '/missions/[id]' matches one segment). */
  readonly page: string
  /** Query parameters that keep the core page ('repo' on /chat opens the Code canvas). */
  readonly keepCorePage?: readonly string[]
  /** Values of a `keepCorePage` parameter the theme's page serves ({ tab: ['outputs'] } on /deliverables). */
  readonly themeServes?: ThemeServedValues
  /** The core's own page. */
  readonly children: ReactNode
}

export interface ThemeNamedSlotProps {
  /** The slot ('notFound@1'). */
  readonly slot: SlotId
  /** The core's own version. */
  readonly children: ReactNode
}

export type ThemeSlotProps = ThemePageSlotProps | ThemeNamedSlotProps

const NO_PARAMS: readonly string[] = []
const NO_SERVED_VALUES: ThemeServedValues = {}
const DYNAMIC_SEGMENT = /^\[[^\]/]+\]$/

/** Whether `pathname` is the route `page` names: the same segments, a `[param]` matching any one. */
export function routeMatches(page: string, pathname: string | null): boolean {
  if (pathname === null) return false
  const want = page.split('/')
  const have = pathname.split('/')
  if (want.length !== have.length) return false
  return want.every((segment, index) => segment === have[index] || (DYNAMIC_SEGMENT.test(segment) && have[index] !== ''))
}

/** Whether the link asks for a surface only the core page has. */
export function corePageAsked(
  params: Pick<URLSearchParams, 'has' | 'get'> | null,
  keepCorePage: readonly string[],
  themeServes: ThemeServedValues,
): boolean {
  if (params === null) return false
  return keepCorePage.some(
    (name) => params.has(name) && !(themeServes[name] ?? NO_PARAMS).includes(params.get(name) ?? ''),
  )
}

/**
 * The theme's version if it draws here now, null for nothing yet, or the core's version. `framed`
 * draws the theme's version inside MainLayout (an installed theme's page).
 */
function chooseVersion(active: ActiveThemeSlots, registered: unknown, core: ReactNode, framed = false): ReactNode {
  const Theme = registeredComponent(registered)
  if (active.layout === 'none' || active.layout === 'narrow') return core
  if (active.layout === 'pending') return active.registrations === null || Theme !== null ? null : core
  return Theme === null ? core : <ThemeVersion component={Theme} framed={framed} />
}

interface ThemeVersionProps {
  readonly component: ComponentType
  readonly framed: boolean
}

/** The theme's component; a lazy one (an installed theme's React.lazy) suspends to nothing. */
function ThemeVersion({ component: Component, framed }: ThemeVersionProps) {
  const version = (
    <Suspense fallback={null}>
      <Component />
    </Suspense>
  )
  return framed ? <MainLayout fullBleed>{version}</MainLayout> : version
}

/** A page entry: the theme's page on its route, nothing yet, or the core page, always at the same place in the tree. */
function ThemePageSlot({ page, keepCorePage = NO_PARAMS, themeServes = NO_SERVED_VALUES, children }: ThemePageSlotProps) {
  const active = useActiveThemeSlots()
  const pathname = usePathname()
  const params = useSearchParams()
  const here = routeMatches(page, pathname) && !corePageAsked(params, keepCorePage, themeServes)
  if (!here) return <>{children}</>
  return <>{chooseVersion(active, active.registrations?.pages.get(page), children, active.framesPages)}</>
}

function ThemeNamedSlot({ slot, children }: ThemeNamedSlotProps) {
  const active = useActiveThemeSlots()
  return <>{chooseVersion(active, active.registrations?.slots.get(slot), children)}</>
}

export function ThemeSlot(props: ThemeSlotProps) {
  if ('slot' in props) return <ThemeNamedSlot {...props} />
  return <ThemePageSlot {...props} />
}
