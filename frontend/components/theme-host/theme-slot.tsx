'use client'

import { Suspense, type ComponentType, type ReactNode } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'

import type { SlotId } from '@/lib/theme-host/registry-types'

import { registeredComponent, useActiveThemeSlots, type ActiveThemeSlots } from './theme-slot-registrations'

/**
 * PRD-261 A-04b — the seam a theme's page or slot mounts through (it generalises PRD-260 K7's
 * PlayRouteSwitch to every theme). A route's layout (or the 404) wraps the core's own version in
 * it, and it renders the active theme's version instead when the theme registered one:
 *
 * - a **page** entry (`page="/chat"`, registerPage): on that route only, unless the link asks for
 *   a surface only the core page has (`keepCorePage` names those query parameters; `themeServes`
 *   names values of one that the theme's page still is). The choice reads the search params, so
 *   it sits inside its own Suspense boundary whose fallback is the core page: a static prerender
 *   renders the core page.
 * - a **slot** (`slot="notFound@1"`, registerSlot): wherever it is placed.
 *
 * Where the theme decides (it registered this page or slot, or an installed theme is still
 * loading and may have), nothing renders while the layout is 'pending' (the width is not known:
 * the server and hydration), so neither version mounts for a render it then loses (P260-FX-25,
 * M69). With no theme, below the theme's width ('narrow'), or with nothing registered here, the
 * core's version renders, exactly as before. The core's page files are never edited.
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

/** The theme's version if it draws here now, null for nothing yet, or the core's version. */
function chooseVersion(active: ActiveThemeSlots, registered: unknown, core: ReactNode): ReactNode {
  const Theme = registeredComponent(registered)
  if (active.layout === 'none' || active.layout === 'narrow') return core
  if (active.layout === 'pending') return active.registrations === null || Theme !== null ? null : core
  return Theme === null ? core : <ThemeVersion component={Theme} />
}

/** The theme's component; a lazy one (an installed theme's React.lazy) suspends to nothing. */
function ThemeVersion({ component: Component }: { readonly component: ComponentType }) {
  return (
    <Suspense fallback={null}>
      <Component />
    </Suspense>
  )
}

function ThemePageChoice({ page, keepCorePage = NO_PARAMS, themeServes = NO_SERVED_VALUES, children }: ThemePageSlotProps) {
  const active = useActiveThemeSlots()
  const pathname = usePathname()
  const params = useSearchParams()
  const here = routeMatches(page, pathname) && !corePageAsked(params, keepCorePage, themeServes)
  if (!here) return <>{children}</>
  return <>{chooseVersion(active, active.registrations?.pages.get(page), children)}</>
}

function ThemeNamedSlot({ slot, children }: ThemeNamedSlotProps) {
  const active = useActiveThemeSlots()
  return <>{chooseVersion(active, active.registrations?.slots.get(slot), children)}</>
}

export function ThemeSlot(props: ThemeSlotProps) {
  if ('slot' in props) return <ThemeNamedSlot {...props} />
  return (
    <Suspense fallback={props.children}>
      <ThemePageChoice {...props} />
    </Suspense>
  )
}
