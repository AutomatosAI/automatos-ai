'use client'

import { Suspense, type ReactNode } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'

import { useThemeLayout } from '@/hooks/use-studio-theme'

/**
 * PRD-260 K7 — the seam a Play-designed page mounts through. A route's
 * existing layout wraps its page in this switch: under Play at >= 1024 px (D7),
 * on one of `paths`, it renders Play's screen instead of the page; otherwise the
 * page itself, exactly as before. The page file is never edited, so a page
 * component over the code-shape limit (ChatPage) is never touched.
 *
 * `keepPageFor` names query parameters that ask for a surface Play's screen
 * doesn't have (the chat's Code canvas, a session's runtime, plan mode): with
 * one present the route's own page renders inside Play's frame, as every page
 * without a Play design does (D8). `playServes` names the values of such a
 * parameter that Play's screen IS (the Deliverables page's Outputs tab is
 * Play's shelf), so a link carrying one keeps Play's screen.
 *
 * The choice reads the search params, so it sits inside its own Suspense
 * boundary whose fallback is the route's own page: a static prerender renders
 * the page, and both layouts are covered without a boundary of their own.
 *
 * Where the width decides (under Play, on one of `paths`, no page asked for),
 * it renders nothing until the width is known (useThemeLayout 'pending': the
 * server and hydration), so neither Play's screen nor the page mounts for a
 * render the width then takes back (on a phone, Play's chat consuming a deep
 * link and racing the cold start; on a desktop, the route's own page and its
 * deep-link sheet): P260-FX-25, M69.
 */

/** Values of a `keepPageFor` parameter that Play's screen still serves. Serializable: a server layout passes it. */
export type PlayServedValues = Readonly<Record<string, readonly string[]>>

export interface PlayRouteSwitchProps {
  /** Play's screen for these paths. */
  readonly play: ReactNode
  /** The exact pathnames Play's screen serves ('/chat'). */
  readonly paths: readonly string[]
  /** Query parameters that keep the route's own page ('repo' on /chat opens the Code canvas). */
  readonly keepPageFor?: readonly string[]
  /** Values of a `keepPageFor` parameter Play's screen serves ({ tab: ['outputs'] } on /deliverables). */
  readonly playServes?: PlayServedValues
  /** The route's own page: everything else. */
  readonly children: ReactNode
}

const NO_PARAMS: readonly string[] = []
const NO_SERVED_VALUES: PlayServedValues = {}

/** Whether the link asks for a surface only the route's own page has. */
function pageAsked(
  params: Pick<URLSearchParams, 'has' | 'get'> | null,
  keepPageFor: readonly string[],
  playServes: PlayServedValues,
): boolean {
  if (params === null) return false
  return keepPageFor.some(
    (name) => params.has(name) && !(playServes[name] ?? NO_PARAMS).includes(params.get(name) ?? ''),
  )
}

function PlayRouteChoice({
  play,
  paths,
  keepPageFor = NO_PARAMS,
  playServes = NO_SERVED_VALUES,
  children,
}: PlayRouteSwitchProps) {
  const layout = useThemeLayout()
  const pathname = usePathname()
  const params = useSearchParams()
  const playsHere = pathname !== null && paths.includes(pathname) && !pageAsked(params, keepPageFor, playServes)
  if (layout === 'fallback' || !playsHere) return <>{children}</>
  return layout === 'theme' ? <>{play}</> : null
}

export function PlayRouteSwitch(props: PlayRouteSwitchProps) {
  return (
    <Suspense fallback={props.children}>
      <PlayRouteChoice {...props} />
    </Suspense>
  )
}
