'use client'

import type { ReactNode } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'

import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useIsPlay } from '@/hooks/use-studio-theme'

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
 * without a Play design does (D8). The switch reads the search params, so on a
 * statically rendered route it sits inside a Suspense boundary.
 */

export interface PlayRouteSwitchProps {
  /** Play's screen for these paths. */
  readonly play: ReactNode
  /** The exact pathnames Play's screen serves ('/chat'). */
  readonly paths: readonly string[]
  /** Query parameters that keep the route's own page ('repo' on /chat opens the Code canvas). */
  readonly keepPageFor?: readonly string[]
  /** The route's own page: everything else. */
  readonly children: ReactNode
}

const NO_PARAMS: readonly string[] = []

export function PlayRouteSwitch({ play, paths, keepPageFor = NO_PARAMS, children }: PlayRouteSwitchProps) {
  const isPlay = useIsPlay()
  const narrow = useIsTabletOrBelow()
  const pathname = usePathname()
  const params = useSearchParams()
  const pageAsked = keepPageFor.some((name) => params?.has(name) ?? false)
  const showPlay = isPlay && !narrow && pathname !== null && paths.includes(pathname) && !pageAsked
  return <>{showPlay ? play : children}</>
}
