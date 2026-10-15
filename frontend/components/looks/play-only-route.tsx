'use client'

import { type ReactNode, useEffect, useRef } from 'react'

import { MainLayout } from '@/components/layout/main-layout'
import { PageHeader } from '@/components/shared/page-header'
import { Button } from '@/components/ui/button'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useIsPlay } from '@/hooks/use-studio-theme'
import { refocusWhenShown } from '@/lib/play/focus'

/**
 * PRD-260 K9, D7 — a page only Play has (Make it yours at /looks, the style
 * sheet at /looks/style). Under Play
 * at >= 1024 px it renders Play's screen full bleed inside MainLayout (Play's
 * frame); under Classic or Studio, a short note with 'Switch to Play' (the
 * theme menu's Play choice: setStyle('play'), so the screen appears at once,
 * never a redirect loop); under Play below 1024 px, where Play's pages don't
 * render (D7), a note that it opens on a wider screen. The notes wear the
 * core's own frame, the shared PageHeader (Studio's editorial head under
 * Studio, the classic head under Classic: PRD-244, app/__tests__/page-frames),
 * since Play's variables exist only under html.play. Each note is a region
 * named by the page, so its words and Switch to Play read as one block.
 * Switch to Play replaces the whole frame, the button with it: once Play's
 * screen renders, its h1 (marked PLAY_ONLY_HEADING, tabIndex={-1}) takes the
 * focus that dropped to <body> (P260-FX-22, M21).
 */

export const SWITCH_TO_PLAY = 'Switch to Play'
export const WIDER_SCREEN_NOTE = "It opens on a laptop or desktop screen: Play's pages need a wider window."

const HEADING_ATTRIBUTE = 'data-play-only-heading'
/** Spread on a Play-only screen's h1 (with tabIndex={-1}): where focus goes after Switch to Play. */
export const PLAY_ONLY_HEADING = { [HEADING_ATTRIBUTE]: true } as const

export interface PlayOnlyRouteProps {
  /** The page's name, the note's heading ('Make it yours'). */
  readonly title: string
  /** What the note says under Classic or Studio ('Make it yours is part of the Play theme.'). */
  readonly note: string
  /** Play's screen for this page. */
  readonly children: ReactNode
}

/**
 * Remembers that the note offered Switch to Play; once Play's screen shows, its
 * heading takes focus (when Play's code loads late, as soon as it renders).
 */
function useHeadingFocusAfterSwitch(isPlay: boolean, showsScreen: boolean): void {
  const offered = useRef(false)
  useEffect(() => {
    if (!isPlay) {
      offered.current = true
      return undefined
    }
    if (!showsScreen || !offered.current) return undefined
    offered.current = false
    return refocusWhenShown(`[${HEADING_ATTRIBUTE}]`)
  }, [isPlay, showsScreen])
}

export function PlayOnlyRoute({ title, note, children }: PlayOnlyRouteProps) {
  const isPlay = useIsPlay()
  const narrow = useIsTabletOrBelow()
  useHeadingFocusAfterSwitch(isPlay, isPlay && !narrow)
  if (!isPlay) {
    return (
      <MainLayout>
        <PlayOnlyNote title={title} note={note} offerSwitch />
      </MainLayout>
    )
  }
  if (narrow) {
    return (
      <MainLayout>
        <PlayOnlyNote title={title} note={WIDER_SCREEN_NOTE} offerSwitch={false} />
      </MainLayout>
    )
  }
  return <MainLayout fullBleed>{children}</MainLayout>
}

interface PlayOnlyNoteProps {
  readonly title: string
  readonly note: string
  /** Show 'Switch to Play' (not when Play is already the style). */
  readonly offerSwitch: boolean
}

/** PageHeader's two-part title: the words before the last, then the last word as its accent. */
function titleParts(title: string): { readonly lead: string; readonly accent: string } {
  const words = title.trim().split(/\s+/)
  return { lead: words.slice(0, -1).join(' '), accent: words[words.length - 1] ?? '' }
}

export function PlayOnlyNote({ title, note, offerSwitch }: PlayOnlyNoteProps) {
  const uiStyle = useUiStyleOptional()
  const { lead, accent } = titleParts(title)
  const switchToPlay = () => {
    if (!uiStyle) {
      console.warn('[looks] Switch to Play needs the UiStyleProvider above it; nothing changed')
      return
    }
    uiStyle.setStyle('play')
  }
  const action = offerSwitch ? (
    <Button type="button" onClick={switchToPlay}>
      {SWITCH_TO_PLAY}
    </Button>
  ) : undefined
  return (
    <section aria-label={title}>
      <PageHeader title={lead} titleAccent={accent} subtitle={note} actions={action} />
    </section>
  )
}
