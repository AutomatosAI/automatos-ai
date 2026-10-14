'use client'

import { type ReactNode, useId } from 'react'

import { MainLayout } from '@/components/layout/main-layout'
import { Button } from '@/components/ui/button'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { useIsTabletOrBelow } from '@/hooks/use-mobile'
import { useIsPlay } from '@/hooks/use-studio-theme'

/**
 * PRD-260 K9, D7 — a page only Play has (Make it yours at /looks). Under Play
 * at >= 1024 px it renders Play's screen full bleed inside MainLayout (Play's
 * frame); under Classic or Studio, a short note with 'Switch to Play' (the
 * theme menu's Play choice: setStyle('play'), so the screen appears at once,
 * never a redirect loop); under Play below 1024 px, where Play's pages don't
 * render (D7), a note that it opens on a wider screen. The notes are drawn
 * with the core's tokens, since Play's variables exist only under html.play.
 */

export const SWITCH_TO_PLAY = 'Switch to Play'
export const WIDER_SCREEN_NOTE = "It opens on a laptop or desktop screen: Play's pages need a wider window."

export interface PlayOnlyRouteProps {
  /** The page's name, the note's heading ('Make it yours'). */
  readonly title: string
  /** What the note says under Classic or Studio ('Make it yours is part of the Play theme.'). */
  readonly note: string
  /** Play's screen for this page. */
  readonly children: ReactNode
}

export function PlayOnlyRoute({ title, note, children }: PlayOnlyRouteProps) {
  const isPlay = useIsPlay()
  const narrow = useIsTabletOrBelow()
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

export function PlayOnlyNote({ title, note, offerSwitch }: PlayOnlyNoteProps) {
  const uiStyle = useUiStyleOptional()
  const headingId = useId()
  const switchToPlay = () => {
    if (!uiStyle) {
      console.warn('[looks] Switch to Play needs the UiStyleProvider above it; nothing changed')
      return
    }
    uiStyle.setStyle('play')
  }
  return (
    <section
      aria-labelledby={headingId}
      className="mx-auto mt-12 max-w-md rounded-lg border border-border bg-card p-6 text-center"
    >
      <h1 id={headingId} className="text-lg font-semibold text-foreground">
        {title}
      </h1>
      <p className="mt-2 text-sm text-muted-foreground">{note}</p>
      {offerSwitch && (
        <Button type="button" className="mt-5" onClick={switchToPlay}>
          {SWITCH_TO_PLAY}
        </Button>
      )}
    </section>
  )
}
