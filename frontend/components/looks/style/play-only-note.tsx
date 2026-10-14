'use client'

import { PageHeader } from '@/components/shared/page-header'
import { Button } from '@/components/ui/button'
import { useUiStyleOptional } from '@/contexts/ui-style-context'

/**
 * PRD-260 K9 — what a Play-only page shows under Classic or Studio: the page's
 * name, a short note that it is part of the Play theme, and Switch to Play,
 * which is the theme menu's Play choice (setStyle('play')). Never a redirect,
 * so there is no loop; once the style is Play the page itself renders. Drawn
 * with the core's own header and button, so it reads as a Classic or Studio
 * page (Play's --pl-* tokens exist only under html.play).
 */

export const SWITCH_TO_PLAY = 'Switch to Play'

export interface PlayOnlyNoteProps {
  /** The page's name, as the shared PageHeader takes it: "Style" + "sheet". */
  readonly title: string
  readonly titleAccent: string
  /** "The style sheet is part of the Play theme." */
  readonly note: string
}

export function PlayOnlyNote({ title, titleAccent, note }: PlayOnlyNoteProps) {
  const uiStyle = useUiStyleOptional()
  const switchToPlay = (
    <Button type="button" onClick={() => uiStyle?.setStyle('play')}>
      {SWITCH_TO_PLAY}
    </Button>
  )
  return <PageHeader title={title} titleAccent={titleAccent} subtitle={note} actions={switchToPlay} />
}
