'use client'

import { isDisabled } from '@/components/settings/themes/theme-labels'
import { Button } from '@/components/ui/button'
import { PageHeader } from '@/components/shared/page-header'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import type { InstalledTheme } from '@/lib/api-client'
import { BUILTIN_PLAY_REF, BUILTIN_PLAY_STYLE } from '@/lib/theme-host/builtin/play-entry'
import { packageIdOfBuiltin, useChooseStyle, useWantInstalledThemes } from '@/lib/theme-host/choice-sync'
import { themeStyleOf } from '@/lib/theme-host/registry'
import type { UiStyle } from '@/lib/theme-host/registry-types'

/**
 * PRD-260 K9, D7 — the notes the pages only Play has (Make it yours at /looks, the style sheet at
 * /looks/style) show where Play's screen does not: under Classic or Studio, a short note with
 * 'Switch to Play' (the theme menu's Play choice, through useChooseStyle: the pick is stored on the
 * server, which refuses it when members may not choose, and a refused pick is put back; builtin
 * Play shows at once, never a redirect loop; once the workspace installed Play's own package the
 * builtin entry is hidden (D1-c), so the switch picks that install, as the menu would); under Play below 1024 px, where Play's pages don't
 * render (D7), a note that it opens on a wider screen. PRD-261 A-04b: the pages pick the note or
 * the theme's screen through components/theme-host/theme-slot-page.tsx (ThemeSlotPage), which
 * renders each note inside MainLayout. The notes wear the core's own frame, the shared PageHeader
 * (Studio's editorial head under Studio, the classic head under Classic: PRD-244,
 * app/__tests__/page-frames), since Play's variables exist only under html.play. Each note is a
 * region named by the page, so its words and Switch to Play read as one block.
 * Switch to Play replaces the whole note, the button with it: once Play's screen renders, its h1
 * (marked PLAY_ONLY_HEADING, tabIndex={-1}) takes the focus that dropped to <body> (P260-FX-22,
 * M21), which ThemeSlotPage finds by SCREEN_HEADING_SELECTOR.
 */

export const SWITCH_TO_PLAY = 'Switch to Play'
export const WIDER_SCREEN_NOTE = "It opens on a laptop or desktop screen: Play's pages need a wider window."

const HEADING_ATTRIBUTE = 'data-play-only-heading'
/** Spread on a Play-only screen's h1 (with tabIndex={-1}): where focus goes after Switch to Play. */
export const PLAY_ONLY_HEADING = { [HEADING_ATTRIBUTE]: true } as const
/** The screen's heading, as ThemeSlotPage looks for it once the screen replaces the note. */
export const SCREEN_HEADING_SELECTOR = `[${HEADING_ATTRIBUTE}]`

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

const PLAY_PACKAGE_ID = packageIdOfBuiltin(BUILTIN_PLAY_REF)

/** The style 'Switch to Play' picks: the workspace's usable install of Play's package, else builtin Play. */
export function playStyleFor(installs: readonly InstalledTheme[]): UiStyle {
  const install = installs.find((row) => row.themeId === PLAY_PACKAGE_ID && !isDisabled(row))
  return install ? themeStyleOf(`install:${install.installId}`) : BUILTIN_PLAY_STYLE
}

export function PlayOnlyNote({ title, note, offerSwitch }: PlayOnlyNoteProps) {
  const uiStyle = useUiStyleOptional()
  const chooseStyle = useChooseStyle()
  // 'Switch to Play' picks Play's installed package when there is one, so the installs are wanted.
  useWantInstalledThemes(offerSwitch)
  const { lead, accent } = titleParts(title)
  const switchToPlay = () => {
    if (!uiStyle) {
      console.warn('[looks] Switch to Play needs the UiStyleProvider above it; nothing changed')
      return
    }
    chooseStyle(playStyleFor(uiStyle.installedThemes))
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
