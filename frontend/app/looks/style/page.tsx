'use client'

import { PlayOnlyNote, SCREEN_HEADING_SELECTOR, WIDER_SCREEN_NOTE } from '@/lib/theme-host/builtin/play-only-route'
import { STYLE_HEADING, STYLE_SHEET_NOTE } from '@/components/looks/style/style-screen'
import { ThemeSlotPage } from '@/components/theme-host/theme-slot-page'

/**
 * PRD-260 §4, D8; PRD-261 A-04b (PA-14) — the live style sheet at /looks/style: the active
 * theme's style sheet (styleSheet@1), through the same ThemeSlotPage as Make it yours at /looks,
 * so the two pages behave alike. Under a theme at its width: the sheet on the active look, full
 * bleed inside MainLayout (the theme's frame). Under Classic or Studio: the same note and Switch
 * to Play as /looks (K9), never a redirect, so no loop, and it follows a style change on this page
 * at once. Under a theme below its width (D7): the same 'wider screen' note as /looks.
 */
export default function LookStylePage() {
  return (
    <ThemeSlotPage
      slot="styleSheet@1"
      note={<PlayOnlyNote title={STYLE_HEADING} note={STYLE_SHEET_NOTE} offerSwitch />}
      narrowNote={<PlayOnlyNote title={STYLE_HEADING} note={WIDER_SCREEN_NOTE} offerSwitch={false} />}
      headingSelector={SCREEN_HEADING_SELECTOR}
    />
  )
}
