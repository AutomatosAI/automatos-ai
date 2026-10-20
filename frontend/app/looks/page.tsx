'use client'

import { LOOKS_PLAY_ONLY_NOTE, LOOKS_TITLE } from '@/components/looks/looks-screen'
import { PlayOnlyNote, SCREEN_HEADING_SELECTOR, WIDER_SCREEN_NOTE } from '@/components/looks/play-only-route'
import { ThemeSlotPage } from '@/components/theme-host/theme-slot-page'

/**
 * PRD-260 §4, K9; PRD-261 A-04b (PA-14) — Make it yours: the active theme's look editor
 * (lookEditor@1), full bleed inside MainLayout (the theme's frame). Under Classic or Studio: a
 * short note with 'Switch to Play'. The style comes from the UiStyleProvider, which the server
 * starts from the style cookie, so the first paint is already the right one.
 */
export default function LooksPage() {
  return (
    <ThemeSlotPage
      slot="lookEditor@1"
      note={<PlayOnlyNote title={LOOKS_TITLE} note={LOOKS_PLAY_ONLY_NOTE} offerSwitch />}
      narrowNote={<PlayOnlyNote title={LOOKS_TITLE} note={WIDER_SCREEN_NOTE} offerSwitch={false} />}
      headingSelector={SCREEN_HEADING_SELECTOR}
    />
  )
}
