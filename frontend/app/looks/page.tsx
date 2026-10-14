'use client'

import { LOOKS_PLAY_ONLY_NOTE, LOOKS_TITLE, LooksScreen } from '@/components/looks/looks-screen'
import { PlayOnlyRoute } from '@/components/looks/play-only-route'

/**
 * PRD-260 §4, K9 — Make it yours. Under Play: MainLayout (Play's frame) with
 * the look editor. Under Classic or Studio: a short note with 'Switch to Play'.
 * The style comes from the UiStyleProvider, which the server starts from the
 * style cookie, so the first paint is already the right one.
 */
export default function LooksPage() {
  return (
    <PlayOnlyRoute title={LOOKS_TITLE} note={LOOKS_PLAY_ONLY_NOTE}>
      <LooksScreen />
    </PlayOnlyRoute>
  )
}
