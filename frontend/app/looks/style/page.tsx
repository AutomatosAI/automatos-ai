'use client'

import { PlayOnlyRoute } from '@/components/looks/play-only-route'
import { STYLE_HEADING, STYLE_SHEET_NOTE, StyleScreen } from '@/components/looks/style/style-screen'

/**
 * PRD-260 §4, D8 — the live style sheet at /looks/style, through the same
 * PlayOnlyRoute as Make it yours at /looks, so the two Play-only pages behave
 * alike. Under Play at >= 1024 px: Play's style sheet on the active look, full
 * bleed inside MainLayout (Play's frame). Under Classic or Studio: the same
 * note and Switch to Play as /looks (K9), never a redirect, so no loop, and it
 * follows a style change on this page at once. Under Play below 1024 px, where
 * Play's pages don't render (D7): the same 'wider screen' note as /looks.
 */
export default function LookStylePage() {
  return (
    <PlayOnlyRoute title={STYLE_HEADING} note={STYLE_SHEET_NOTE}>
      <StyleScreen />
    </PlayOnlyRoute>
  )
}
