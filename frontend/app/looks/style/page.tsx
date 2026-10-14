'use client'

import { MainLayout } from '@/components/layout/main-layout'
import { PlayOnlyNote } from '@/components/looks/style/play-only-note'
import { STYLE_SHEET_NOTE, StyleScreen } from '@/components/looks/style/style-screen'
import { useIsPlay } from '@/hooks/use-studio-theme'

/**
 * PRD-260 §4, D8 — the live style sheet at /looks/style. Under Play it is
 * Play's style sheet on the active look, inside MainLayout (Play's frame at
 * >= 1024 px). Under Classic or Studio it is the same short note as /looks
 * with Switch to Play (K9): never a redirect, so no loop, and it follows a
 * style change on this page at once.
 */
export default function LookStylePage() {
  const isPlay = useIsPlay()
  return (
    <MainLayout>
      {isPlay ? <StyleScreen /> : <PlayOnlyNote title="Style" titleAccent="sheet" note={STYLE_SHEET_NOTE} />}
    </MainLayout>
  )
}
