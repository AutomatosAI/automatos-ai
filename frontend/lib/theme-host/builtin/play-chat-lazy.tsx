'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — builtin Play's page for /chat: its registrations
 * (./play-slots.tsx) name this slot, and ThemeSlot in app/chat/layout.tsx draws
 * it (PRD-261 A-04b). Only play-slots.tsx imports this small module, so Play's
 * chat, its turns and their CSS load in their own chunk when Play's chat
 * renders, never for a Classic or Studio chat. SSR stays on.
 *
 * PRD-261 A-14a — host code since W-A5: next/dynamic and MainLayout are host-only
 * (docs/theme-host/sdk-surface-map.md), so Play's chat imports neither, and this
 * slot draws the frame around it as ThemeSlot frames an installed theme's page:
 * MainLayout full bleed, exactly the tree Play's chat drew itself before. MainLayout
 * comes with import() too: a static import would close a cycle (main-layout →
 * theme-slot-registrations → ./slots → ./play-slots → here → main-layout) in which
 * ./slots reads ./play-slots' registrations before they exist whenever ./play-slots
 * is the first of them to load.
 */
const PlayFrameLayout = dynamic(() => import('@/components/layout/main-layout').then((m) => m.MainLayout))
const PlayChatPage = dynamic(() => import('@/components/play/chat/play-chat').then((m) => m.PlayChat))

export function PlayChat() {
  return (
    <PlayFrameLayout fullBleed>
      <PlayChatPage />
    </PlayFrameLayout>
  )
}
