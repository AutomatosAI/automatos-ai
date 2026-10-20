'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — builtin Play's page for /chat: its registrations
 * (lib/theme-host/builtin/play-slots.tsx) name this slot, and ThemeSlot in
 * app/chat/layout.tsx draws it (PRD-261 A-04b). Only play-slots.tsx imports this
 * small module, so Play's chat, its turns and their CSS load in their own chunk
 * when Play's chat renders, never for a Classic or Studio chat. SSR stays on.
 */
export const PlayChat = dynamic(() => import('./play-chat').then((m) => m.PlayChat))
