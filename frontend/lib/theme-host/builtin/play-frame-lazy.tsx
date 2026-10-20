'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — the slot MainLayout renders Play's frame through.
 * The frame, its sidebar, the asleep screen and their CSS sit in their own
 * chunk, fetched only when the frame renders (under Play at >= 1024 px), so a
 * Classic or Studio page never downloads them. SSR stays on: a first paint
 * under Play still has the frame, and Next preloads the chunk only for a page
 * that rendered it.
 *
 * PRD-261 A-14f — host code since W-A5, as the chat's and the shelf's slots
 * (./play-chat-lazy.tsx, ./play-shelf-lazy.tsx): next/dynamic is host-only
 * (docs/theme-host/sdk-surface-map.md), so Play's frame folder imports none of it.
 * Only ./play-slots.tsx imports this module.
 */
export const PlayFrame = dynamic(() => import('@/components/play/shell/play-frame').then((m) => m.PlayFrame))
