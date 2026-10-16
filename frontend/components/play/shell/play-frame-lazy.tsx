'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — the slot MainLayout renders Play's frame through.
 * The frame, its sidebar, the asleep screen and their CSS sit in their own
 * chunk, fetched only when the frame renders (under Play at >= 1024 px), so a
 * Classic or Studio page never downloads them. SSR stays on: a first paint
 * under Play still has the frame, and Next preloads the chunk only for a page
 * that rendered it.
 */
export const PlayFrame = dynamic(() => import('./play-frame').then((m) => m.PlayFrame))
