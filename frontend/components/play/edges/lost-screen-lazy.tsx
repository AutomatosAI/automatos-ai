'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — the slot NotFoundView renders Play's 404 through.
 * The 404 is part of every route, so Play's lost screen, Auto and their CSS
 * load in their own chunk only when that screen renders (under Play at
 * >= 1024 px); every other style gets the plain 404 without them. SSR stays on.
 */
export const LostScreen = dynamic(() => import('./lost-screen').then((m) => m.LostScreen))
