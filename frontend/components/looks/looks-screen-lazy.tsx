'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-261 A-04b — the slot builtin Play registers as its look editor (lookEditor@1,
 * lib/theme-host/builtin/play-slots.tsx). Every page reaches builtin Play's registrations, so
 * they name only this small module: Make it yours, its editor and their CSS load in their own
 * chunk when /looks renders them, never for another page. SSR stays on.
 */
export const LooksScreen = dynamic(() => import('./looks-screen').then((m) => m.LooksScreen))
