'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-261 A-04b — the slot builtin Play registers as its style sheet (styleSheet@1,
 * lib/theme-host/builtin/play-slots.tsx). Every page reaches builtin Play's registrations, so
 * they name only this small module: the style sheet, Auto's 12 specimens and their CSS load in
 * their own chunk when /looks/style renders them, never for another page. SSR stays on.
 */
export const StyleScreen = dynamic(() => import('./style-screen').then((m) => m.StyleScreen))
