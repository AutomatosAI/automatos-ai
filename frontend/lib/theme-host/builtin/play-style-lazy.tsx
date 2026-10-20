'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-261 A-04b — the slot builtin Play registers as its style sheet (styleSheet@1,
 * ./play-slots.tsx). Every page reaches builtin Play's registrations, so they name only this
 * small module: the style sheet, Auto's 12 specimens and their CSS load in their own chunk when
 * /looks/style renders them, never for another page. SSR stays on.
 *
 * PRD-261 A-14d — host code since W-A5: next/dynamic is host-only (docs/theme-host/sdk-surface-map.md),
 * so the slot lives with builtin Play's registrations and Play's style sheet imports none of it.
 */
export const StyleScreen = dynamic(() => import('@/components/looks/style/style-screen').then((m) => m.StyleScreen))
