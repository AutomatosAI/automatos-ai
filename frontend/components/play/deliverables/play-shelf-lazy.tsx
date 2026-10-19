'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — builtin Play's page for /deliverables: its
 * registrations (lib/theme-host/builtin/play-slots.tsx) name this slot, and
 * ThemeSlot in app/deliverables/layout.tsx draws it (PRD-261 A-04b). Only
 * play-slots.tsx imports this small module, so Play's shelf, its Deliverable page
 * and their CSS load in their own chunk when the shelf renders, never for the
 * Classic or Studio Deliverables page. SSR stays on.
 */
export const PlayShelf = dynamic(() => import('./play-shelf').then((m) => m.PlayShelf))
