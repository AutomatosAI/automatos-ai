'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — the slot app/deliverables/layout.tsx hands
 * PlayRouteSwitch. The server layout's payload names only this small module,
 * so Play's shelf, its Deliverable page and their CSS load in their own chunk
 * when the shelf renders, never for the Classic or Studio Deliverables page.
 * SSR stays on.
 */
export const PlayShelf = dynamic(() => import('./play-shelf').then((m) => m.PlayShelf))
