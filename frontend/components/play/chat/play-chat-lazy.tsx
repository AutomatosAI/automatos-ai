'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-260 P260-FX-23 (M17) — the slot app/chat/layout.tsx hands PlayRouteSwitch.
 * The server layout's payload names only this small module, so Play's chat,
 * its turns and their CSS load in their own chunk when Play's chat renders,
 * never for a Classic or Studio chat. SSR stays on.
 */
export const PlayChat = dynamic(() => import('./play-chat').then((m) => m.PlayChat))
