'use client'

import dynamic from 'next/dynamic'

/**
 * PRD-261 A-04b — the slot builtin Play registers as its route `conversations`, served at
 * /x/conversations (lib/theme-host/builtin/play-slots.tsx). Every page reaches builtin Play's
 * registrations, so they name only this small module: Conversations, its list and their CSS
 * load in their own chunk when that route renders, never for another page. SSR stays on.
 */
export const PlayConversations = dynamic(() => import('./play-conversations').then((m) => m.PlayConversations))
