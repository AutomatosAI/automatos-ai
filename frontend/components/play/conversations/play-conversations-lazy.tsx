'use client'

import { lazy } from 'react'

/**
 * PRD-261 A-04b — the slot builtin Play registers as its route `conversations`, served at
 * /x/conversations (lib/theme-host/builtin/play-slots.tsx). Every page reaches builtin Play's
 * registrations, so they name only this small module: Conversations, its list and their CSS
 * load in their own chunk when that route renders, never for another page. SSR stays on.
 *
 * PRD-261 A-14c: React.lazy and import(), not next/dynamic (sdk-surface-map.md: a theme splits
 * its own bundle; next/* is host-only). The host's ThemeSlotRoute already renders every route
 * inside its own Suspense boundary (fallback null, as next/dynamic's was), so this adds none.
 */
export const PlayConversations = lazy(() =>
  import('./play-conversations').then((m) => ({ default: m.PlayConversations })),
)
