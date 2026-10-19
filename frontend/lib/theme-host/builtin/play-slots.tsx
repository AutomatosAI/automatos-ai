'use client'

/**
 * PRD-261 A-04b (PA-7) — what builtin Play registers with the host, as an installed theme's
 * register(host) does: its frame (frame@1), its chat and Deliverables pages, its look editor,
 * style sheet and 404, and its route `conversations` (served at /x/conversations). They are
 * registered synchronously, as module constants, so the frame and the pages draw on the render
 * the width is known, never a render later (P260-FX-25, M69).
 *
 * Every component here is one of Play's lazy slots: this module is reached from every page
 * (./slots.ts, read by components/theme-host/theme-slot-registrations.tsx), so it names only the
 * small modules that fetch each screen with import() when it renders, and Classic and Studio
 * download none of Play's screens (P260-FX-23, M17; lib/__tests__/prd260-play-guards.test.ts).
 *
 * The keys are the host's own (THEME_SLOT_IDS, THEME_PAGE_ROUTES, THEME_ROUTE_PATTERN), and
 * ../__tests__/builtin-slots.test.tsx runs them through the loader's registerTheme, as an
 * installed theme's are.
 * Play's entry (./play-entry.ts) lists the same pages, routes and slots; asleep@1 is drawn by
 * Play's frame itself (AsleepGate), so it is not registered on its own.
 */
import { LooksScreen } from '@/components/looks/looks-screen-lazy'
import { StyleScreen } from '@/components/looks/style/style-screen-lazy'
import { PlayChat } from '@/components/play/chat/play-chat-lazy'
import { PlayConversations } from '@/components/play/conversations/play-conversations-lazy'
import { PlayShelf } from '@/components/play/deliverables/play-shelf-lazy'
import { LostScreen } from '@/components/play/edges/lost-screen-lazy'
import { PlayFrame } from '@/components/play/shell/play-frame-lazy'
import { CHAT_HREF, DELIVERABLES_HREF } from '@/lib/ticket-links'

import { FRAME_SLOT } from '../constants'
import type { SlotId } from '../registry-types'
import type { FrameProps, ThemeRegistrations } from '../types'

/** Builtin Play's route for its Conversations page: /x/conversations. */
export const PLAY_CONVERSATIONS_ROUTE = 'conversations'

/** frame@1: Play's frame around the page, with MainLayout's full-bleed flag and Auto's widget. */
function PlayFrameSlot({ children, fullBleed, autoWidget }: FrameProps) {
  return (
    <PlayFrame fullBleed={fullBleed} autoWidget={autoWidget}>
      {children}
    </PlayFrame>
  )
}

export const BUILTIN_PLAY_REGISTRATIONS: ThemeRegistrations = {
  frame: PlayFrameSlot,
  slots: new Map<SlotId, unknown>([
    [FRAME_SLOT, PlayFrameSlot],
    ['lookEditor@1', LooksScreen],
    ['styleSheet@1', StyleScreen],
    ['notFound@1', LostScreen],
  ]),
  pages: new Map<string, unknown>([
    [CHAT_HREF, PlayChat],
    [DELIVERABLES_HREF, PlayShelf],
  ]),
  routes: new Map<string, unknown>([[PLAY_CONVERSATIONS_ROUTE, PlayConversations]]),
}
