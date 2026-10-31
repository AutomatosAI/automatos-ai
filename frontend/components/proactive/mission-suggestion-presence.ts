'use client'

/**
 * PRD-268 §6.12.4 (US-W01b) — one suggestion surface: whether a turn's Mission suggestion is waiting for an answer,
 * and which suggestion is pinned above the composer. Both are useSyncExternalStore stores (presence.ts).
 *
 * - While a MissionSuggestionCard is open in the chat (shown, and neither launched nor dismissed), the suggestion
 *   pinned above the composer hides. The card marks itself here (mission-suggestion-with-presence.tsx) and lets go
 *   when it is answered or unmounted; the pinned line reads `useMissionSuggestionOpen`.
 * - While a suggestion is pinned, its card in the thread steps aside (thread-cards.tsx reads `usePinnedSuggestionIds`),
 *   so each suggestion has one live Start and Hide on screen. Once the line hides, the thread card is back.
 */
import { createPresence, usePresence } from './presence'

const missionCards = createPresence<true>('mission-suggestion')
const pinnedSuggestions = createPresence<string>('pinned-suggestion')

/** True while some Mission suggestion is open and unanswered. */
export function missionSuggestionOpen(): boolean {
  return missionCards.snapshot().size > 0
}

/** A Mission suggestion is open: returns its release, which is safe to call more than once. */
export function markMissionSuggestionOpen(): () => void {
  return missionCards.mark(true)
}

/** Whether a Mission suggestion is open, re-read on every change; false on the server. */
export function useMissionSuggestionOpen(): boolean {
  return usePresence(missionCards).size > 0
}

/** The suggestion `id` is pinned above the composer: returns its release. */
export function markSuggestionPinned(id: string): () => void {
  return pinnedSuggestions.mark(id)
}

/** The ids of the suggestions pinned now (at most the one above the open chat's composer). */
export function usePinnedSuggestionIds(): readonly string[] {
  return Array.from(usePresence(pinnedSuggestions).values())
}
