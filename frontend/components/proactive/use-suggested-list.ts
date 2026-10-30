'use client'

/**
 * PRD-268 §6.5.1 (US-W01a) — what the Command Center's Suggested list keeps between the list's answers.
 *
 * - A card just started stays where it was for STARTED_MS, so "Started. Auto is on it." shows in its place:
 *   Start asks the open list again, and a started suggestion is no longer in it.
 * - The bell's suggestion_ready row opens `?suggestion=<id>` (lib/ticket-links.ts suggestionHref): the list
 *   scrolls to that card once, outlines it and moves focus to it. An id that isn't a suggestion's is ignored.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { isProactiveId } from '@/lib/proactive/wire'
import { SUGGESTION_PARAM } from '@/lib/ticket-links'
import type { Suggestion } from '@/lib/proactive/types'
import { STARTED_MS } from './use-suggestion-card'

/**
 * How long the list keeps a card started here: a millisecond under the card's own STARTED_MS, so the list lets
 * it go before the card settles into its one-line "Started" and that line never flashes.
 */
export const KEPT_MS = STARTED_MS - 1

/** A card started here: where it stood in the list, and until when it stays. */
export interface KeptCard {
  readonly suggestion: Suggestion
  readonly index: number
  readonly until: number
}

/** The open list, with each card just started back in its place. */
export function withKept(open: readonly Suggestion[], kept: readonly KeptCard[]): readonly Suggestion[] {
  const listed = new Set(open.map((suggestion) => suggestion.id))
  return kept
    .filter((card) => !listed.has(card.suggestion.id))
    .slice()
    .sort((a, b) => a.index - b.index)
    .reduce<readonly Suggestion[]>((list, card) => [...list.slice(0, card.index), card.suggestion, ...list.slice(card.index)], open)
}

/** The cards started here, each kept for KEPT_MS from its Start. */
export function useKeptStarted() {
  const [kept, setKept] = useState<readonly KeptCard[]>([])

  useEffect(() => {
    if (kept.length === 0) return undefined
    const soonest = Math.min(...kept.map((card) => card.until))
    const timer = setTimeout(() => setKept((cards) => cards.filter((card) => card.until > Date.now())), Math.max(0, soonest - Date.now()))
    return () => clearTimeout(timer)
  }, [kept])

  const keep = useCallback((suggestion: Suggestion, index: number) => {
    const card = { suggestion, index, until: Date.now() + KEPT_MS }
    setKept((cards) => [...cards.filter((other) => other.suggestion.id !== suggestion.id), card])
  }, [])

  return { kept, keep }
}

/** The suggestion the page was opened at (`?suggestion=<id>`), or null. */
export function useFocusedSuggestion(): string | null {
  const value = useSearchParams()?.get(SUGGESTION_PARAM) ?? null
  return isProactiveId(value) ? value : null
}

function reducedMotion(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/** A ref for the focused card: once per focused id, it scrolls into view and takes focus (it has tabIndex -1). */
export function useScrollOnce(focusId: string | null): (node: HTMLElement | null) => void {
  const scrolled = useRef<string | null>(null)
  return useCallback(
    (node: HTMLElement | null) => {
      if (!node || !focusId || scrolled.current === focusId) return
      scrolled.current = focusId
      if (typeof node.scrollIntoView === 'function') {
        node.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' })
      }
      node.focus({ preventScroll: true })
    },
    [focusId],
  )
}
