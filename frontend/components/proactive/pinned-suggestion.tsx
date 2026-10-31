'use client'

/**
 * PRD-268 §6.5.1, §6.12.3, §6.12.4 (US-W01b) — the newest open suggestion, pinned above the composer in the Auto
 * thread.
 *
 * Only in the Auto thread (the chat's kind is 'auto', use-auto-thread.ts), and only a suggestion addressed to that
 * thread's owner: the open list (hooks/use-suggestions.ts) gives an owner or admin everyone's. It sits on one line
 * (SuggestionCard's pinned variant: "Auto suggests", the title, Start and Hide); its title opens the full card in
 * the same place, and the card folds back to the line on Escape or once focus leaves it. It hides while a turn's
 * Mission suggestion is unanswered (mission-suggestion-presence.ts: one suggestion surface), and draws nothing in any
 * other chat or when no suggestion is open. Started from here, the line says "Started. Auto is on it." in its place
 * before the next open one takes it; hidden, it goes at once. While it shows a suggestion, that suggestion's card in
 * the thread steps aside (mission-suggestion-presence.ts), so its Start and Hide are on screen once.
 */
import { useEffect, useRef, useState, type FocusEvent, type KeyboardEvent } from 'react'
import { useSuggestions } from '@/hooks/use-suggestions'
import type { Suggestion } from '@/lib/proactive/types'
import { markSuggestionPinned, useMissionSuggestionOpen } from './mission-suggestion-presence'
import { SuggestionCard } from './suggestion-card'
import { useAutoThreadOwner } from './use-auto-thread'
import { useKeptStarted, withKept } from './use-suggested-list'

const NO_SUGGESTIONS: readonly Suggestion[] = Object.freeze([])

/** The newest suggestion addressed to `owner` (users.id as text) in a newest-first list, or null. */
export function newestFor(list: readonly Suggestion[], owner: string): Suggestion | null {
  return list.find((suggestion) => suggestion.user_id !== null && String(suggestion.user_id) === owner) ?? null
}

/** The line, or the full card it opened; the same card either way, so Start's and Hide's states carry over. */
function NewestSuggestion({ owner }: { readonly owner: string }) {
  const { data } = useSuggestions()
  const { kept, keep } = useKeptStarted()
  const [openedId, setOpenedId] = useState<string | null>(null)
  const holder = useRef<HTMLDivElement>(null)
  const newest = newestFor(withKept(data ?? NO_SUGGESTIONS, kept), owner)
  const pinnedId = newest?.id ?? null
  useEffect(() => (pinnedId === null ? undefined : markSuggestionPinned(pinnedId)), [pinnedId])
  if (newest === null) return null
  const opened = openedId === newest.id

  const open = () => {
    setOpenedId(newest.id)
    holder.current?.focus()
  }
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!opened || event.key !== 'Escape' || event.defaultPrevented) return
    setOpenedId(null)
    holder.current?.focus()
  }
  const onBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (opened && !event.currentTarget.contains(event.relatedTarget as Node | null)) setOpenedId(null)
  }
  return (
    <div ref={holder} tabIndex={-1} data-testid="pinned-suggestion" className="outline-none" onKeyDown={onKeyDown} onBlur={onBlur}>
      <SuggestionCard
        key={newest.id}
        suggestion={newest}
        variant={opened ? 'card' : 'pinned'}
        onOpen={open}
        onStarted={() => keep(newest, 0)}
      />
    </div>
  )
}

/** The pinned suggestion for the chat `chatId`, or nothing. */
export function PinnedSuggestion({ chatId }: { readonly chatId: string }) {
  const owner = useAutoThreadOwner(chatId)
  const missionOpen = useMissionSuggestionOpen()
  if (owner === null || missionOpen) return null
  return <NewestSuggestion owner={owner} />
}
