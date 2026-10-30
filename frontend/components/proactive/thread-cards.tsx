'use client'

/**
 * PRD-268 §6.5.1, §6.12.4, K17 (US-W01b) — the cards an Auto-thread message carries: the day's brief and the
 * suggestions it delivered.
 *
 * `MessageWithReceipts` draws these under the message, outside its data-ai-generated fence, so Start, Hide and the
 * thumbs sit outside it and each card marks Auto's own words. The brief card draws from its `brief@1` part. A
 * suggestion card draws from the open list (hooks/use-suggestions.ts) by the id its `suggestion@1` part names, so its
 * state is the live one and survives a reload: while the suggestion is open it has Start and Hide; once it is decided
 * (here, on the phone, on the Command Center or by expiry) its card goes and the message's text stays. A card just
 * started here stays in its place for "Started. Auto is on it." (use-suggested-list.ts). Start is never the thread's
 * ink primary: the line pinned above the composer holds it. The suggestion that line shows has its Start and Hide
 * there, so its card here steps aside until the line hides (mission-suggestion-presence.ts).
 */
import type { ReactNode } from 'react'
import { useSuggestions } from '@/hooks/use-suggestions'
import type { Brief, Suggestion } from '@/lib/proactive/types'
import { BriefCard } from './brief-card'
import { usePinnedSuggestionIds } from './mission-suggestion-presence'
import { SuggestionCard } from './suggestion-card'
import { useKeptStarted, withKept } from './use-suggested-list'

/** The avatar's width and gap (w-8 + space-x-3), so a card lines up with the message's bubble. */
const ALIGNED_CLASS = 'max-w-[92%] space-y-2 pl-11'
const NO_SUGGESTIONS: readonly Suggestion[] = Object.freeze([])

export interface ThreadCardsProps {
  readonly brief: Brief | null
  readonly suggestionIds: readonly string[]
}

function Aligned({ children }: { readonly children: ReactNode }) {
  return (
    <div className={ALIGNED_CLASS} data-testid="thread-cards">
      {children}
    </div>
  )
}

/**
 * The brief, if any, then each named suggestion that is open (or was just started here) and not pinned above the
 * composer; nothing when none shows.
 */
function WithSuggestions({ brief, ids }: { readonly brief: Brief | null; readonly ids: readonly string[] }) {
  const { data } = useSuggestions()
  const { kept, keep } = useKeptStarted()
  const pinned = usePinnedSuggestionIds()
  const shown = withKept(data ?? NO_SUGGESTIONS, kept)
  const cards = ids
    .filter((id) => !pinned.includes(id))
    .map((id) => shown.find((open) => open.id === id))
    .filter((suggestion): suggestion is Suggestion => suggestion !== undefined)
  if (brief === null && cards.length === 0) return null
  return (
    <Aligned>
      {brief !== null ? <BriefCard brief={brief} /> : null}
      {cards.map((suggestion) => (
        <SuggestionCard key={suggestion.id} suggestion={suggestion} primary={false} onStarted={() => keep(suggestion, 0)} />
      ))}
    </Aligned>
  )
}

/** A message's brief and suggestion cards, lined up with its bubble; nothing when it carries neither. */
export function ThreadCards({ brief, suggestionIds }: ThreadCardsProps) {
  if (suggestionIds.length > 0) return <WithSuggestions brief={brief} ids={suggestionIds} />
  if (brief === null) return null
  return (
    <Aligned>
      <BriefCard brief={brief} />
    </Aligned>
  )
}
