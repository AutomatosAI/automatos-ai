'use client'

/**
 * PRD-268 §6.5.1, §6.12.4 (US-W01a) — the Command Center's Suggested list, beside Auto's Read.
 *
 * Every open suggestion (hooks/use-suggestions.ts: the same list the thread's cards and the pinned line
 * read) as a SuggestionCard: its title and detail in Auto's voice, the evidence it cites as links that
 * open in the app, what Start does, "about N credits" once that is known, Start, and Hide with the reason
 * menu. Nothing renders while there is none.
 *
 * The view keeps one ink primary: Start is the primary on the card the bell opened, else on the first,
 * and quiet on the rest. A card just started stays in its place for "Started. Auto is on it.", and the
 * bell's `?suggestion=<id>` scrolls to its card, outlines it and focuses it (use-suggested-list.ts). Start's ticket
 * goes on the Board under "From suggestions"; nothing here navigates by itself.
 */
import { useSuggestions } from '@/hooks/use-suggestions'
import { cn } from '@/lib/utils'
import type { Suggestion } from '@/lib/proactive/types'
import { Eyebrow } from './card-parts'
import { EvidenceLinks } from './evidence-links'
import { SuggestionCard } from './suggestion-card'
import { useFocusedSuggestion, useKeptStarted, useScrollOnce, withKept } from './use-suggested-list'

export const SUGGESTED_LABEL = 'Suggested'
/** The outline on the card the bell opened. */
export const FOCUSED_CLASS = 'ring-2 ring-ring ring-offset-2 ring-offset-background'

const NO_SUGGESTIONS: readonly Suggestion[] = Object.freeze([])

export function SuggestedList({ className }: { readonly className?: string }) {
  const { data } = useSuggestions()
  const { kept, keep } = useKeptStarted()
  const focusId = useFocusedSuggestion()
  const scrollTo = useScrollOnce(focusId)
  const shown = withKept(data ?? NO_SUGGESTIONS, kept)
  if (shown.length === 0) return null
  const primaryId = shown.some((suggestion) => suggestion.id === focusId) ? focusId : shown[0].id

  return (
    <section aria-label={SUGGESTED_LABEL} className={cn('space-y-3', className)}>
      <Eyebrow>{SUGGESTED_LABEL}</Eyebrow>
      <ul className="space-y-3">
        {shown.map((suggestion, index) => {
          const focused = suggestion.id === focusId
          return (
            <li
              key={suggestion.id}
              ref={focused ? scrollTo : undefined}
              tabIndex={focused ? -1 : undefined}
              data-focused={focused ? 'true' : undefined}
              className="outline-none"
            >
              <SuggestionCard
                suggestion={suggestion}
                primary={suggestion.id === primaryId}
                onStarted={() => keep(suggestion, index)}
                className={focused ? FOCUSED_CLASS : undefined}
              >
                <EvidenceLinks evidence={suggestion.evidence} />
              </SuggestionCard>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
