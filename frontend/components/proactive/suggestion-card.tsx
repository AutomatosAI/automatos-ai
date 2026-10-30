'use client'

/**
 * PRD-268 §6.2.5, §6.12.3, §6.12.4, K17 (US-C21c) — a suggestion: the card, and its pinned one-line
 * variant (above the composer in the Auto thread, US-W01b).
 *
 * The card shows "Auto suggests" (studio-eyebrow), the suggestion's title and detail in Auto's voice
 * (marked data-ai-generated, drawn as text: they are model-written), what a surface adds under them
 * (the Command Center's evidence links, US-W01a), what Start does, Start (the one ink primary, with
 * "about N credits" once that is known) and Hide with its reason menu, and the tiny "Auto is an AI"
 * caption once, outside the fence. Its states (use-suggestion-card.ts): open; starting; started
 * ("Started. Auto is on it." in place for 4 s, then one line); hidden and expired (one line); refused
 * (one line, and the card stays put). The pinned line keeps Start and Hide on one line and leaves
 * once the suggestion is decided, so the next open one can take its place.
 *
 * Start goes through the host's route only (hooks/use-suggestions.ts); `onStarted` hands the surface
 * what Start made (the ticket, or the chat to open), and nothing here navigates by itself.
 *
 * The card's states live in the card. A Start that succeeds asks the open list again, and a started
 * suggestion is no longer in it, so a surface that draws one card per open suggestion keeps a card it
 * has just started mounted for STARTED_MS (from `onStarted`); otherwise "Started. Auto is on it." goes
 * with the card on the list's next answer.
 */
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'
import type { StartAnswer, Suggestion, SuggestionVia } from '@/lib/proactive/types'
import { AiCaption, CARD_CLASS, ENTRANCE_CLASS, Eyebrow } from './card-parts'
import { ClosedLine, RefusedLine, SuggestionActions, startDoesOf, type ClosedPhase } from './suggestion-parts'
import { STARTED_LINE, useSuggestionCard, type SuggestionCardState, type SuggestionPhase } from './use-suggestion-card'

export const SUGGESTS_LABEL = 'Auto suggests'

export interface SuggestionCardProps {
  readonly suggestion: Suggestion
  /** The card (the default), or the one-line variant pinned above the composer. */
  readonly variant?: 'card' | 'pinned'
  /** Where the person decided; the web by default. */
  readonly via?: SuggestionVia
  /** Start was accepted: what it made, for the surface to open. */
  readonly onStarted?: (answer: StartAnswer) => void
  /** The pinned line's title opens the full card. */
  readonly onOpen?: () => void
  /** What a surface draws under the text, such as the evidence links. */
  readonly children?: ReactNode
  readonly className?: string
}

const CLOSED: ReadonlySet<SuggestionPhase> = new Set<SuggestionPhase>(['started', 'hidden', 'expired'])

interface ShownProps {
  readonly suggestion: Suggestion
  readonly card: SuggestionCardState
  readonly onOpen?: () => void
  readonly children?: ReactNode
  readonly className?: string
}

function StartedLine() {
  return (
    <p role="status" className="min-w-0 flex-1 text-sm text-foreground">
      {STARTED_LINE}
    </p>
  )
}

function Actions({ suggestion, card }: Pick<ShownProps, 'suggestion' | 'card'>) {
  return (
    <>
      <SuggestionActions suggestion={suggestion} starting={card.phase === 'starting'} busy={card.busy} onStart={card.start} onHide={card.hide} />
      <RefusedLine text={card.refused} />
    </>
  )
}

function FullCard({ suggestion, card, children, className }: ShownProps) {
  const startDoes = startDoesOf(suggestion.proposed_action)
  return (
    <article
      aria-label={`${SUGGESTS_LABEL}: ${suggestion.title}`}
      data-suggestion-id={suggestion.id}
      data-state={card.phase}
      className={cn(CARD_CLASS, ENTRANCE_CLASS, 'space-y-3', className)}
    >
      <Eyebrow>{SUGGESTS_LABEL}</Eyebrow>
      {card.phase === 'just_started' ? (
        <StartedLine />
      ) : (
        <>
          <div data-ai-generated="true" className="space-y-1">
            <p className="text-sm font-medium text-foreground">{suggestion.title}</p>
            {suggestion.detail ? <p className="text-sm text-muted-foreground">{suggestion.detail}</p> : null}
          </div>
          {children}
          {startDoes ? <p className="text-xs text-muted-foreground">{startDoes}</p> : null}
          <Actions suggestion={suggestion} card={card} />
        </>
      )}
      <AiCaption className="block" />
    </article>
  )
}

function PinnedTitle({ title, onOpen }: { readonly title: string; readonly onOpen?: () => void }) {
  const words = (
    <span data-ai-generated="true" className="truncate">
      {title}
    </span>
  )
  if (!onOpen) return <span className="flex min-w-0 flex-1 text-sm font-medium text-foreground">{words}</span>
  return (
    <button type="button" aria-label={`Open: ${title}`} onClick={onOpen} className="flex min-w-0 flex-1 text-left text-sm font-medium text-foreground hover:underline">
      {words}
    </button>
  )
}

function PinnedLine({ suggestion, card, onOpen, className }: ShownProps) {
  return (
    <div
      role="group"
      aria-label={`${SUGGESTS_LABEL}: ${suggestion.title}`}
      data-suggestion-id={suggestion.id}
      data-state={card.phase}
      data-variant="pinned"
      className={cn('flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card px-3 py-2', ENTRANCE_CLASS, className)}
    >
      <Eyebrow className="shrink-0">{SUGGESTS_LABEL}</Eyebrow>
      {card.phase === 'just_started' ? (
        <StartedLine />
      ) : (
        <>
          <PinnedTitle title={suggestion.title} onOpen={onOpen} />
          <Actions suggestion={suggestion} card={card} />
        </>
      )}
      <AiCaption className="shrink-0" />
    </div>
  )
}

/** A suggestion with Start and Hide, as a card or as the pinned line. */
export function SuggestionCard({ suggestion, variant = 'card', via = 'web', onStarted, ...shown }: SuggestionCardProps) {
  const card = useSuggestionCard(suggestion, via, onStarted)
  if (CLOSED.has(card.phase)) {
    return variant === 'pinned' ? null : <ClosedLine suggestion={suggestion} phase={card.phase as ClosedPhase} />
  }
  const Shown = variant === 'pinned' ? PinnedLine : FullCard
  return <Shown suggestion={suggestion} card={card} {...shown} />
}
