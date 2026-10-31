'use client'

/**
 * PRD-268 §6.12.3 (US-C21c) — where a suggestion card stands: open, starting, started (the in-place
 * "Started. Auto is on it." for STARTED_MS, then settled), hidden, refused or expired.
 *
 * Start and Hide go through the host's hooks (hooks/use-suggestions.ts), one mutation each per card.
 * The live status from the server wins once it says the suggestion was decided; a decision made on
 * this card shows at once, before the list is asked again. A refusal keeps the card where it is with
 * one line: the server's own sentence when it answered one (a proactive route's `{code, detail}`, or
 * any refusal below 500), else a line of the card's, so a transport or server-error message never
 * reaches the person. Pressing Start or Hide clears the other's old refusal.
 */
import { useEffect, useState } from 'react'
import { useHideSuggestion, useStartSuggestion } from '@/hooks/use-suggestions'
import { refusalOf } from '@/lib/proactive/wire'
import type { HideReason, StartAnswer, Suggestion, SuggestionVia } from '@/lib/proactive/types'

/** How long "Started. Auto is on it." shows in the card's place. */
export const STARTED_MS = 4_000
export const STARTED_LINE = 'Started. Auto is on it.'
/** The refused line when the server answered no sentence of its own (offline, a server or gateway error). */
export const REFUSED_FALLBACK = "Couldn't reach Automatos. Try again in a moment."
/** From here up, a status's words are the server's failure, not a refusal meant for the person. */
const SERVER_ERROR_STATUS = 500

export type SuggestionPhase = 'open' | 'starting' | 'just_started' | 'started' | 'hidden' | 'expired'

interface PhaseFacts {
  readonly status: Suggestion['status'] | undefined
  readonly starting: boolean
  readonly justStarted: boolean
  readonly startedHere: boolean
  readonly hiddenHere: boolean
}

/** The card's state from the live status and what was decided on this card. */
export function phaseOf({ status, starting, justStarted, startedHere, hiddenHere }: PhaseFacts): SuggestionPhase {
  if (justStarted) return 'just_started'
  if (startedHere || status === 'started') return 'started'
  if (hiddenHere || status === 'hidden') return 'hidden'
  if (status === 'expired') return 'expired'
  return starting ? 'starting' : 'open'
}

/** The one line a refused Start or Hide shows. */
export function refusedLine(error: unknown): string | null {
  const refusal = refusalOf(error)
  if (refusal === null) return null
  const answered = refusal.code !== null || (refusal.status !== null && refusal.status < SERVER_ERROR_STATUS)
  return answered && refusal.detail ? refusal.detail : REFUSED_FALLBACK
}

export interface SuggestionCardState {
  readonly phase: SuggestionPhase
  readonly refused: string | null
  /** Start or Hide is on its way: both are held until the server answers. */
  readonly busy: boolean
  readonly start: () => void
  readonly hide: (reason?: HideReason) => void
}

export function useSuggestionCard(
  suggestion: Suggestion,
  via: SuggestionVia,
  onStarted?: (answer: StartAnswer) => void,
): SuggestionCardState {
  const starter = useStartSuggestion()
  const hider = useHideSuggestion()
  const [startedHere, setStartedHere] = useState(false)
  const [justStarted, setJustStarted] = useState(false)
  const [hiddenHere, setHiddenHere] = useState(false)

  useEffect(() => {
    if (!justStarted) return undefined
    const timer = setTimeout(() => setJustStarted(false), STARTED_MS)
    return () => clearTimeout(timer)
  }, [justStarted])

  const onStartedHere = (answer: StartAnswer) => {
    setStartedHere(true)
    setJustStarted(true)
    onStarted?.(answer)
  }
  const start = () => {
    hider.reset()
    starter.mutate({ id: suggestion.id, via }, { onSuccess: onStartedHere })
  }
  const hide = (reason?: HideReason) => {
    starter.reset()
    hider.mutate({ id: suggestion.id, via, ...(reason ? { reason } : {}) }, { onSuccess: () => setHiddenHere(true) })
  }

  const facts = { status: suggestion.status, starting: starter.isLoading, justStarted, startedHere, hiddenHere }
  const failure = starter.isError ? starter.error : hider.isError ? hider.error : null
  return {
    phase: phaseOf(facts),
    refused: refusedLine(failure),
    busy: starter.isLoading || hider.isLoading,
    start,
    hide,
  }
}
