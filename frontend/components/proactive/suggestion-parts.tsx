'use client'

/**
 * PRD-268 §6.2.5, §6.12.3, K17 (US-C21c) — the suggestion card's parts: what Start does, its
 * credits, Start and Hide with the reason menu, and the one-line states.
 *
 * Start and Hide are named by the suggestion ("Start: <title>", "Hide: <title>"), on the card and on
 * the pinned line. The words here are the card's own control labels; the suggestion's title and
 * detail are model-written and drawn as text only, never as markup.
 */
import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { HIDE_REASONS, type HideReason, type ProposedAction, type Suggestion } from '@/lib/proactive/types'
import { AiCaption, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from './card-parts'

/** What Start does, per kind of proposed action. */
export const START_DOES: Readonly<Record<ProposedAction['kind'], string>> = Object.freeze({
  board_task: 'Start does: files a ticket on the Board for an agent to work.',
  chat: 'Start does: opens a chat with Auto about this.',
  mission: 'Start does: plans a Mission.',
  playbook_run: 'Start does: runs a Playbook.',
})

/** The reason menu's words. */
export const HIDE_REASON_LABELS: Readonly<Record<HideReason, string>> = Object.freeze({
  not_now: 'Not now',
  not_useful: 'Not useful',
  wrong_agent: 'Wrong agent',
})
export const HIDE_WITHOUT_REASON = 'Just hide'

/** The one-line words of a decided card. */
export const CLOSED_WORDS = Object.freeze({ started: 'Started', hidden: 'Hidden', expired: 'Expired' })
export type ClosedPhase = keyof typeof CLOSED_WORDS

export function startDoesOf(action: ProposedAction | null | undefined): string | null {
  const kind = action?.kind
  return typeof kind === 'string' && Object.prototype.hasOwnProperty.call(START_DOES, kind) ? START_DOES[kind] : null
}

/** "about N credits", on the web only, once US-C15 knows the cost; null until then. */
export function creditsOf(estCredits: number | null | undefined): string | null {
  if (typeof estCredits !== 'number' || !Number.isFinite(estCredits) || estCredits < 0) return null
  const credits = Math.round(estCredits)
  return `about ${credits} credit${credits === 1 ? '' : 's'}`
}

interface HideMenuProps {
  readonly id: string
  readonly busy: boolean
  readonly onHide: (reason?: HideReason) => void
}

/** The reasons, with focus on the first as the menu opens. */
function HideMenu({ id, busy, onHide }: HideMenuProps) {
  const first = useRef<HTMLButtonElement>(null)
  useEffect(() => first.current?.focus(), [])
  return (
    <div id={id} role="group" aria-label="Why hide it?" className="flex basis-full flex-wrap gap-2">
      {HIDE_REASONS.map((reason, index) => (
        <button
          key={reason}
          ref={index === 0 ? first : undefined}
          type="button"
          disabled={busy}
          onClick={() => onHide(reason)}
          className={QUIET_BUTTON_CLASS}
        >
          {HIDE_REASON_LABELS[reason]}
        </button>
      ))}
      <button type="button" disabled={busy} onClick={() => onHide()} className={QUIET_BUTTON_CLASS}>
        {HIDE_WITHOUT_REASON}
      </button>
    </div>
  )
}

export interface SuggestionActionsProps {
  readonly suggestion: Suggestion
  readonly starting: boolean
  readonly busy: boolean
  readonly onStart: () => void
  readonly onHide: (reason?: HideReason) => void
}

/**
 * Start (the view's one ink primary, with its credits when known) and Hide with its reason menu.
 * Escape closes the menu and puts focus back on Hide.
 */
export function SuggestionActions({ suggestion, starting, busy, onStart, onHide }: SuggestionActionsProps) {
  const [menuOpen, setMenuOpen] = useState(false)
  const hideButton = useRef<HTMLButtonElement>(null)
  const menuId = useId()
  const credits = creditsOf(suggestion.est_credits)
  const startName = `Start: ${suggestion.title}${credits ? ` (${credits})` : ''}`
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Escape' || !menuOpen) return
    setMenuOpen(false)
    hideButton.current?.focus()
  }
  return (
    <div className="flex flex-wrap items-center gap-2" onKeyDown={onKeyDown}>
      <button type="button" aria-label={startName} aria-busy={starting} disabled={busy} onClick={onStart} className={PRIMARY_BUTTON_CLASS}>
        Start
        {credits ? <span className="font-normal opacity-80">({credits})</span> : null}
      </button>
      <button
        ref={hideButton}
        type="button"
        aria-label={`Hide: ${suggestion.title}`}
        aria-expanded={menuOpen}
        aria-controls={menuOpen ? menuId : undefined}
        disabled={busy}
        onClick={() => setMenuOpen((open) => !open)}
        className={QUIET_BUTTON_CLASS}
      >
        Hide
      </button>
      {menuOpen ? <HideMenu id={menuId} busy={busy} onHide={onHide} /> : null}
    </div>
  )
}

/** A refused Start or Hide: one line, and the card stays where it is. */
export function RefusedLine({ text }: { readonly text: string | null }) {
  if (!text) return null
  return (
    <p role="alert" className="text-sm text-[hsl(var(--destructive))]">
      {text}
    </p>
  )
}

/** A decided card, on one line: the suggestion's title, what became of it, and the AI caption. */
export function ClosedLine({ suggestion, phase }: { readonly suggestion: Suggestion; readonly phase: ClosedPhase }) {
  return (
    <div data-suggestion-id={suggestion.id} data-state={phase} className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
      <span data-ai-generated="true" className="truncate">
        {suggestion.title}
      </span>
      <span aria-hidden="true">·</span>
      <span>{CLOSED_WORDS[phase]}</span>
      <AiCaption className="shrink-0" />
    </div>
  )
}
