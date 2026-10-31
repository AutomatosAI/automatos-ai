'use client'

/**
 * PRD-268 §6.12.3, K17, K23 (US-C21d) — reply chips: two or three short answers under Auto's last
 * message.
 *
 * It renders the chips it is given (the reply-chip tool and its storage are US-C12's): each a
 * non-empty answer of at most 24 characters, the first three that are, and nothing unless at least
 * two are. They show only while their message is Auto's last one in the thread (`latest`), so they
 * are gone once a newer message exists. A tap sends the chip's own text as the person's message
 * (`onReply`), once: the chips are held until the newer message takes them away, or given back when
 * the send fails (onReply's promise rejects); another set of chips starts open. Each chip is named
 * "Reply: <chip>"; its words are Auto's (data-ai-generated) and the chips sit outside the composer's
 * AI-label fence (§12-Q29).
 */
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { ENTRANCE_CLASS } from './card-parts'

/** A chip's longest answer, in characters. */
export const REPLY_CHIP_MAX_CHARS = 24
export const REPLY_CHIPS_MIN = 2
export const REPLY_CHIPS_MAX = 3

const LOG_PREFIX = '[proactive]'
/** Joins a chip set into one key (a chip never holds this character). */
const SET_SEPARATOR = '\u0000'

const CHIP_CLASS =
  'inline-flex max-w-full items-center rounded-xl border border-border bg-card px-3 py-1 text-sm text-foreground ' +
  'transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ' +
  'disabled:pointer-events-none disabled:opacity-60'

export interface ReplyChipsProps {
  /** The answers Auto offered with its message (checked again here: they are model-written). */
  readonly chips: readonly string[]
  /** True while the message is Auto's last one in the thread; a newer message takes the chips away. */
  readonly latest: boolean
  /** Sends the chip's text as the person's message; a rejected promise gives the chips back. */
  readonly onReply: (text: string) => void | Promise<unknown>
  readonly className?: string
}

/** The chips that can be drawn: trimmed, non-empty, at most 24 characters, no repeats, the first three. */
export function replyChipsOf(chips: readonly unknown[]): readonly string[] {
  const usable = chips
    .filter((chip): chip is string => typeof chip === 'string')
    .map((chip) => chip.trim())
    .filter((chip) => chip.length > 0 && [...chip].length <= REPLY_CHIP_MAX_CHARS)
  return [...new Set(usable)].slice(0, REPLY_CHIPS_MAX)
}

/** Two or three answers under Auto's last message; a tap sends one as the person's message. */
export function ReplyChips({ chips, latest, onReply, className }: ReplyChipsProps) {
  const [sentFor, setSentFor] = useState<string | null>(null)
  const shown = replyChipsOf(chips)
  const setKey = shown.join(SET_SEPARATOR)
  const sent = sentFor === setKey
  if (!latest || shown.length < REPLY_CHIPS_MIN) return null
  const giveBack = (error: unknown) => {
    console.warn(LOG_PREFIX, 'the reply was not sent', error)
    setSentFor(null)
  }
  const reply = (chip: string) => {
    if (sent) return
    setSentFor(setKey)
    try {
      Promise.resolve(onReply(chip)).catch(giveBack)
    } catch (error) {
      giveBack(error)
    }
  }
  return (
    <div role="group" aria-label="Replies" data-reply-chips="" className={cn('flex flex-wrap gap-2', ENTRANCE_CLASS, className)}>
      {shown.map((chip) => (
        <button key={chip} type="button" aria-label={`Reply: ${chip}`} disabled={sent} onClick={() => reply(chip)} className={CHIP_CLASS}>
          <span data-ai-generated="true" className="truncate">
            {chip}
          </span>
        </button>
      ))}
    </div>
  )
}
