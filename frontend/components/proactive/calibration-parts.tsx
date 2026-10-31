'use client'

/**
 * PRD-268 §6.11.5, §6.12.3 (US-C21d) — the calibration card's parts: one line Auto thinks it knows,
 * with what it read it from as a count and its Reject, and the three offers.
 *
 * A line shows where it came from as a count of sources by kind ("from 6 Deliverables"), never as a
 * percentage: a confidence number would be one the model made up, shown as if measured. A rejected
 * line stays, struck through, with "I'll forget that". The lines and the offers are Auto's words
 * (data-ai-generated); Reject sits outside them and is named by its line.
 */
import { useId } from 'react'
import type { CalibrationLine, CalibrationSource } from '@/lib/proactive/types'
import { TEXT_BUTTON_CLASS } from './card-parts'

export const REJECT = 'Reject'
export const FORGET_THAT = "I'll forget that"
export const OFFERS_LABEL = 'What Auto can do'

/** A source kind's noun, singular and plural. */
const SOURCE_NOUNS: Readonly<Record<CalibrationSource['type'], readonly [string, string]>> = Object.freeze({
  deliverable: ['Deliverable', 'Deliverables'],
  conversation: ['Conversation', 'Conversations'],
  document: ['document', 'documents'],
  skill: ['skill', 'skills'],
})

function nounOf(type: string, count: number): string | null {
  if (!Object.prototype.hasOwnProperty.call(SOURCE_NOUNS, type)) return null
  const [one, many] = SOURCE_NOUNS[type as CalibrationSource['type']]
  return `${count} ${count === 1 ? one : many}`
}

/** "from 6 Deliverables and 2 Conversations": each kind counted, in the order it first appears; null for none. */
export function sourceCountOf(sources: readonly CalibrationSource[]): string | null {
  const counts = sources.reduce<ReadonlyMap<string, number>>(
    (seen, source) => new Map(seen).set(source.type, (seen.get(source.type) ?? 0) + 1),
    new Map(),
  )
  const parts = [...counts].map(([type, count]) => nounOf(type, count)).filter((part): part is string => part !== null)
  if (parts.length === 0) return null
  const last = parts[parts.length - 1]
  return `from ${parts.length === 1 ? last : `${parts.slice(0, -1).join(', ')} and ${last}`}`
}

export interface CalibrationLineItemProps {
  readonly line: CalibrationLine
  readonly rejected: boolean
  /** Reject this line; without it Reject is not offered. */
  readonly onReject?: (line: CalibrationLine) => void
  readonly busy: boolean
}

/** One line, its source count, and Reject; struck through with "I'll forget that" once rejected. */
export function CalibrationLineItem({ line, rejected, onReject, busy }: CalibrationLineItemProps) {
  const count = sourceCountOf(line.sources)
  return (
    <li data-line-id={line.id} data-state={rejected ? 'rejected' : 'open'} className="flex flex-wrap items-start justify-between gap-2 py-2">
      <div className="min-w-0 space-y-0.5">
        <p data-ai-generated="true" className={rejected ? 'text-sm text-muted-foreground line-through' : 'text-sm text-foreground'}>
          {line.text}
        </p>
        {count ? <p className="text-xs text-muted-foreground">{count}</p> : null}
      </div>
      {rejected ? (
        <span role="status" className="text-xs text-muted-foreground">
          {FORGET_THAT}
        </span>
      ) : onReject ? (
        <button type="button" aria-label={`${REJECT}: ${line.text}`} disabled={busy} onClick={() => onReject(line)} className={TEXT_BUTTON_CLASS}>
          {REJECT}
        </button>
      ) : null}
    </li>
  )
}

/** The three things Auto offers to do, as Auto's words. */
export function CalibrationOffers({ offers }: { readonly offers: readonly string[] }) {
  const labelId = useId()
  if (offers.length === 0) return null
  return (
    <div className="space-y-1">
      <p id={labelId} className="text-xs font-medium text-muted-foreground">
        {OFFERS_LABEL}
      </p>
      <ul aria-labelledby={labelId} className="list-disc space-y-1 pl-5 text-sm text-foreground">
        {offers.map((offer, index) => (
          <li key={`${index}:${offer}`} data-ai-generated="true">
            {offer}
          </li>
        ))}
      </ul>
    </div>
  )
}
