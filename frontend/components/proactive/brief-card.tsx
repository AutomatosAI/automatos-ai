'use client'

/**
 * PRD-268 §6.4, §6.11.2, §6.12.3, §6.12.4, K17 (US-C21c) — the day's brief as a card.
 *
 * It shows the date; the greeting, in Newsreader serif (the only serif in the proactive
 * components); Auto's Read (the prose) inside the card, so a surface that draws the card doesn't
 * draw the message's text part again; the five sections (brief-sections.tsx) with their title and
 * why rows, row actions as text, "Show N more" and the empty sentence; the tiny "Auto is an AI"
 * caption once; Listen, the host's BriefVoice (US-C21d), in its slot; and thumbs up and down, which
 * post the existing digest feedback route (POST /api/activity/digest/feedback) with the brief's
 * state_hash. Auto's words (the greeting, the prose, the rows) are marked data-ai-generated and the
 * controls sit outside them. Nothing here plays or navigates by itself: the surface passes what a
 * row's action and its chevron do.
 */
import { useState, type ReactNode } from 'react'
import { ThumbsDown, ThumbsUp } from 'lucide-react'
import { useSubmitDigestFeedback } from '@/hooks/use-digest-api'
import type { Brief } from '@/lib/proactive/types'
import { cn } from '@/lib/utils'
import { AiCaption, CARD_CLASS, ENTRANCE_CLASS, Eyebrow } from './card-parts'
import { BriefSections, type BriefSectionsProps } from './brief-sections'

/** The greeting's face: Newsreader, in Classic as in Studio. */
const GREETING_CLASS = 'm-0 text-2xl font-medium leading-tight text-foreground [font-family:var(--font-newsreader),ui-serif,serif]'
const THUMB_CLASS =
  'rounded-md p-1 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-60'

type Rating = 1 | -1

export interface BriefCardProps extends Omit<BriefSectionsProps, 'brief'> {
  readonly brief: Brief
  /** Listen: the host's BriefVoice. It never plays until pressed. */
  readonly listen?: ReactNode
  readonly className?: string
}

/** The brief's local date (YYYY-MM-DD) as the browser says a day; the date as it is when it isn't one. */
export function briefDateLabel(date: string): string {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date)
  if (!parts) return date
  const day = new Date(Number(parts[1]), Number(parts[2]) - 1, Number(parts[3]))
  if (Number.isNaN(day.getTime())) return date
  return day.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })
}

/** Thumbs up and down on the brief, keyed by its state_hash; one rating per card, kept unless the post fails. */
function BriefThumbs({ stateHash }: { readonly stateHash: string }) {
  const submit = useSubmitDigestFeedback()
  const [rated, setRated] = useState<Rating | null>(null)
  const rate = (rating: Rating) => {
    if (rated !== null || !stateHash) return
    setRated(rating)
    submit.mutate({ state_hash: stateHash, rating }, { onError: () => setRated(null) })
  }
  const thumb = (rating: Rating, label: string, icon: ReactNode) => (
    <button
      type="button"
      aria-label={label}
      aria-pressed={rated === rating}
      disabled={rated !== null || !stateHash}
      onClick={() => rate(rating)}
      className={cn(THUMB_CLASS, rated === rating ? 'text-foreground' : 'text-muted-foreground')}
    >
      {icon}
    </button>
  )
  return (
    <div role="group" aria-label="Rate this brief" className="flex items-center gap-1">
      {thumb(1, 'Helpful', <ThumbsUp aria-hidden="true" className="h-3.5 w-3.5" />)}
      {thumb(-1, 'Not helpful', <ThumbsDown aria-hidden="true" className="h-3.5 w-3.5" />)}
    </div>
  )
}

/** The day's brief: date, greeting, Auto's Read, the sections, Listen and thumbs. */
export function BriefCard({ brief, listen, className, ...sections }: BriefCardProps) {
  return (
    <article
      aria-label={brief.greeting}
      data-brief-date={brief.date}
      className={cn(CARD_CLASS, ENTRANCE_CLASS, 'space-y-4', className)}
    >
      <header className="space-y-1">
        <Eyebrow>
          <time dateTime={brief.date}>{briefDateLabel(brief.date)}</time>
        </Eyebrow>
        <h2 data-studio-serif="" data-ai-generated="true" className={GREETING_CLASS}>
          {brief.greeting}
        </h2>
      </header>
      {brief.prose ? (
        <p data-ai-generated="true" data-brief-prose="" className="whitespace-pre-line text-sm leading-relaxed text-muted-foreground">
          {brief.prose}
        </p>
      ) : null}
      <BriefSections brief={brief} {...sections} />
      <footer className="flex flex-wrap items-center justify-between gap-2">
        <AiCaption />
        <div className="flex items-center gap-2">
          {listen ? <div data-brief-listen="">{listen}</div> : null}
          <BriefThumbs stateHash={brief.state_hash} />
        </div>
      </footer>
    </article>
  )
}
