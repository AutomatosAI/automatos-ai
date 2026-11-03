'use client'

/**
 * PRD-268 §6.11.5, §6.12.4, K15, K18, K19 (US-W06) — the consent line above the composer, once.
 *
 * composer-with-suggestion.tsx draws this above the chat's composer, in any chat, only while the server offers
 * calibration and the person hasn't answered it (GET /api/auto/calibration's `available` and `offer`). It says the
 * server's consent line, in the person's language (K18), then Go ahead (the view's ink primary: consent true, with
 * this chat's id and the page's locale) and Not now (consent false: nothing is read, and it isn't offered again).
 * Only one is sent: while it is on its way both are held, and once the server has it the line goes. A refused answer
 * keeps the line, with the server's sentence (else the cards' own). The server posts the calibration into this chat
 * when it is the person's own, else into their Auto thread (a new chat the server doesn't have yet): then one line
 * says so, with a link to it. While the consent line shows, the pinned suggestion steps aside, so one line, and one
 * ink primary, sits above the composer. The consent line is the platform's words, not Auto's: no data-ai-generated.
 */
import Link from 'next/link'
import { useRef, useState } from 'react'
import { useCalibrationConsent, useCalibrationView } from '@/hooks/use-calibration'
import type { CalibrationView } from '@/lib/proactive/calibration'
import { chatHref } from '@/lib/ticket-links'
import { cn } from '@/lib/utils'
import { ENTRANCE_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS, TEXT_BUTTON_CLASS } from './card-parts'
import { RefusedLine } from './suggestion-parts'
import { refusedLine } from './use-suggestion-card'

export const GO_AHEAD = 'Go ahead'
export const NOT_NOW = 'Not now'
export const POSTED_ELSEWHERE = 'Auto answered in your Auto thread.'
export const OPEN_IT = 'Open it'
const LINE_CLASS = 'flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card px-3 py-2'
const LOG_PREFIX = '[proactive]'

/** Where a Go ahead posted the calibration, from which chat. */
interface Posted {
  readonly from: string
  readonly to: string
}

/** True when the server offers calibration and the person hasn't answered it, with a consent line to say. */
function isOffered(view: CalibrationView | undefined): view is CalibrationView {
  return view?.available === true && view.offer && view.consent_line !== ''
}

/** True while the consent line shows. */
export function useCalibrationOffered(): boolean {
  return isOffered(useCalibrationView().data)
}

function PostedElsewhere({ chatId }: { readonly chatId: string }) {
  return (
    <p role="status" data-testid="calibration-posted" className={cn(LINE_CLASS, 'text-sm text-muted-foreground')}>
      {POSTED_ELSEWHERE}
      <Link href={chatHref(chatId)} className={TEXT_BUTTON_CLASS}>
        {OPEN_IT}
      </Link>
    </p>
  )
}

/** The consent line for chat `chatId` while it is offered; after Go ahead, where Auto answered when it isn't here. */
export function CalibrationOffer({ chatId }: { readonly chatId: string }) {
  const { data } = useCalibrationView()
  const consent = useCalibrationConsent()
  const [posted, setPosted] = useState<Posted | null>(null)
  // Held from the press itself, before the mutation's state reaches a render: one answer is sent.
  const sending = useRef(false)
  if (posted !== null && posted.from === chatId && posted.to !== chatId) return <PostedElsewhere chatId={posted.to} />
  if (!isOffered(data)) return null
  const answer = (yes: boolean) => {
    if (sending.current) return
    sending.current = true
    consent
      .mutateAsync({ consent: yes, chatId })
      .then(
        (answered) => setPosted(answered.chat_id === null ? null : { from: chatId, to: answered.chat_id }),
        // The line stays, saying why (consent.error).
        (error: unknown) => console.warn(LOG_PREFIX, 'the consent answer was not taken', error),
      )
      .finally(() => {
        sending.current = false
      })
  }
  return (
    <div role="group" aria-label={data.consent_line} data-testid="calibration-offer" className={cn(LINE_CLASS, ENTRANCE_CLASS)}>
      <p className="min-w-0 flex-1 text-sm text-foreground">{data.consent_line}</p>
      <div className="flex shrink-0 items-center gap-2">
        <button type="button" aria-busy={consent.isLoading} disabled={consent.isLoading} onClick={() => answer(true)} className={PRIMARY_BUTTON_CLASS}>
          {GO_AHEAD}
        </button>
        <button type="button" disabled={consent.isLoading} onClick={() => answer(false)} className={QUIET_BUTTON_CLASS}>
          {NOT_NOW}
        </button>
      </div>
      <RefusedLine text={refusedLine(consent.error)} />
    </div>
  )
}
