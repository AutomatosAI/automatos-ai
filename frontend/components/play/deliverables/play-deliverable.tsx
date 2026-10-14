'use client'

import Link from 'next/link'
import { useEffect } from 'react'

import { PageColumns } from '@/components/play/shell/page-columns'
import { type Deliverable, type DeliverableDetailResponse, useDeliverable } from '@/hooks/use-deliverables-api'
import { deliverableFilename } from '@/lib/deliverables/download'
import { contentView, detailDate, detailTitle, madeLine } from '@/lib/play/detail'
import { kindOf } from '@/lib/play/shelf'

import { DeliverableContent } from './deliverable-content'
import styles from './deliverable-detail.module.css'
import { DetailPanel } from './detail-panel'
import { TypeBadge } from './shelf-card'

/**
 * PRD-260 §4, K8 — one Deliverable's page under Play (Ember mock 05), ported
 * from Automatos Play's components/deliverables/{DeliverableScreen,
 * DeliverableView}.tsx onto the core's useDeliverable(id, true)
 * (hooks/use-deliverables-api.ts: GET /api/deliverables/{id}?include_content=true).
 * It loads, then shows the title, type, date and "Made by", the content
 * (deliverable-content.tsx) and the side panel (detail-panel.tsx); or says the
 * Deliverable isn't on the shelf (the backend's 404), or that it couldn't be
 * opened, with Try again. No mascot here: the Deliverable is the hero, and
 * Auto comes back with "Ask Auto about this".
 *
 * The shelf switch (PlayShelf) renders it for the core's own
 * /deliverables?deliverable=<id> (K8), in place of DeliverableDeepLink's sheet.
 */

export const OPENING = 'Opening this Deliverable…'
export const OPEN_FAILED = "Play couldn't open this Deliverable."
export const NOT_ON_SHELF = "This Deliverable isn't on your shelf. It may have been deleted."
export const BACK_TO_SHELF = 'Shelf'
export const TRY_AGAIN = 'Try again'
const SHELF_HREF = '/deliverables'
const NOT_FOUND = 404

export interface PlayDeliverableProps {
  /** A Deliverable id the shelf switch checked against the id pattern. */
  readonly id: string
  /** What "today" is; tests pin it. */
  readonly now?: Date
}

type DetailState =
  | { readonly status: 'loading' }
  | { readonly status: 'failed' }
  | { readonly status: 'missing' }
  | { readonly status: 'ready'; readonly deliverable: Deliverable }

/** What useDeliverable gives the page, read only. */
interface DetailQuery {
  readonly data?: DeliverableDetailResponse
  readonly isError: boolean
  readonly error: unknown
}

function statusOf(error: unknown): number | undefined {
  const status = (error as { status?: unknown } | null)?.status
  return typeof status === 'number' ? status : undefined
}

/** The page's state: a 404 is "not on your shelf", any other failure (success: false included) "couldn't open". */
export function detailState(query: DetailQuery): DetailState {
  if (query.isError) return statusOf(query.error) === NOT_FOUND ? { status: 'missing' } : { status: 'failed' }
  if (!query.data) return { status: 'loading' }
  if (query.data.success === false || !query.data.deliverable) return { status: 'failed' }
  return { status: 'ready', deliverable: query.data.deliverable }
}

/** The backend's own words for a failure go to the log, never the screen. */
function useLogFailure(id: string, query: DetailQuery) {
  const { isError, error } = query
  const detail = query.data?.success === false ? query.data.error : undefined
  useEffect(() => {
    if (isError) console.error(`[deliverable] Deliverable ${id} didn't load:`, error)
    else if (detail) console.error(`[deliverable] Deliverable ${id} answered success: false:`, detail)
  }, [id, isError, error, detail])
}

export function PlayDeliverable({ id, now }: PlayDeliverableProps) {
  const query = useDeliverable(id, true)
  useLogFailure(id, query)
  const retry = () => void query.refetch()
  return (
    <PageColumns>
      <div className={styles.page}>
        <Link href={SHELF_HREF} className={styles.back}>
          <span aria-hidden="true">← </span>
          {BACK_TO_SHELF}
        </Link>
        <DetailBody state={detailState(query)} retry={retry} now={now ?? new Date()} />
      </div>
    </PageColumns>
  )
}

interface DetailBodyProps {
  readonly state: DetailState
  readonly retry: () => void
  readonly now: Date
}

function DetailBody({ state, retry, now }: DetailBodyProps) {
  if (state.status === 'loading') {
    return (
      <p className={styles.note} role="status">
        {OPENING}
      </p>
    )
  }
  if (state.status === 'missing') return <p className={styles.note}>{NOT_ON_SHELF}</p>
  if (state.status === 'failed') {
    return (
      <div className={styles.failed} role="alert">
        <p className={styles.error}>
          <span aria-hidden="true">✕</span> {OPEN_FAILED}
        </p>
        <button type="button" className={styles.primary} onClick={retry}>
          {TRY_AGAIN}
        </button>
      </div>
    )
  }
  return <DeliverableView deliverable={state.deliverable} now={now} />
}

interface DeliverableViewProps {
  readonly deliverable: Deliverable
  readonly now: Date
}

/** A loaded Deliverable: title, type badge, date and "Made by", the content, the side panel. */
function DeliverableView({ deliverable, now }: DeliverableViewProps) {
  const kind = kindOf(deliverable)
  const title = detailTitle(deliverable)
  const date = detailDate(deliverable.created_at, now)
  return (
    <div className={styles.layout}>
      <article className={styles.article} aria-label={title}>
        <header className={styles.header}>
          <h1 className={styles.title}>{title}</h1>
          <p className={styles.meta}>
            <TypeBadge kind={kind} />
            {date && <time dateTime={deliverable.created_at}>{date}</time>}
            {date && <span aria-hidden="true">·</span>}
            <span>{madeLine(deliverable.agent_name)}</span>
          </p>
        </header>
        <DeliverableContent
          view={contentView(deliverable)}
          kind={kind}
          title={title}
          filename={deliverableFilename(deliverable)}
        />
      </article>
      <DetailPanel deliverable={deliverable} kind={kind} title={title} date={date} />
    </div>
  )
}
