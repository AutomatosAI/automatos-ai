'use client'

import { useCallback, useEffect, useRef } from 'react'

import type { Deliverable } from '@/hooks/use-deliverables-api'
import { useStartChat } from '@/hooks/use-start-chat'
import { refocus } from '@/lib/play/focus'
import { emptyMessage, type ShelfFilter, type ShelfView } from '@/lib/play/shelf'

import { ShelfCard } from './shelf-card'
import cardStyles from './shelf-card.module.css'
import styles from './shelf.module.css'

/**
 * PRD-260 §4 — the shelf under the chips, ported from Automatos Play's
 * components/deliverables/ShelfBody.tsx: loading, the error state (a failed
 * request, or the list's success: false — never "nothing here yet"), the
 * friendly empty state, or the grid with Load more. A later page that failed
 * keeps the grid and says so, with a way to ask again. When the last Load more
 * goes with the page it asked for, focus moves to that page's first card, never
 * to <body> (P260-FX-22, M20).
 */

export const LOADING_SHELF = 'Loading your shelf…'
export const SHELF_FAILED = "Your shelf didn't load. Try again in a moment."
export const MORE_FAILED = "More Deliverables didn't load."
export const START_A_CHAT = 'Start a chat'
export const LOAD_MORE = 'Load more'
export const LOADING_MORE = 'Loading more…'
export const TRY_AGAIN = 'Try again'
export const DELIVERABLES_LIST_LABEL = 'Deliverables'

export interface ShelfPaging {
  /** The list has another page to ask for (the hook's hasNextPage). */
  readonly hasMore: boolean
  readonly loadingMore: boolean
  /** A later page didn't load: the grid stays. */
  readonly moreFailed: boolean
  readonly loadMore: () => void
  /** Ask for every loaded page again (the hook's refetch). */
  readonly retry: () => void
}

export interface ShelfBodyProps {
  readonly view: ShelfView
  readonly filter: ShelfFilter
  readonly paging: ShelfPaging
  readonly now: Date
}

export function ShelfBody({ view, filter, paging, now }: ShelfBodyProps) {
  if (view.status === 'loading') {
    return (
      <p className={styles.status} role="status">
        {LOADING_SHELF}
      </p>
    )
  }
  if (view.status === 'failed') return <Failed message={SHELF_FAILED} retry={paging.retry} />
  if (view.items.length === 0) return <Empty filter={filter} />
  return <Grid items={view.items} paging={paging} now={now} />
}

/**
 * Load more remembers where its page starts. Once that page is in and no Load
 * more is left (the button that had focus is gone), the first card it brought
 * takes focus; a Load more that stays keeps it. Only when focus was dropped:
 * a chip picked meanwhile keeps it.
 */
function useFocusFirstNewCard(count: number, paging: ShelfPaging) {
  const list = useRef<HTMLUListElement>(null)
  const firstNew = useRef<number | null>(null)
  const { loadMore: askForMore, loadingMore } = paging
  const showsMore = paging.hasMore || paging.moreFailed
  const loadMore = useCallback(() => {
    firstNew.current = count
    askForMore()
  }, [count, askForMore])
  useEffect(() => {
    const index = firstNew.current
    if (index === null || loadingMore) return
    firstNew.current = null
    if (showsMore) return
    refocus(list.current?.children.item(index)?.querySelector<HTMLElement>('a'))
  }, [count, loadingMore, showsMore])
  return { list, loadMore }
}

interface GridProps {
  readonly items: readonly Deliverable[]
  readonly paging: ShelfPaging
  readonly now: Date
}

function Grid({ items, paging, now }: GridProps) {
  const { list, loadMore } = useFocusFirstNewCard(items.length, paging)
  return (
    <>
      <ul ref={list} className={cardStyles.grid} aria-label={DELIVERABLES_LIST_LABEL}>
        {items.map((deliverable, index) => (
          <ShelfCard key={deliverable.id} deliverable={deliverable} now={now} newest={index === 0} />
        ))}
      </ul>
      <More paging={{ ...paging, loadMore }} />
    </>
  )
}

function Empty({ filter }: { readonly filter: ShelfFilter }) {
  const { startChat, ready } = useStartChat()
  return (
    <div className={styles.empty}>
      <p className={styles.status}>{emptyMessage(filter)}</p>
      <button type="button" className={styles.button} onClick={startChat} disabled={!ready}>
        {START_A_CHAT}
      </button>
    </div>
  )
}

/** Load more while there is a next page; after a failed one, the error and a way to ask again. */
function More({ paging }: { readonly paging: ShelfPaging }) {
  if (!paging.hasMore && !paging.moreFailed) return null
  const action = paging.hasMore ? paging.loadMore : paging.retry
  const label = paging.loadingMore ? LOADING_MORE : paging.hasMore ? LOAD_MORE : TRY_AGAIN
  return (
    <div className={styles.more}>
      {paging.moreFailed && (
        <p className={styles.error} role="alert">
          <span aria-hidden="true">✕</span> {MORE_FAILED}
        </p>
      )}
      <button type="button" className={styles.moreButton} onClick={action} disabled={paging.loadingMore}>
        {label}
      </button>
    </div>
  )
}

function Failed({ message, retry }: { readonly message: string; readonly retry: () => void }) {
  return (
    <div className={styles.failed} role="alert">
      <p className={styles.error}>
        <span aria-hidden="true">✕</span> {message}
      </p>
      <button type="button" className={styles.button} onClick={retry}>
        {TRY_AGAIN}
      </button>
    </div>
  )
}
