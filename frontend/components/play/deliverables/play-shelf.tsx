'use client'

import Link from 'next/link'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { MainLayout } from '@/components/layout/main-layout'
import { AutoMascot } from '@/components/mascot/auto-mascot'
import { PageColumns } from '@/components/play/shell/page-columns'
import { useWorkspace } from '@/components/workspace-provider'
import { useDeliverables } from '@/hooks/use-deliverables-api'
import { useShelfMascot } from '@/hooks/use-shelf-mascot'
import { EXPLORER_HREF, TAB_LABELS, visibleDeliverableTabs } from '@/lib/deliverables/tabs'
import {
  ALL_FILTER,
  countLine,
  failureDetail,
  SHELF_FILTERS,
  shelfFilters,
  shelfView,
  type ShelfFilter,
} from '@/lib/play/shelf'

import { ShelfBody, type ShelfPaging } from './shelf-body'
import styles from './shelf.module.css'

/**
 * PRD-260 §4, D5 — Play's Deliverables shelf (Ember mock 04), ported from
 * Automatos Play's components/deliverables/ShelfScreen.tsx onto the core's
 * list (useDeliverables, hooks/use-deliverables-api.ts): "Your shelf" with
 * Auto in the header, the type chips (the list's artifact_type), the card
 * grid with the newest in the spotlight, and Load more (the hook's next page:
 * its offset, buildListQuery). A card opens /deliverables?deliverable=<id> (K8).
 *
 * app/deliverables/layout.tsx renders it through PlayRouteSwitch under Play at
 * >= 1024 px on /deliverables (K7); a link into one of the page's other
 * sections (?tab=, a post, a type drill-down) keeps the core's page inside
 * Play's frame (D8), and the shelf links to those sections, so none of them
 * is out of reach under Play.
 */

export const SHELF_TITLE = 'Your shelf'
export const FILTER_GROUP_LABEL = 'Show Deliverables by type'
export const SECTIONS_LABEL = 'More in Deliverables'
const EXPLORER_LABEL = 'Explorer'

/** The shelf in Play's frame: the route's layout shows it instead of the page. */
export function PlayShelf() {
  return (
    <MainLayout fullBleed>
      <ShelfScreen />
    </MainLayout>
  )
}

export interface ShelfScreenProps {
  /** What "today" is; tests pin it. */
  readonly now?: Date
}

/**
 * The list query for one chip, read as the shelf shows it. A Load more that
 * failed is remembered for the chip it was asked under, so another chip starts
 * clean; the backend's own words for a failure go to the log, never the screen.
 */
function useShelfList(filter: ShelfFilter) {
  const filters = useMemo(() => shelfFilters(filter), [filter])
  const { data, isError, hasNextPage, isFetchingNextPage, fetchNextPage, refetch } = useDeliverables(filters)
  const view = useMemo(() => shelfView({ data, isError }), [data, isError])
  const [moreFailedFor, setMoreFailedFor] = useState<ShelfFilter | null>(null)
  const loadMore = useCallback(() => {
    setMoreFailedFor(null)
    void fetchNextPage().then((result) => {
      if (!result.isError) return
      console.error('[shelf] the next page of Deliverables did not load:', result.error)
      setMoreFailedFor(filter)
    })
  }, [fetchNextPage, filter])
  const retry = useCallback(() => {
    setMoreFailedFor(null)
    void refetch()
  }, [refetch])
  const detail = failureDetail(data?.pages)
  useEffect(() => {
    if (detail) console.error('[shelf] the Deliverables list answered success: false:', detail)
  }, [detail])
  const paging: ShelfPaging = {
    hasMore: hasNextPage === true,
    loadingMore: isFetchingNextPage,
    moreFailed: moreFailedFor === filter,
    loadMore,
    retry,
  }
  return { view, paging }
}

export function ShelfScreen({ now }: ShelfScreenProps) {
  const [filter, setFilter] = useState<ShelfFilter>(ALL_FILTER)
  const { view, paging } = useShelfList(filter)
  const auto = useShelfMascot(view)
  const total = view.status === 'ready' ? view.total : null
  return (
    <PageColumns>
      <div className={styles.page}>
        <header className={styles.header}>
          <div className={styles.heading}>
            <h1 className={styles.title}>{SHELF_TITLE}</h1>
            {total !== null && <p className={styles.count}>{countLine(total)}</p>}
            <Sections />
          </div>
          <div className={styles.auto}>
            <AutoMascot state={auto.mascot} onClipEnded={auto.onClipEnded} caption={auto.caption} size="small" />
          </div>
        </header>
        <FilterChips selected={filter} onSelect={setFilter} />
        <ShelfBody view={view} filter={filter} paging={paging} now={now ?? new Date()} />
      </div>
    </PageColumns>
  )
}

interface FilterChipsProps {
  readonly selected: ShelfFilter
  readonly onSelect: (filter: ShelfFilter) => void
}

function FilterChips({ selected, onSelect }: FilterChipsProps) {
  return (
    <div className={styles.chips} role="group" aria-label={FILTER_GROUP_LABEL}>
      {SHELF_FILTERS.map((filter) => (
        <button
          key={filter.label}
          type="button"
          className={styles.chip}
          aria-pressed={filter.artifactType === selected.artifactType}
          onClick={() => onSelect(filter)}
        >
          {filter.label}
        </button>
      ))}
    </div>
  )
}

/** The page's other sections (Blogs, Templates, Brand kit, Socials while offered, Explorer): the core's own pages. */
function Sections() {
  const { workspace } = useWorkspace()
  // PRD-251 D1: Socials only while the platform offers it, as on the page's own tabs.
  const tabs = visibleDeliverableTabs(workspace?.socials?.available === true).filter((tab) => tab !== 'outputs')
  const links = [
    ...tabs.map((tab) => ({ href: `/deliverables?tab=${tab}`, label: TAB_LABELS[tab] })),
    { href: EXPLORER_HREF, label: EXPLORER_LABEL },
  ]
  return (
    <nav aria-label={SECTIONS_LABEL}>
      <ul className={styles.sections}>
        {links.map((link) => (
          <li key={link.href}>
            <Link href={link.href} className={styles.sectionLink}>
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  )
}
