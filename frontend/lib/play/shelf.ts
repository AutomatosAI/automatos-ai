import { deliverableLabel, isDeliverableType } from '@/components/icons/deliverable-icon'
import {
  type Deliverable,
  type DeliverableListResponse,
  FEED_DEFAULT_FILTERS,
  type FilterState,
} from '@/hooks/use-deliverables-api'

import { groupOf, relativeTime } from './conversations'

/**
 * PRD-260 §4 — Play's Deliverables shelf (Ember mock 04) as pure data, ported
 * from Automatos Play's lib/deliverables/shelf.ts onto the core's Deliverables
 * (hooks/use-deliverables-api.ts): the type chips as the core's artifact_type
 * filter, the type badge, the card date, the count line, Auto's caption, the
 * empty states, and what the list query's pages amount to on screen. Heartbeat
 * outputs stay off the shelf, as on the core's feed (FEED_DEFAULT_FILTERS).
 * "Now" is always passed in, so tests pin it.
 */

export type ShelfFilter = {
  readonly label: string
  /** The list's artifact_type; null for All. */
  readonly artifactType: string | null
}

export const ALL_FILTER: ShelfFilter = { label: 'All', artifactType: null }

/** Mock 04's chips, in its order; each label is the core's own word for the type. */
const CHIP_TYPES = ['report', 'document', 'image', 'spreadsheet', 'slide'] as const

export const SHELF_FILTERS: readonly ShelfFilter[] = [
  ALL_FILTER,
  ...CHIP_TYPES.map((type) => ({ label: deliverableLabel(type), artifactType: type })),
]

/** The type a Deliverable with none is shown as. */
export const DEFAULT_KIND = 'document'

/**
 * How often the open shelf asks for its pages again while the tab is visible
 * (P260-FX-12, M38): a Deliverable a running Mission files lands, and Auto
 * celebrates it, without a refocus or a navigation.
 */
export const SHELF_REFRESH_MS = 30_000

/** Badge words that aren't the type with a capital (mock 04 says "Slides"). */
const KIND_LABELS: Readonly<Record<string, string>> = { slide: 'Slides', blog_post: 'Blog post' }

/** The list query's filters for one chip: the core's feed, heartbeats hidden, narrowed to the type. */
export function shelfFilters(filter: ShelfFilter): FilterState {
  return { ...FEED_DEFAULT_FILTERS, artifact_type: filter.artifactType }
}

export function kindOf(deliverable: Pick<Deliverable, 'artifact_type'>): string {
  const kind = deliverable.artifact_type?.trim().toLowerCase()
  return kind ? kind : DEFAULT_KIND
}

/** "Report", "Slides", "Spreadsheet": one Deliverable's type on its card. */
export function typeLabel(kind: string): string {
  if (Object.prototype.hasOwnProperty.call(KIND_LABELS, kind)) return KIND_LABELS[kind]
  return kind.charAt(0).toUpperCase() + kind.slice(1).replace(/_/g, ' ')
}

/** "Today", "Yesterday", "Mon", "4 Oct"; "" when the date is missing or unreadable. */
export function shelfDate(createdAt: string | null | undefined, now: Date): string {
  if (!createdAt) return ''
  const date = new Date(createdAt)
  if (Number.isNaN(date.getTime())) return ''
  return groupOf(date, now) === 'Today' ? 'Today' : relativeTime(date, now)
}

/** "1 Deliverable", "15 Deliverables". */
export function countLine(total: number): string {
  return `${total} ${total === 1 ? 'Deliverable' : 'Deliverables'}`
}

/** What Auto says over the shelf while he idles. */
export function shelfCaption(total: number | null): string {
  if (total === null) return 'Fetching your shelf'
  if (total === 0) return 'Your shelf is waiting'
  if (total === 1) return 'Your first Deliverable!'
  return `${total} on the shelf. Nice collection!`
}

export function emptyMessage(filter: ShelfFilter): string {
  if (filter.artifactType === null) {
    return 'Nothing on your shelf yet. Ask Auto for a report or a document and it lands here.'
  }
  return `No ${filter.label.toLowerCase()} yet. Ask Auto for one and it lands here.`
}

/** The chip's type for a ?artifact_type= link, when it is one of the chips; else All. */
export function filterForType(artifactType: string | null | undefined): ShelfFilter {
  const wanted = artifactType?.trim().toLowerCase()
  if (!wanted || !isDeliverableType(wanted)) return ALL_FILTER
  return SHELF_FILTERS.find((filter) => filter.artifactType === wanted) ?? ALL_FILTER
}

// --- The list query's pages, as the shelf shows them --------------------------------

export type ShelfView =
  | { readonly status: 'loading' }
  | { readonly status: 'failed' }
  | {
      readonly status: 'ready'
      readonly items: readonly Deliverable[]
      readonly total: number
    }

/** What useDeliverables(filters) gives the shelf: react-query's infinite result, read only. */
export interface ShelfQuery {
  readonly data?: { readonly pages: readonly DeliverableListResponse[] }
  readonly isError: boolean
}

/** One per id, the first one kept: a new Deliverable shifts the offsets between pages. */
function uniqueById(pages: readonly DeliverableListResponse[]): readonly Deliverable[] {
  const items = pages.flatMap((page) => page.deliverables ?? [])
  return items.filter((item, index) => items.findIndex((other) => other.id === item.id) === index)
}

/**
 * The shelf's state from the list query. A list page that answers
 * success: false is the query's error (useDeliverables throws it), never a
 * page: with no page loaded that is the error state, never "nothing here yet";
 * once pages are loaded they stay on screen through a failed refetch or a
 * failed Load more (react-query keeps the data), with the latest total.
 */
export function shelfView(query: ShelfQuery): ShelfView {
  const pages = query.data?.pages ?? []
  if (pages.length === 0) return query.isError ? { status: 'failed' } : { status: 'loading' }
  return { status: 'ready', items: uniqueById(pages), total: pages[pages.length - 1].total }
}
