/**
 * PRD-260 T-007a — the shelf's pure data (lib/play/shelf.ts), ported from
 * Automatos Play's tests/deliverables/shelf.test.ts onto the core's
 * Deliverables: the chips as the core's artifact_type values and words, the
 * badge, the card date, the count and caption lines, the empty states, the
 * feed's filters, and what the list query's pages amount to. A list answer
 * of success: false never reaches the pages: the hook throws it (P260-FX-12;
 * play-shelf.test.tsx runs that through the real hook). "Now" is pinned to a
 * local date.
 */
import { describe, expect, it } from 'vitest'

import { type Deliverable, type DeliverableListResponse, FEED_DEFAULT_FILTERS } from '@/hooks/use-deliverables-api'
import {
  ALL_FILTER,
  countLine,
  emptyMessage,
  kindOf,
  SHELF_FILTERS,
  shelfCaption,
  shelfDate,
  shelfFilters,
  shelfView,
  typeLabel,
} from '../shelf'

const NOW = new Date(2026, 9, 14, 15, 30)

function item(id: string): Deliverable {
  return { id, title: id, artifact_type: 'document', created_at: '2026-10-01T09:00:00+00:00' } as Deliverable
}

function page(ids: readonly string[], total: number, offset = 0): DeliverableListResponse {
  return { success: true, deliverables: ids.map(item), total, limit: 24, offset }
}

describe('the chips', () => {
  it("maps each chip to the core's artifact_type, in the core's words", () => {
    expect(SHELF_FILTERS.map((filter) => [filter.label, filter.artifactType])).toEqual([
      ['All', null],
      ['Reports', 'report'],
      ['Documents', 'document'],
      ['Images', 'image'],
      ['Spreadsheets', 'spreadsheet'],
      ['Slides', 'slide'],
    ])
    expect(SHELF_FILTERS[0]).toBe(ALL_FILTER)
  })

  it("narrows the core's feed filters (heartbeats hidden) to the chip's type", () => {
    expect(shelfFilters(ALL_FILTER)).toEqual({ ...FEED_DEFAULT_FILTERS, artifact_type: null })
    expect(shelfFilters(SHELF_FILTERS[1])).toEqual({ ...FEED_DEFAULT_FILTERS, artifact_type: 'report' })
    expect(shelfFilters(SHELF_FILTERS[1]).source_type_exclude).toEqual(['heartbeat'])
  })
})

describe('the card', () => {
  it('labels types as the mock does', () => {
    expect(typeLabel('report')).toBe('Report')
    expect(typeLabel('slide')).toBe('Slides')
    expect(typeLabel('spreadsheet')).toBe('Spreadsheet')
    expect(typeLabel('blog_post')).toBe('Blog post')
    expect(typeLabel('constructor')).toBe('Constructor')
  })

  it('falls back to document for a missing type', () => {
    expect(kindOf({ artifact_type: '' })).toBe('document')
    expect(kindOf({ artifact_type: ' Image ' })).toBe('image')
  })

  it('dates cards Today, Yesterday, by weekday, then by day and month', () => {
    expect(shelfDate(new Date(2026, 9, 14, 8).toISOString(), NOW)).toBe('Today')
    expect(shelfDate(new Date(2026, 9, 13, 8).toISOString(), NOW)).toBe('Yesterday')
    expect(shelfDate(new Date(2026, 9, 12, 8).toISOString(), NOW)).toBe('Mon')
    expect(shelfDate(new Date(2026, 9, 4, 8).toISOString(), NOW)).toBe('4 Oct')
    expect(shelfDate(null, NOW)).toBe('')
    expect(shelfDate('not a date', NOW)).toBe('')
  })

  it('words the count, the caption and the empty states', () => {
    expect(countLine(1)).toBe('1 Deliverable')
    expect(countLine(15)).toBe('15 Deliverables')
    expect(shelfCaption(null)).toBe('Fetching your shelf')
    expect(shelfCaption(0)).toBe('Your shelf is waiting')
    expect(shelfCaption(1)).toBe('Your first Deliverable!')
    expect(shelfCaption(15)).toBe('15 on the shelf. Nice collection!')
    expect(emptyMessage(ALL_FILTER)).toMatch(/^Nothing on your shelf yet/)
    expect(emptyMessage(SHELF_FILTERS[3])).toBe('No images yet. Ask Auto for one and it lands here.')
  })
})

describe('shelfView', () => {
  it("is loading before the first page, failed when the request failed (an HTTP error, or the list's success: false)", () => {
    expect(shelfView({ isError: false })).toEqual({ status: 'loading' })
    expect(shelfView({ isError: true })).toEqual({ status: 'failed' })
    expect(shelfView({ data: { pages: [] }, isError: true })).toEqual({ status: 'failed' })
  })

  it('joins the pages, one row per id, with the latest total', () => {
    const view = shelfView({ data: { pages: [page(['a', 'b'], 3), page(['b', 'c'], 3, 2)] }, isError: false })
    expect(view.status === 'ready' && view.items.map((row) => row.id)).toEqual(['a', 'b', 'c'])
    expect(view).toEqual({ status: 'ready', items: expect.any(Array), total: 3 })
  })

  it('keeps the loaded rows when a later page or a background refetch failed (react-query keeps the data)', () => {
    const view = shelfView({ data: { pages: [page(['a', 'b'], 30)] }, isError: true })
    expect(view).toMatchObject({ status: 'ready', total: 30 })
    expect(view.status === 'ready' && view.items).toHaveLength(2)
  })
})
