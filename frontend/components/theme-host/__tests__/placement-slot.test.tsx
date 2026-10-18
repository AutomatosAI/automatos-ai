/**
 * PRD-261 A-12b — the placement slots in PageHeader and EmptyState. With no theme (no provider,
 * or an active theme that placed nothing) both draw exactly the markup they drew before the
 * slots existed, pinned here as HTML. A registered `page-header.mascot@1` / `page-header.art@1`
 * draws in the header under both bases, a registered `empty-state.mascot@1` stands in for the
 * icon tile, and a placement that throws draws the fallback and tells the host.
 */
import React, { type ReactNode } from 'react'
import { cleanup, render, screen } from '@testing-library/react'
import type { LucideIcon } from 'lucide-react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('next-themes', () => ({ useTheme: () => ({ theme: 'dark', setTheme: vi.fn() }) }))
vi.mock('framer-motion', () => ({
  motion: { div: ({ children, className }: { children: ReactNode; className?: string }) => <div className={className}>{children}</div> },
}))

import { EmptyState } from '@/components/shared/empty-state'
import { PageHeader } from '@/components/shared/page-header'
import { UiStyleProvider } from '@/contexts/ui-style-context'
import { THEME_SLOT_IDS } from '@/lib/theme-host/constants'
import type { SlotId } from '@/lib/theme-host/registry-types'
import { APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION } from '@/lib/ui-style'

import {
  EMPTY_STATE_MASCOT_SLOT,
  PAGE_HEADER_ART_SLOT,
  PAGE_HEADER_MASCOT_SLOT,
  PLACEMENT_SLOT_ATTR,
  PLACEMENT_SLOT_IDS,
  PlacementSlotsProvider,
  type EmptyStatePlacementProps,
  type PageHeaderPlacementProps,
} from '../placement-slot'
import { ThemeHostContext, type ThemeHostContextValue } from '../theme-host-context'

/** The markup each primitive drew before A-12b, for the props the renders below pass. */
const STUDIO_HEADER_HTML =
  '<div class="cc-headrow"><div class="cc-head"><p class="cc-eyebrow">Workforce</p>' +
  '<h1 data-testid="page-title" class="cc-h1">Knowledge Bases</h1><p class="cc-sub">What they read.</p></div>' +
  '<div class="cc-actions"><button>Add</button></div></div>'
const CLASSIC_HEADER_HTML =
  '<div class="flex flex-col sm:flex-row justify-between items-start gap-3"><div class="min-w-0">' +
  '<h1 data-testid="page-title" class="text-2xl md:text-3xl font-bold mb-1 md:mb-2">Knowledge <span class="gradient-text">Bases</span></h1>' +
  '<p class="text-sm md:text-base text-muted-foreground">Docs</p></div></div>'
const EMPTY_STATE_HTML =
  '<div class="flex flex-col items-center justify-center py-16 px-6 text-center">' +
  '<div class="w-16 h-16 rounded-2xl bg-secondary/50 border border-border/30 flex items-center justify-center mb-4">' +
  '<svg data-testid="empty-icon" class="w-8 h-8 text-muted-foreground/60"></svg></div>' +
  '<h3 class="text-lg font-semibold mb-1">No templates yet</h3>' +
  '<p class="text-sm text-muted-foreground max-w-sm mb-6">Start from a blank page.</p></div>'

const PLACEMENT_FAILURE = new Error('deliberate placement failure')

function EmptyIconStub({ className }: { className?: string }) {
  return <svg data-testid="empty-icon" className={className} />
}

/** EmptyState takes a lucide icon; a plain svg keeps the pinned markup short. */
const EmptyIcon = EmptyIconStub as unknown as LucideIcon

function HelloFrame({ children }: { children?: ReactNode }) {
  return <>{children}</>
}

function HeaderMascot({ base, title, titleAccent, eyebrow }: PageHeaderPlacementProps) {
  return <span data-testid="header-mascot">{[base, title, titleAccent, eyebrow ?? '-'].join('|')}</span>
}

function HeaderArt({ base }: PageHeaderPlacementProps) {
  return <span data-testid="header-art">{base}</span>
}

function EmptyMascot({ base, title, description }: EmptyStatePlacementProps) {
  return <span data-testid="empty-mascot">{[base, title, description].join('|')}</span>
}

function BrokenPlacement(): React.ReactElement {
  throw PLACEMENT_FAILURE
}

function slotsOf(entries: ReadonlyArray<readonly [SlotId, unknown]>): ReadonlyMap<SlotId, unknown> {
  return new Map<SlotId, unknown>(entries)
}

/** An active theme that registered only its frame: no placement anywhere. */
const FRAME_ONLY = slotsOf([['frame@1', HelloFrame]])

function withSlots(slots: ReadonlyMap<SlotId, unknown> | null, node: React.ReactElement, studio = false): React.ReactElement {
  const placed = slots === null ? node : <PlacementSlotsProvider slots={slots}>{node}</PlacementSlotsProvider>
  return studio ? <UiStyleProvider initialStyle="studio">{placed}</UiStyleProvider> : placed
}

const studioHeader = <PageHeader title="Knowledge" titleAccent="Bases" eyebrow="Workforce" lede="What they read." actions={<button>Add</button>} />
const classicHeader = <PageHeader title="Knowledge" titleAccent="Bases" subtitle="Docs" />
const emptyState = <EmptyState icon={EmptyIcon} title="No templates yet" description="Start from a blank page." />

beforeEach(() => window.localStorage.setItem(APPEARANCE_DEFAULTS_KEY, APPEARANCE_DEFAULTS_VERSION))
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function expectUnchanged(slots: ReadonlyMap<SlotId, unknown> | null): void {
  expect(render(withSlots(slots, studioHeader, true)).container.innerHTML).toBe(STUDIO_HEADER_HTML)
  cleanup()
  expect(render(withSlots(slots, classicHeader)).container.innerHTML).toBe(CLASSIC_HEADER_HTML)
  cleanup()
  expect(render(withSlots(slots, emptyState)).container.innerHTML).toBe(EMPTY_STATE_HTML)
}

describe('PRD-261 A-12b — no theme draws the same DOM', () => {
  it('PageHeader (Studio and Classic) and EmptyState are unchanged with no provider', () => {
    expectUnchanged(null)
  })

  it('…and unchanged under an active theme that placed nothing', () => {
    expectUnchanged(FRAME_ONLY)
  })

  it('every placement slot is one the loader accepts', () => {
    expect(PLACEMENT_SLOT_IDS.every((slot) => THEME_SLOT_IDS.includes(slot))).toBe(true)
  })
})

describe('PRD-261 A-12b — a registered placement renders', () => {
  const headerSlots = slotsOf([['frame@1', HelloFrame], [PAGE_HEADER_MASCOT_SLOT, HeaderMascot], [PAGE_HEADER_ART_SLOT, HeaderArt]])

  it('Studio: art before the head, the mascot after it and before the actions', () => {
    const { container } = render(withSlots(headerSlots, studioHeader, true))
    const row = container.querySelector('.cc-headrow')!
    expect(Array.from(row.children).map((child) => child.getAttribute(PLACEMENT_SLOT_ATTR) ?? child.className)).toEqual([
      PAGE_HEADER_ART_SLOT,
      'cc-head',
      PAGE_HEADER_MASCOT_SLOT,
      'cc-actions',
    ])
    expect(screen.getByTestId('header-mascot')).toHaveTextContent('studio|Knowledge|Bases|Workforce')
    expect(screen.getByTestId('header-art')).toHaveTextContent('studio')
  })

  it('Classic: the same two slots around the gradient title', () => {
    const { container } = render(withSlots(headerSlots, classicHeader))
    const mascotHost = container.querySelector(`[${PLACEMENT_SLOT_ATTR}="${PAGE_HEADER_MASCOT_SLOT}"]`)
    expect(mascotHost).toContainElement(screen.getByTestId('header-mascot'))
    expect(screen.getByTestId('header-mascot')).toHaveTextContent('classic|Knowledge|Bases|-')
    expect(container.querySelector('.gradient-text')).toHaveTextContent('Bases')
  })

  it('EmptyState: the mascot stands in for the icon tile', () => {
    render(withSlots(slotsOf([[EMPTY_STATE_MASCOT_SLOT, EmptyMascot]]), emptyState))
    expect(screen.getByTestId('empty-mascot')).toHaveTextContent('classic|No templates yet|Start from a blank page.')
    expect(screen.queryByTestId('empty-icon')).toBeNull()
    expect(screen.getByRole('heading', { name: 'No templates yet' })).toBeInTheDocument()
  })

  it('a placement that throws draws the fallback and tells the host', () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const reportRenderFailure = vi.fn()
    const host: ThemeHostContextValue = { status: 'active', reason: null, frame: HelloFrame, reportRenderFailure }
    render(
      <ThemeHostContext.Provider value={host}>
        {withSlots(slotsOf([[EMPTY_STATE_MASCOT_SLOT, BrokenPlacement]]), emptyState)}
      </ThemeHostContext.Provider>,
    )
    expect(screen.getByTestId('empty-icon')).toBeInTheDocument()
    expect(reportRenderFailure).toHaveBeenCalledTimes(1)
    expect(reportRenderFailure).toHaveBeenCalledWith(PLACEMENT_FAILURE)
  })
})
