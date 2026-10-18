'use client'

/**
 * PRD-261 A-12b — placement slots: named spots in the shared PageHeader and EmptyState where the
 * active theme may place its mascot or its art, so a theme reaches those pages without cloning
 * them (PRD §4.3 item 6, §4.6 way 2). Which pages reach each slot under each base is in
 * docs/theme-host/placement-reach.md.
 *
 * With nothing registered for a slot, PlacementSlot draws its fallback as it is (nothing for the
 * header slots, the core icon tile for the empty state), so a page with no theme has exactly the
 * DOM it had before. A registered component draws inside a host element carrying
 * `data-theme-slot`, inside a boundary: if it throws, the fallback draws again and the host is
 * told, as the frame slot does (theme-frame-slot.tsx).
 *
 * The registrations come from PlacementSlotsProvider, which the theme host renders around the
 * app with the active theme's slots (ThemeRegistrations.slots, lib/theme-host/types.ts).
 */
import { createContext, useContext, type ComponentType, type ReactNode } from 'react'

import { useIsStudio } from '@/hooks/use-studio-theme'
import type { SlotId, ThemeBase } from '@/lib/theme-host/registry-types'

import { ThemeErrorBoundary } from './theme-error-boundary'
import { useThemeHost } from './theme-host-context'

/** The attribute on a placed component's host element, naming its slot (for theme CSS and the browser lane). */
export const PLACEMENT_SLOT_ATTR = 'data-theme-slot'

export const PAGE_HEADER_MASCOT_SLOT = 'page-header.mascot@1'
export const PAGE_HEADER_ART_SLOT = 'page-header.art@1'
export const EMPTY_STATE_MASCOT_SLOT = 'empty-state.mascot@1'

export type PageHeaderSlotId = typeof PAGE_HEADER_MASCOT_SLOT | typeof PAGE_HEADER_ART_SLOT
export type EmptyStateSlotId = typeof EMPTY_STATE_MASCOT_SLOT
export type PlacementSlotId = PageHeaderSlotId | EmptyStateSlotId

/** Every placement slot; each one is also a SlotId the loader accepts (constants.ts THEME_SLOT_IDS). */
export const PLACEMENT_SLOT_IDS: readonly PlacementSlotId[] = [
  PAGE_HEADER_MASCOT_SLOT,
  PAGE_HEADER_ART_SLOT,
  EMPTY_STATE_MASCOT_SLOT,
]

/** Props of `page-header.mascot@1` and `page-header.art@1`: the header the component sits in. */
export interface PageHeaderPlacementProps {
  /** The base the page draws on: Studio's editorial head or Classic's gradient title. */
  readonly base: ThemeBase
  readonly title: string
  readonly titleAccent: string
  readonly eyebrow: string | null
}

/** Props of `empty-state.mascot@1`: the empty state the component stands in, in place of its icon. */
export interface EmptyStatePlacementProps {
  readonly base: ThemeBase
  readonly title: string
  readonly description: string
}

export type PlacementSlotProps =
  | { readonly slot: PageHeaderSlotId; readonly props: PageHeaderPlacementProps; readonly fallback?: ReactNode }
  | { readonly slot: EmptyStateSlotId; readonly props: EmptyStatePlacementProps; readonly fallback?: ReactNode }

type PlacedComponent = ComponentType<PageHeaderPlacementProps | EmptyStatePlacementProps>

/** The active theme's registered slots; null with no theme (and outside the provider). */
const PlacementSlotsContext = createContext<ReadonlyMap<SlotId, unknown> | null>(null)

interface PlacementSlotsProviderProps {
  readonly slots: ReadonlyMap<SlotId, unknown> | null
  readonly children: ReactNode
}

/** Hands the active theme's registered slots to every PlacementSlot below it. */
export function PlacementSlotsProvider({ slots, children }: PlacementSlotsProviderProps) {
  return <PlacementSlotsContext.Provider value={slots}>{children}</PlacementSlotsContext.Provider>
}

/** The component the active theme registered for `slot`, or null. The loader checked it is a component. */
function usePlacedComponent(slot: PlacementSlotId): PlacedComponent | null {
  const registered = useContext(PlacementSlotsContext)?.get(slot)
  return registered === undefined ? null : (registered as PlacedComponent)
}

/** The base a placement draws on: Studio wherever the Studio layout renders, else Classic. */
export function usePlacementBase(): ThemeBase {
  return useIsStudio() ? 'studio' : 'classic'
}

function ignoreRenderFailure(): void {
  // No theme host above (an isolated mount): the boundary's fallback is the whole recovery.
}

/** One placement slot: the theme's component when one is registered, else `fallback` unchanged. */
export function PlacementSlot({ slot, props, fallback = null }: PlacementSlotProps) {
  const Placed = usePlacedComponent(slot)
  const host = useThemeHost()
  if (Placed === null) return <>{fallback}</>
  const hostAttrs = { [PLACEMENT_SLOT_ATTR]: slot }
  return (
    <ThemeErrorBoundary fallback={fallback} onError={host?.reportRenderFailure ?? ignoreRenderFailure}>
      <div {...hostAttrs}>
        <Placed {...props} />
      </div>
    </ThemeErrorBoundary>
  )
}
