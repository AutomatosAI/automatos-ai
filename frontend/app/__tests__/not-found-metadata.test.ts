/**
 * PRD-260 §4 (P260-FX-21, M68), PRD-261 W-A8 — the 404 keeps the document title Next's own 404
 * sets, so every style's tab and screen reader say it, never the app's own title. app/not-found.tsx
 * is a server component for exactly this: it exports `metadata`. Builtin Play's edges test held
 * this until it left core; a theme's notFound@1 slot draws inside the page and cannot set it.
 *
 * Only the module's exports are read here: the slot the page renders through is stood in for, so
 * the import stays the page module's own.
 */
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/components/theme-host/theme-slot', () => ({ ThemeSlot: () => null }))

import { metadata, NOT_FOUND_TITLE } from '@/app/not-found'

/** Next's default 404 document title, word for word. */
const NEXT_404_TITLE = '404: This page could not be found.'

describe('app/not-found.tsx metadata (M68)', () => {
  it('titles the document NOT_FOUND_TITLE', () => {
    expect(metadata.title).toBe(NOT_FOUND_TITLE)
  })

  it("NOT_FOUND_TITLE is Next's own 404 title", () => {
    expect(NOT_FOUND_TITLE).toBe(NEXT_404_TITLE)
  })
})
