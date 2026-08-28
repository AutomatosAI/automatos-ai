/**
 * PRD-222 W2·S1b (US-024) — marketplace plan-label chips. The full catalog stays
 * VISIBLE for every tier; items beyond the workspace's depth carry a plan chip
 * ('Pro' / 'Business'). No item is hidden (D5). MarketplaceCard + useWorkspace
 * are stubbed so the test isolates the grid's chip behaviour.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('../marketplace-card', () => ({
  MarketplaceCard: ({ item }: any) => <div data-testid="card">{item.name}</div>,
}))

// Basic tier: marketplace_depth 1 ⇒ pro/business items get chipped.
vi.mock('@/components/workspace-provider', () => ({
  useWorkspace: () => ({ workspace: { exposure: { marketplace_depth: 1 } } }),
}))

import { MarketplaceGrid } from '../marketplace-grid'

function item(id: number, name: string, metadata: Record<string, unknown>) {
  return {
    id,
    type: 'agent',
    name,
    description: '',
    creator_name: '',
    tags: [],
    install_count: 0,
    is_featured: false,
    version: '1',
    metadata,
    created_at: '',
    updated_at: '',
  }
}

const items = [
  item(1, 'Basic Agent', {}),
  item(2, 'Pro Agent', { min_plan: 'pro' }),
  item(3, 'Biz Agent', { tier: 'business' }),
]

describe('MarketplaceGrid — plan-label chips (US-024)', () => {
  it('shows chips beyond the tier depth and hides nothing', () => {
    render(<MarketplaceGrid items={items as any} onItemClick={() => {}} />)

    // All three cards render — the catalog is never trimmed for a lower tier.
    expect(screen.getAllByTestId('card')).toHaveLength(3)

    // Two chips for a basic (depth 1) workspace: Pro + Business.
    const chips = screen.getAllByTestId('plan-chip')
    expect(chips).toHaveLength(2)
    expect(chips.map((c) => c.textContent).sort()).toEqual(['Business', 'Pro'])
  })
})
