/**
 * PRD-261 C-04b — the Marketplace homepage gains one tab, "Themes" (Gerard, D3/Q11), after the
 * six it had, and Packages still leads (PRD-230). The Themes tab's content mounts only once it
 * is chosen, so the Marketplace asks for no theme data before then. The other tabs and the
 * homepage's data hooks are doubles: this test is about the tab strip, not their content.
 */
import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('framer-motion', () => ({
  motion: { div: ({ children }: { children?: ReactNode }) => <div>{children}</div> },
}))
vi.mock('next/navigation', () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn() }),
}))
vi.mock('@/lib/api-client', () => ({
  apiClient: { setCurrentPage: vi.fn(), get: vi.fn().mockResolvedValue([]) },
  getAdminWorkspaceOverride: () => null,
}))
vi.mock('@/hooks/use-workspace', () => ({ useWorkspace: () => ({ workspaceId: 'ws-1' }) }))
vi.mock('@/contexts/role-context', () => ({ useSystemRole: () => ({ isAdmin: false }) }))
vi.mock('@/hooks/use-marketplace-api', () => ({
  useFeaturedItems: () => ({ data: [], isLoading: false }),
  useToggleFeatured: () => ({ mutate: vi.fn(), isLoading: false }),
}))
vi.mock('@/components/analytics/admin-workspace-switcher', () => ({ AdminWorkspaceSwitcher: () => null }))
vi.mock('@/components/shared', () => ({
  PageHeader: ({ title }: { title: string }) => <h1>{title}</h1>,
  SearchInput: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <input aria-label="Search the marketplace" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}))
vi.mock('@/components/marketplace/marketplace-item-modal', () => ({ MarketplaceItemModal: () => null }))
vi.mock('@/components/marketplace/featured-showcase-card', () => ({ FeaturedShowcaseCard: () => null }))
vi.mock('@/components/marketplace/marketplace-packages-tab', () => ({
  MarketplacePackagesTab: () => <div data-testid="packages-tab" />,
}))
vi.mock('@/components/marketplace/marketplace-tools-tab', () => ({ MarketplaceToolsTab: () => null }))
vi.mock('@/components/marketplace/marketplace-agents-tab', () => ({ MarketplaceAgentsTab: () => null }))
vi.mock('@/components/marketplace/marketplace-playbooks-tab', () => ({ MarketplacePlaybooksTab: () => null }))
vi.mock('@/components/marketplace/marketplace-llms-tab', () => ({ MarketplaceLlmsTab: () => null }))
vi.mock('@/components/marketplace/marketplace-plugins-tab', () => ({ MarketplacePluginsTab: () => null }))
vi.mock('@/components/marketplace/marketplace-skills-tab', () => ({ MarketplaceSkillsTab: () => null }))
// The tab's real name and value (listing-format.ts, a module of constants); its content is a double.
vi.mock('@/components/marketplace/themes/tab', async () => {
  const { THEMES_TAB_LABEL, THEMES_TAB_VALUE } = await import('../listing-format')
  return {
    THEMES_TAB_LABEL,
    THEMES_TAB_VALUE,
    MarketplaceThemesTab: ({ searchQuery }: { searchQuery?: string }) => (
      <div data-testid="themes-tab">{`search: ${searchQuery ?? ''}`}</div>
    ),
  }
})

import { MarketplaceHomepage } from '../../marketplace-homepage'

const TABS_BEFORE = ['Packages', 'Applications', 'Agents', 'Playbooks', 'LLMs', 'Capabilities']

function chooseTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.mouseDown(tab) // Radix Tabs activates on mouse down
  fireEvent.click(tab)
}

describe('the Marketplace’s Themes tab', () => {
  it('is one more tab after the six it had, named "Themes", and Packages still leads', () => {
    render(<MarketplaceHomepage />)

    const names = screen.getAllByRole('tab').map((tab) => tab.textContent?.trim())
    expect(names).toEqual([...TABS_BEFORE, 'Themes'])
    expect(screen.queryByRole('tab', { name: /skins/i })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Packages' })).toHaveAttribute('data-state', 'active')
    expect(screen.getByTestId('packages-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('themes-tab')).not.toBeInTheDocument()
  })

  it('opens the Themes tab with the Marketplace’s search', () => {
    render(<MarketplaceHomepage />)
    fireEvent.change(screen.getByLabelText('Search the marketplace'), { target: { value: 'aurora' } })

    chooseTab('Themes')

    expect(screen.getByRole('tab', { name: 'Themes' })).toHaveAttribute('data-state', 'active')
    expect(screen.getByTestId('themes-tab')).toHaveTextContent('search: aurora')
  })
})
