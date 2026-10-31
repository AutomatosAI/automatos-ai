/**
 * PRD-261 — the Marketplace homepage's tab strip: "Themes" (Gerard, D3/Q11) is the first tab and
 * the one the Marketplace opens on (the storefront, 30 Oct 2026), before the six it had in their
 * order (Packages first among them, PRD-230). A tab's content mounts only once it is chosen, so
 * the Marketplace asks for no package data until Packages is. A `?tab=` link opens the tab it
 * names. The other tabs and the homepage's data hooks are doubles: this test is about the tab
 * strip, not their content.
 */
import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const nav = vi.hoisted(() => ({ params: new URLSearchParams() }))

vi.mock('framer-motion', () => ({
  motion: { div: ({ children }: { children?: ReactNode }) => <div>{children}</div> },
}))
vi.mock('next/navigation', () => ({
  useSearchParams: () => nav.params,
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

const TABS_AFTER_THEMES = ['Packages', 'Applications', 'Agents', 'Playbooks', 'LLMs', 'Capabilities']

function chooseTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.mouseDown(tab) // Radix Tabs activates on mouse down
  fireEvent.click(tab)
}

beforeEach(() => {
  nav.params = new URLSearchParams()
})

describe('the Marketplace’s Themes tab', () => {
  it('is the first tab, named "Themes", and the Marketplace opens on it', () => {
    render(<MarketplaceHomepage />)

    const names = screen.getAllByRole('tab').map((tab) => tab.textContent?.trim())
    expect(names).toEqual(['Themes', ...TABS_AFTER_THEMES])
    expect(screen.queryByRole('tab', { name: /skins/i })).not.toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Themes' })).toHaveAttribute('data-state', 'active')
    expect(screen.getByTestId('themes-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('packages-tab')).not.toBeInTheDocument()
  })

  it('opens with the Marketplace’s search', () => {
    render(<MarketplaceHomepage />)
    fireEvent.change(screen.getByLabelText('Search the marketplace'), { target: { value: 'aurora' } })

    expect(screen.getByTestId('themes-tab')).toHaveTextContent('search: aurora')
  })

  it('gives way to Packages when it is chosen', () => {
    render(<MarketplaceHomepage />)

    chooseTab('Packages')

    expect(screen.getByRole('tab', { name: 'Packages' })).toHaveAttribute('data-state', 'active')
    expect(screen.getByTestId('packages-tab')).toBeInTheDocument()
    expect(screen.queryByTestId('themes-tab')).not.toBeInTheDocument()
  })

  it.each([
    ['tools', 'Applications'],
    ['playbooks', 'Playbooks'],
    ['packages', 'Packages'],
  ])('a ?tab=%s link opens %s', (param, name) => {
    nav.params = new URLSearchParams({ tab: param })
    render(<MarketplaceHomepage />)

    expect(screen.getByRole('tab', { name })).toHaveAttribute('data-state', 'active')
    expect(screen.getByRole('tab', { name: 'Themes' })).toHaveAttribute('data-state', 'inactive')
  })

  it('an unknown ?tab= opens Themes', () => {
    nav.params = new URLSearchParams({ tab: 'nothing-here' })
    render(<MarketplaceHomepage />)

    expect(screen.getByRole('tab', { name: 'Themes' })).toHaveAttribute('data-state', 'active')
  })
})
