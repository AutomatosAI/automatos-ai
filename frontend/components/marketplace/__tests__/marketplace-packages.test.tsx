/**
 * PRD-230 US-007 — Marketplace Packages tab + detail popup.
 *
 * Covers the three ACs:
 *  1. Packages tab renders from the packages API; popup shows grouped members +
 *     setup summary.
 *  2. Showcase row lists showcase=true packages first.
 *  3. Install CTA calls the install path and renders the returned manifest.
 *
 * apiClient + sonner are stubbed so the components render without a backend.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'

const post = vi.fn()
const get = vi.fn()
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    get: (...args: any[]) => get(...args),
    post: (...args: any[]) => post(...args),
  },
}))

// sonner's `toast` is a callable with .success/.error — mirror that shape.
vi.mock('sonner', () => {
  const toast: any = vi.fn()
  toast.success = vi.fn()
  toast.error = vi.fn()
  return { toast }
})

import { MarketplacePackagesTab } from '../marketplace-packages-tab'
import { PackageDetailModal } from '../package-detail-modal'
import type { MarketplacePackage } from '../package-types'

function pkg(overrides: Partial<MarketplacePackage> = {}): MarketplacePackage {
  return {
    id: 'id-' + (overrides.slug || 'x'),
    slug: 'slug-x',
    name: 'A Package',
    description: 'Does things',
    vertical_tags: ['shopify', 'ecommerce'],
    matching: {},
    members: [],
    setup_manifest: {},
    showcase: false,
    ...overrides,
  }
}

const SHOPIFY_MGMT = pkg({
  slug: 'shopify-management',
  name: 'Shopify Management',
  description: 'Run the store',
  showcase: true,
  members: [
    { type: 'agent', ref: 'store-manager', name: 'Store Manager', description: 'Runs the shop' },
    { type: 'agent', ref: 'cs-agent', name: 'Customer Service', description: 'Answers buyers' },
    { type: 'playbook', ref: 'weekly-numbers', name: 'Weekly Numbers', description: 'The weekly report' },
    { type: 'llm', ref: 'gpt', name: 'GPT' },
  ],
  setup_manifest: {
    required_connects: [
      { app_name: 'SHOPIFY', app_type: 'ECOMMERCE', needs_oauth: true, note: 'Connect now for store data' },
    ],
    report_templates: [{ name: 'weekly-numbers', title: 'Weekly Numbers' }],
  },
})

const OTHER_PACK = pkg({
  slug: 'support-desk',
  name: 'Support Desk',
  description: 'Not showcased',
  showcase: false,
  members: [{ type: 'agent', ref: 'helpdesk', name: 'Helpdesk' }],
})

beforeEach(() => {
  post.mockReset()
  get.mockReset()
})

describe('MarketplacePackagesTab (US-007)', () => {
  it('renders package cards from injected packages, no fetch needed', () => {
    render(<MarketplacePackagesTab packages={[SHOPIFY_MGMT, OTHER_PACK]} />)
    // The non-showcased pack appears once (grid); the showcased one appears in
    // both the showcase row and the grid.
    expect(screen.getByText('Support Desk')).toBeInTheDocument()
    expect(screen.getAllByText('Shopify Management').length).toBeGreaterThanOrEqual(1)
    // At least the two grid cards + the showcased card in the showcase row.
    expect(screen.getAllByTestId('package-card').length).toBeGreaterThanOrEqual(2)
    // No API fetch when packages are injected.
    expect(get).not.toHaveBeenCalled()
  })

  it('lists showcase=true packages first, under the D11 copy', () => {
    // Non-showcased FIRST in the array — the showcase row must still lead.
    render(<MarketplacePackagesTab packages={[OTHER_PACK, SHOPIFY_MGMT]} />)
    const row = screen.getByTestId('packages-showcase-row')
    expect(within(row).getByText('Starter teams for your business')).toBeInTheDocument()
    // The showcase row contains the showcased package…
    expect(within(row).getByText('Shopify Management')).toBeInTheDocument()
    // …and NOT the non-showcased one.
    expect(within(row).queryByText('Support Desk')).not.toBeInTheDocument()
  })

  it('fetches the packages API when no packages are injected', async () => {
    get.mockResolvedValueOnce([SHOPIFY_MGMT])
    render(<MarketplacePackagesTab />)
    await waitFor(() => expect(get).toHaveBeenCalledWith('/api/marketplace/packages'))
    await waitFor(() => expect(screen.getAllByText('Shopify Management').length).toBeGreaterThan(0))
  })
})

describe('PackageDetailModal (US-007)', () => {
  it('shows members grouped by type with per-member descriptions + setup summary', () => {
    render(<PackageDetailModal pkg={SHOPIFY_MGMT} onClose={() => {}} />)

    // Grouped members — agent group heading + both agent names + a description.
    const agentGroup = screen.getByTestId('member-group-agent')
    expect(within(agentGroup).getByText(/Agents · 2/)).toBeInTheDocument()
    expect(within(agentGroup).getByText('Store Manager')).toBeInTheDocument()
    expect(within(agentGroup).getByText('Runs the shop')).toBeInTheDocument()
    // Playbook + LLM groups also render.
    expect(screen.getByTestId('member-group-playbook')).toBeInTheDocument()
    expect(screen.getByTestId('member-group-llm')).toBeInTheDocument()

    // Setup summary — apps to connect + reports you'll get.
    const setup = screen.getByTestId('package-setup-summary')
    expect(within(setup).getByText('Apps to connect')).toBeInTheDocument()
    expect(within(setup).getByText('SHOPIFY')).toBeInTheDocument()
    expect(within(setup).getByText('Connect now for store data')).toBeInTheDocument()
    expect(within(setup).getByText("Reports you’ll get")).toBeInTheDocument()
    expect(within(setup).getByText('Weekly Numbers')).toBeInTheDocument()
  })

  it('install CTA calls the install path and renders the returned manifest', async () => {
    post.mockResolvedValueOnce({
      success: true,
      slug: 'shopify-management',
      message: 'Installed',
      added_count: 7,
      registrations: [
        { type: 'agent', ref: '101', name: 'Store Manager', status: 'cloned', workspace_owned: true },
        { type: 'llm', ref: 'gpt', name: 'GPT', status: 'installed', workspace_owned: true },
      ],
      required_connects: [{ app_name: 'SHOPIFY', app_type: 'ECOMMERCE', needs_oauth: true }],
      warnings: [],
    })

    render(<PackageDetailModal pkg={SHOPIFY_MGMT} onClose={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Install package/i }))

    await waitFor(() =>
      expect(post).toHaveBeenCalledWith('/api/marketplace/packages/shopify-management/install'),
    )

    const result = await screen.findByTestId('install-result')
    expect(within(result).getByText(/7 registered to your workspace/)).toBeInTheDocument()
    const regs = screen.getByTestId('install-registrations')
    expect(within(regs).getByText(/Store Manager/)).toBeInTheDocument()
    // The required-connect surfaces as a guided next step (FR-4).
    expect(within(result).getByText('Next: connect these apps')).toBeInTheDocument()
  })

  it('renders the honest over-quota conversation without a manifest (D9)', async () => {
    post.mockResolvedValueOnce({
      success: false,
      over_quota: true,
      message: "Nothing's been installed yet — let's pick a plan that fits.",
      plan_recommendation: 'pro',
    })
    render(<PackageDetailModal pkg={SHOPIFY_MGMT} onClose={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: /Install package/i }))

    const result = await screen.findByTestId('install-result')
    expect(within(result).getByText(/pick a plan that fits/)).toBeInTheDocument()
    expect(within(result).getByText(/Recommended plan: pro/)).toBeInTheDocument()
    // No manifest registrations on the non-install path.
    expect(screen.queryByTestId('install-registrations')).not.toBeInTheDocument()
  })
})
