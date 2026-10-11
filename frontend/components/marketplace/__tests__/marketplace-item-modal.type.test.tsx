/**
 * Agents and playbooks have separate id sequences, so the item modal must ask
 * for the item by type and install it through the matching route. Before the
 * fix, a playbook card opened agent #id's details and its install button
 * installed that agent (or 404'd).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const get = vi.fn()
const post = vi.fn()
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    get: (...args: any[]) => get(...args),
    post: (...args: any[]) => post(...args),
  },
}))

const installPlaybook = vi.fn()
vi.mock('@/hooks/use-playbook-api', () => ({
  useInstallPlaybookFromMarketplace: () => ({ mutateAsync: (...args: any[]) => installPlaybook(...args) }),
}))

vi.mock('sonner', () => {
  const toast: any = vi.fn()
  toast.success = vi.fn()
  toast.error = vi.fn()
  return { toast }
})

import { MarketplaceItemModal } from '../marketplace-item-modal'
import { marketplaceItemKey } from '../marketplace-item-ref'

const ITEM = {
  id: 5,
  name: 'Weekly numbers',
  description: 'The weekly report',
  creator_name: 'Automatos',
  tags: [],
  install_count: 3,
  is_featured: false,
  version: '1.0',
  metadata: {},
  created_at: '2026-10-01T00:00:00Z',
  updated_at: '2026-10-01T00:00:00Z',
}

describe('MarketplaceItemModal picks the item by type', () => {
  beforeEach(() => {
    get.mockReset()
    post.mockReset()
    installPlaybook.mockReset()
  })

  it('loads a playbook with ?type=recipe and installs it through the playbook route', async () => {
    get.mockResolvedValue({ ...ITEM, type: 'recipe' })
    installPlaybook.mockResolvedValue({ message: 'Installed' })

    render(<MarketplaceItemModal itemId={5} itemType="recipe" onClose={() => {}} />)

    await waitFor(() => expect(get).toHaveBeenCalledWith('/api/marketplace/items/5?type=recipe'))
    fireEvent.click(await screen.findByRole('button', { name: /add to workspace/i }))

    await waitFor(() => expect(installPlaybook).toHaveBeenCalledWith(5))
    expect(post).not.toHaveBeenCalled()
  })

  it('loads an agent with ?type=agent and installs it through the agent route', async () => {
    get.mockResolvedValue({ ...ITEM, type: 'agent' })
    post.mockResolvedValue({ message: 'Installed' })

    render(<MarketplaceItemModal itemId={5} itemType="agent" onClose={() => {}} />)

    await waitFor(() => expect(get).toHaveBeenCalledWith('/api/marketplace/items/5?type=agent'))
    fireEvent.click(await screen.findByRole('button', { name: /add to workspace/i }))

    await waitFor(() => expect(post).toHaveBeenCalledWith('/api/marketplace/items/5/install'))
    expect(installPlaybook).not.toHaveBeenCalled()
  })
})

describe('marketplaceItemKey', () => {
  it('tells an agent and a playbook with the same id apart', () => {
    expect(marketplaceItemKey({ id: 5, type: 'agent' })).not.toBe(marketplaceItemKey({ id: 5, type: 'recipe' }))
  })
})
