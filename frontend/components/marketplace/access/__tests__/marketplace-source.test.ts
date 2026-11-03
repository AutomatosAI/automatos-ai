/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the Marketplace seam.
 *
 * - the public side reads only the public routes: every page of a tab's read (its keys once, at
 *   most MAX_PUBLIC_PAGES pages), the Featured read and the counts, each asked for once however
 *   many callers read it, and maps them to the member shapes with nothing made up;
 * - the member side reads exactly today's routes, as the homepage and its tabs read them before.
 */
import { QueryClient } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  get: vi.fn(),
  getMarketplaceItems: vi.fn(),
  getFeaturedMarketplaceItems: vi.fn(),
}))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

import { publicAnswer, publicEntry } from '@/lib/marketplace/__tests__/public-fixtures'
import { MAX_PUBLIC_PAGES, readPublicCounts, readPublicEntries, readPublicFeatured } from '@/lib/marketplace/source/public-reads'

import { publicAnswerFor } from '../../public/__tests__/one-page-fixtures'
import { MEMBER_SOURCE } from '../member-source'
import { NOT_LISTED, createPublicSource } from '../public-source'

const page = (offset: number) => `/api/public/marketplace/skills?limit=100&offset=${offset}`

beforeEach(() => {
  api.get.mockReset()
  api.getMarketplaceItems.mockReset()
  api.getFeaturedMarketplaceItems.mockReset()
})

describe('the public reads', () => {
  it('reads every page of a tab from where the last one ended, each key once', async () => {
    api.get.mockImplementation((url: string) =>
      Promise.resolve(
        url === page(0)
          ? publicAnswer('skills', [publicEntry({ key: 'a', name: 'A' }), publicEntry({ key: 'b', name: 'B' })], true)
          : publicAnswer('skills', [publicEntry({ key: 'b', name: 'B again' }), publicEntry({ key: 'c', name: 'C' })], false, 2),
      ),
    )

    const entries = await readPublicEntries('skills')

    expect(api.get.mock.calls.map(([url]) => url)).toEqual([page(0), page(2)])
    expect(entries.map((entry) => [entry.item.key, entry.item.name])).toEqual([['a', 'A'], ['b', 'B'], ['c', 'C']])
    expect(entries[0].raw.key).toBe('a')
  })

  it('stops after MAX_PUBLIC_PAGES pages, and drops an item with no key or name', async () => {
    let next = 0
    api.get.mockImplementation(() => {
      next += 1
      return Promise.resolve(publicAnswer('skills', [publicEntry({ key: `k${next}` }), publicEntry({ key: '' })], true))
    })

    const entries = await readPublicEntries('skills')

    expect(api.get).toHaveBeenCalledTimes(MAX_PUBLIC_PAGES)
    expect(entries).toHaveLength(MAX_PUBLIC_PAGES)
  })

  it('an answer that isn’t the type’s page reads as an empty one', async () => {
    api.get.mockResolvedValue(publicAnswer('agents', [publicEntry()]))
    expect(await readPublicEntries('skills')).toEqual([])
  })

  it('reads the Featured and counts routes, and an answer that isn’t one as nothing', async () => {
    api.get.mockImplementation((url: string) => publicAnswerFor(url))
    expect((await readPublicFeatured()).featured).toHaveLength(2)
    expect(await readPublicCounts()).toEqual({ items: 100, installs: 113 })
    expect(api.get.mock.calls.map(([url]) => url)).toEqual(['/api/public/marketplace/featured', '/api/public/marketplace/counts'])

    api.get.mockResolvedValue(['not', 'an', 'answer'])
    expect(await readPublicFeatured()).toEqual({ featured: [], popular: [] })
    expect(await readPublicCounts()).toEqual({ items: 0, installs: 0 })
  })
})

describe('the seam’s public side', () => {
  const source = () => createPublicSource(new QueryClient({ defaultOptions: { queries: { retry: false } } }))

  beforeEach(() => {
    api.get.mockImplementation((url: string) => publicAnswerFor(url))
  })

  it('lists agents and Playbooks as member items, filtered by type and search on the client, each read once', async () => {
    const seam = source()

    const writers = await seam.items({ type: 'agent', search: 'drafts' })
    const all = await seam.items()

    expect(writers).toHaveLength(1)
    expect(writers[0]).toMatchObject({ id: 8, type: 'agent', name: 'Writer', creator_name: 'Automatos', install_count: 0 })
    expect(writers[0].metadata).toEqual({ tool_names: [], tool_icons: [] })
    expect(all.map((item) => [item.type, item.id])).toEqual([['agent', 7], ['agent', 8], ['recipe', 12]])
    expect(api.get.mock.calls.map(([url]) => url)).toEqual([
      '/api/public/marketplace/agents?limit=100&offset=0',
      '/api/public/marketplace/playbooks?limit=100&offset=0',
    ])
    expect(api.getMarketplaceItems).not.toHaveBeenCalled()
  })

  it('answers one item for the item dialog, and an unlisted one is an error the dialog shows', async () => {
    const seam = source()

    const playbook = await seam.item(12, 'recipe')
    expect(playbook).toMatchObject({ name: 'Weekly report', marketplace_category: 'Reporting' })
    expect(playbook.metadata).toEqual({ required_tools: ['gmail'] })
    expect(playbook).not.toHaveProperty('steps')
    await expect(seam.item(99, 'agent')).rejects.toThrow(NOT_LISTED)
    await expect(seam.item(7, 'skill')).rejects.toThrow(NOT_LISTED)
  })

  it('answers Featured, Popular and the counts from their routes, with their versions and installs', async () => {
    const seam = source()

    const [hero] = await seam.featured(1)
    const popular = await seam.popular([])

    expect(hero).toMatchObject({ id: 7, type: 'agent', name: 'Research Assistant', version: '1.0.0', install_count: 40 })
    expect(popular).toEqual([expect.objectContaining({ id: 12, type: 'recipe', install_count: 61 })])
    expect(await seam.counts()).toEqual({ totalItems: 100, totalInstalls: 113 })
    expect(api.get.mock.calls.filter(([url]) => url === '/api/public/marketplace/featured')).toHaveLength(1)
  })

  it('lists packages with their members and apps, and nothing it doesn’t know', async () => {
    const [pkg] = await source().packages()

    expect(pkg).toEqual({
      id: 'socials',
      slug: 'socials',
      name: 'Socials',
      description: 'Finds, reads and summarises sources.',
      vertical_tags: ['research', 'web'],
      matching: {},
      members: [{ type: 'agent', name: 'Social Writer' }],
      setup_manifest: { required_connects: [{ app_name: 'instagram' }] },
      showcase: false,
    })
  })
})

describe('the seam’s member side: today’s routes, unchanged', () => {
  it('counts the first 100 items and adds up their installs', async () => {
    api.get.mockResolvedValue([{ type: 'agent', id: 1, install_count: 3 }, { type: 'recipe', id: 1, install_count: 4 }])

    expect(await MEMBER_SOURCE.counts()).toEqual({ totalItems: 2, totalInstalls: 7 })
    expect(api.get).toHaveBeenCalledWith('/api/marketplace/items?limit=100')
  })

  it('recommends the most installed of the first 20 that aren’t featured, six at most', async () => {
    const items = Array.from({ length: 9 }, (_, index) => ({ type: 'agent', id: index + 1, install_count: index }))
    api.get.mockResolvedValue(items)

    const recommended = await MEMBER_SOURCE.popular([items[8] as never])

    expect(api.get).toHaveBeenCalledWith('/api/marketplace/items?limit=20')
    expect(recommended.map((item) => item.id)).toEqual([8, 7, 6, 5, 4, 3])
  })

  it('reads items, one item, Featured and packages through the member routes', async () => {
    api.get.mockResolvedValue([])
    await MEMBER_SOURCE.items({ type: 'agent', limit: 100 })
    await MEMBER_SOURCE.featured(5)
    await MEMBER_SOURCE.item(7, 'recipe')
    await MEMBER_SOURCE.packages()

    expect(api.getMarketplaceItems).toHaveBeenCalledWith({ type: 'agent', limit: 100 })
    expect(api.getFeaturedMarketplaceItems).toHaveBeenCalledWith(5)
    expect(api.get.mock.calls.map(([url]) => url)).toEqual(['/api/marketplace/items/7?type=recipe', '/api/marketplace/packages'])
  })
})
