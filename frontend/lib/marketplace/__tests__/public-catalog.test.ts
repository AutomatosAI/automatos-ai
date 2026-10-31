/**
 * PRD-270 US-270-08b (K20) — the visitor's reads and their mapping.
 *
 * - the hosted catalogue as the storefront's listings: one per theme, its newest version first,
 *   every version in `versions`, nothing workspace-scoped (no install, no revocation, `source` null);
 * - path resolution: a path on the catalogue files route becomes a URL on apiClient.getBaseUrl(),
 *   which themeFileUrl draws; anything else is none;
 * - a malformed item is dropped and a malformed answer lists none, never thrown;
 * - one page of a public read: items mapped field by field, facts and includes per type, the next
 *   page from `more`, the client-side search;
 * - both reads go through apiClient.get on the public routes only.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), getBaseUrl: vi.fn() }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))

import { themeFileUrl } from '@/components/marketplace/themes/listing-format'
import { storefrontViews } from '@/components/marketplace/themes/storefront/storefront-views'

import {
  catalogFileUrl,
  catalogListings,
  catalogVersion,
  getPublicPage,
  getPublicThemes,
} from '../public-catalog'
import { CATALOG_ENDPOINT, PUBLIC_MARKETPLACE_ENDPOINT, filterPublicItems, publicItem, publicPage } from '../public-items'
import type { PublicItem } from '../public-items'
import {
  ARCADE_DIGEST,
  ARCADE_NEW,
  ARCADE_OLD,
  ARCADE_OLD_DIGEST,
  BASE,
  HUD,
  catalog,
  catalogItem,
  filesOf,
  publicAnswer,
  publicEntry,
} from './public-fixtures'

const ARCADE_FILES = `${BASE}${filesOf(ARCADE_DIGEST)}`

beforeEach(() => {
  api.get.mockReset()
  api.getBaseUrl.mockReset()
  api.getBaseUrl.mockReturnValue(BASE)
})

describe('catalogListings: the catalogue as the storefront lists it', () => {
  it('lists one listing per theme in the catalogue order, its newest version first, every version in versions', () => {
    const listings = catalogListings(catalog([ARCADE_NEW, ARCADE_OLD, HUD]), BASE)

    expect(listings.map((listing) => listing.listingId)).toEqual(['automatos.arcade', 'automatos.hud'])
    const [arcade] = listings
    expect(arcade).toMatchObject({
      listingId: 'automatos.arcade',
      themeId: 'automatos/arcade',
      name: 'Arcade',
      description: 'Automatos as an arcade cabinet.',
      version: '1.1.0',
      latest: '1.1.0',
      packageId: ARCADE_DIGEST,
      publisher: { name: 'automatos', official: true },
      signature: { state: 'verified', keyId: 'automatos-2026' },
      compatible: true,
      uses: ['frame', 'looks'],
      pricing: 'Free',
    })
    expect(Object.keys(arcade.versions)).toEqual(['1.1.0', '1.0.0'])
    expect(arcade.versions['1.0.0']).toEqual({
      packageId: ARCADE_OLD_DIGEST, engines: { automatos: '^0.9.0' }, compatible: true, revoked: null,
      signature: { state: 'verified', keyId: 'automatos-2026' }, createdAt: null,
    })
  })

  it('makes up nothing workspace-scoped: no install, no revocation, the hosted edition’s own row, no documents', () => {
    const [arcade] = catalogListings(catalog([ARCADE_NEW]), BASE)

    expect(arcade.installed).toBeNull()
    expect(arcade.revoked).toBeNull()
    expect(arcade.source).toBeNull()
    expect(arcade.readme).toBeNull()
    expect(arcade.changelog).toBeNull()
  })

  it('reads the layout facts from theme.json (themeJson, base64)', () => {
    const [arcade] = catalogListings(catalog([ARCADE_NEW]), BASE)

    expect(arcade).toMatchObject({ base: 'studio', tone: 'dark', minWidth: 1024, pages: ['/chat', '/settings'], routes: ['arcade'] })
  })

  it('shows no layout facts when themeJson is not base64 JSON, and keeps the listing', () => {
    const [arcade] = catalogListings(catalog([catalogItem({ themeJson: '***not base64***' })]), BASE)

    expect(arcade.listingId).toBe('automatos.arcade')
    expect(arcade).toMatchObject({ base: null, tone: null, minWidth: null, pages: [], routes: [] })
  })
})

describe('path resolution', () => {
  it('resolves every catalogue path against the API base, and themeFileUrl draws each one', () => {
    const [arcade] = catalogListings(catalog([ARCADE_NEW]), BASE)

    expect(arcade.cover).toBe(`${ARCADE_FILES}/assets/previews/default.png`)
    expect(arcade.phoneCover).toBe(`${ARCADE_FILES}/assets/previews/phone/default.png`)
    expect(arcade.presets).toEqual([
      {
        id: 'default',
        default: true,
        previews: [{ screen: null, url: `${ARCADE_FILES}/assets/previews/default.png` }],
        phonePreviews: [{ screen: null, url: `${ARCADE_FILES}/assets/previews/phone/default.png` }],
      },
    ])
    expect(themeFileUrl(arcade.cover)).toBe(arcade.cover)
    expect(themeFileUrl(arcade.phoneCover)).toBe(arcade.phoneCover)
  })

  it('gives the storefront hero its web and phone screens from the catalogue', () => {
    const [arcade] = catalogListings(catalog([ARCADE_NEW]), BASE)

    expect(storefrontViews(arcade)).toEqual([
      {
        key: 'default:',
        label: 'Default',
        web: `${ARCADE_FILES}/assets/previews/default.png`,
        phone: `${ARCADE_FILES}/assets/previews/phone/default.png`,
      },
    ])
  })

  it.each([
    ['another origin', 'https://evil.example/api/marketplace/catalog/files/x/a.png'],
    ['protocol-relative', '//evil.example/a.png'],
    ['the local files route', `/api/themes/files/${ARCADE_DIGEST}/a.png`],
    ['a climb', `${filesOf(ARCADE_DIGEST)}/../../../api/me`],
    ['a backslash', `${filesOf(ARCADE_DIGEST)}\\a.png`],
    ['whitespace', `${filesOf(ARCADE_DIGEST)}/a b.png`],
    ['not a string', 42],
  ])('drops %s', (_name, path) => {
    expect(catalogFileUrl(path, BASE)).toBeNull()
  })

  it('drops a preview off the files route and keeps the rest of the listing', () => {
    const files = filesOf(ARCADE_DIGEST)
    const item = catalogItem({
      cover: 'https://evil.example/cover.png',
      presets: [{ id: 'default', default: true, previews: [{ screen: null, url: '//evil.example/x.png' }, { screen: 'chat', url: `${files}/assets/previews/chat.png` }] }],
      otherPreviews: ['javascript:alert(1)', `${files}/assets/previews/extra.png`],
    })
    const [arcade] = catalogListings(catalog([item]), BASE)

    expect(arcade.cover).toBeNull()
    expect(arcade.presets[0].previews).toEqual([{ screen: 'chat', url: `${BASE}${files}/assets/previews/chat.png` }])
    expect(arcade.otherPreviews).toEqual([`${BASE}${files}/assets/previews/extra.png`])
  })
})

describe('malformed answers are dropped, never thrown', () => {
  it.each([
    ['not an object', 'theme'],
    ['another type', catalogItem({ type: 'agent' })],
    ['no id', catalogItem({ id: undefined })],
    ['no listing id', catalogItem({ listingId: '' })],
    ['no version', catalogItem({ version: null })],
    ['a digest that is not one', catalogItem({ digest: 'not-a-digest' })],
    ['no publisher', catalogItem({ publisher: null })],
    ['a publisher with no name', catalogItem({ publisher: { official: true } })],
  ])('drops an item with %s, and lists the rest', (_name, bad) => {
    expect(catalogVersion(bad, BASE)).toBeNull()
    expect(catalogListings(catalog([bad, HUD]), BASE).map((listing) => listing.listingId)).toEqual(['automatos.hud'])
  })

  it.each([
    ['null', null],
    ['a string', 'catalog'],
    ['another schema', { schema: 2, items: [HUD] }],
    ['items that are not a list', { schema: 1, items: { HUD } }],
  ])('lists none for %s', (_name, answer) => {
    expect(catalogListings(answer, BASE)).toEqual([])
  })

  it('keeps only the strings of a list, and reads an unknown pricing as Free', () => {
    const [arcade] = catalogListings(catalog([catalogItem({ uses: ['frame', 7, null, ' '], pricing: 3 })]), BASE)

    expect(arcade.uses).toEqual(['frame'])
    expect(arcade.pricing).toBe('Free')
  })
})

describe('publicPage: one page of a public read', () => {
  it('maps each item field by field and drops one with no key or name', () => {
    const page = publicPage(
      publicAnswer('agents', [publicEntry(), publicEntry({ key: '' }), publicEntry({ name: null }), 'agent']),
      'agents',
      0,
    )

    expect(page.items).toEqual([
      {
        key: 'research-assistant', name: 'Research Assistant', description: 'Finds, reads and summarises sources.',
        category: 'Research', icon: '🔎', tags: ['research', 'web'], publisher: 'Automatos', requiredApps: ['notion'],
        facts: [], includes: [],
      },
    ])
    expect(page.next).toBeNull()
  })

  it('answers the next offset while more is true', () => {
    const page = publicPage(publicAnswer('agents', [publicEntry(), publicEntry({ key: 'b', name: 'B' })], true, 100), 'agents', 100)

    expect(page.next).toBe(102)
  })

  it('reads an answer for another type, or not a page at all, as an empty last page', () => {
    expect(publicPage(publicAnswer('playbooks', [publicEntry()]), 'agents', 0)).toEqual({ items: [], next: null })
    expect(publicPage(null, 'agents', 0)).toEqual({ items: [], next: null })
  })

  it('gives each type its own facts, and a package its members, an LLM its capabilities', () => {
    const app = publicItem(publicEntry({ actionCount: 12, triggerCount: 0 }), 'applications')
    const plugin = publicItem(publicEntry({ version: '2.1.0', skillsCount: 3, hooksCount: -1 }), 'plugins')
    const llm = publicItem(
      publicEntry({
        provider: 'anthropic', vendor: 'anthropic', contextWindow: 200000, priceTier: 'high',
        capabilities: [{ name: 'reasoning', rating: 'excellent' }, { name: 'vision' }, 'x'],
      }),
      'llms',
    )
    const pack = publicItem(
      publicEntry({ members: [{ type: 'agent', name: 'Researcher' }, { type: 'playbook', name: null }, { type: 'tool', name: 'Notion' }] }),
      'packages',
    )

    expect(app?.facts).toEqual([{ label: 'Actions', value: '12' }])
    expect(plugin?.facts).toEqual([{ label: 'Version', value: '2.1.0' }, { label: 'Skills', value: '3' }])
    expect(llm?.facts).toEqual([
      { label: 'Served by', value: 'anthropic' }, { label: 'Made by', value: 'anthropic' },
      { label: 'Context window', value: '200,000' }, { label: 'Price tier', value: 'high' },
    ])
    expect(llm?.includes).toEqual(['reasoning · excellent', 'vision'])
    expect(pack?.includes).toEqual(['Agent · Researcher', 'Playbook', 'Application · Notion'])
  })

  it.each([
    ['an https logo', 'https://logos.composio.dev/api/notion', 'https://logos.composio.dev/api/notion'],
    ['an emoji', '🤖', '🤖'],
    ['an emoji joined from several', '🧑‍💻', '🧑‍💻'],
    ['an icon name', 'bot', null],
    ['an http URL', 'http://example.com/logo.png', null],
    ['a javascript: URL', 'javascript:alert(1)', null],
    ['a long text', '🤖🤖🤖🤖🤖🤖🤖🤖🤖', null],
  ])('reads %s as its icon', (_name, icon, expected) => {
    expect(publicItem(publicEntry({ icon }), 'agents')?.icon).toBe(expected)
  })

  it('filters on the client: every word of the search, in the name, description, category, publisher, tags or apps', () => {
    const items = [publicEntry(), publicEntry({ key: 'w', name: 'Writer', description: 'Drafts posts.', category: 'Content', tags: [], requiredApps: ['linkedin'] })]
      .map((entry) => publicItem(entry, 'agents'))
      .filter((item): item is PublicItem => item !== null)

    expect(filterPublicItems(items, '').map((item) => item.key)).toEqual(['research-assistant', 'w'])
    expect(filterPublicItems(items, 'LinkedIn drafts').map((item) => item.key)).toEqual(['w'])
    expect(filterPublicItems(items, 'notion web').map((item) => item.key)).toEqual(['research-assistant'])
    expect(filterPublicItems(items, 'nothing like it')).toEqual([])
  })
})

describe('the reads go through apiClient.get on the public routes', () => {
  it('getPublicThemes reads the catalogue and resolves its paths against the API base', async () => {
    api.get.mockResolvedValue(catalog([ARCADE_NEW]))

    const [arcade] = await getPublicThemes()

    expect(api.get).toHaveBeenCalledTimes(1)
    expect(api.get).toHaveBeenCalledWith(CATALOG_ENDPOINT)
    expect(CATALOG_ENDPOINT).toBe('/api/marketplace/catalog')
    expect(arcade.cover).toBe(`${ARCADE_FILES}/assets/previews/default.png`)
  })

  it('getPublicPage reads one page of the type from its offset', async () => {
    api.get.mockResolvedValue(publicAnswer('skills', [publicEntry({ key: 'pdf', name: 'PDF', version: '1.0.0' })], false, 100))

    const page = await getPublicPage('skills', 100)

    expect(api.get).toHaveBeenCalledWith(`${PUBLIC_MARKETPLACE_ENDPOINT}/skills?limit=100&offset=100`)
    expect(PUBLIC_MARKETPLACE_ENDPOINT).toBe('/api/public/marketplace')
    expect(page.items.map((item) => item.key)).toEqual(['pdf'])
    expect(page.items[0].facts).toEqual([{ label: 'Version', value: '1.0.0' }])
  })

  it('lets a refusal through, so the tab shows the server’s sentence', async () => {
    const refusal = Object.assign(new Error('Too many requests'), { body: { detail: 'Too many requests. Try again in a minute.' } })
    api.get.mockRejectedValue(refusal)

    await expect(getPublicThemes()).rejects.toBe(refusal)
  })
})
