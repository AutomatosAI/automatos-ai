// @vitest-environment node
/**
 * PRD-270 US-270-10 (§5.7, K23) — /marketplace's generateMetadata: a Marketplace link's share card.
 *
 * The page module is imported for real (its client switch stood in for), with a fresh module graph
 * per test so the per-process read cache starts empty. The edition is a getter over a hoisted
 * flag; the backend is the global fetch, answering the hosted edition's public routes in their
 * wire shapes (lib/marketplace/__tests__/public-fixtures.ts). Clerk and apiClient are spies that
 * must never be called: the card is read with no credential and no workspace. Time is Date.now,
 * stood in for where a test needs a read to have expired; the timeout test waits the real 1.5 s.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { AUTOMATOS_CLOUD_ORIGIN } from '@/lib/marketplace/cloud-files'
import {
  ARCADE_DIGEST,
  ARCADE_NEW,
  ARCADE_OLD,
  HUD,
  catalog,
  filesOf,
  publicAnswer,
  publicEntry,
} from '@/lib/marketplace/__tests__/public-fixtures'
import { MAX_DESCRIPTION_CHARS } from '@/lib/marketplace/share-card'
import {
  SHARE_MAX_PAGES,
  SHARE_READ_FAILED_KEPT_MS,
  SHARE_READ_KEPT_MS,
} from '@/lib/marketplace/share-card-read'

const edition = vi.hoisted(() => ({ saas: true }))
const request = vi.hoisted(() => ({ headers: new Headers() }))
const clerk = vi.hoisted(() => ({ auth: vi.fn() }))
const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), request: vi.fn(), getBaseUrl: vi.fn() }))

vi.mock('@/lib/auth-edition', () => ({
  get isSaaS() {
    return edition.saas
  },
  get isLocal() {
    return !edition.saas
  },
}))
vi.mock('next/headers', () => ({ headers: async () => request.headers }))
vi.mock('@clerk/nextjs/server', () => ({ auth: clerk.auth }))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('@/components/marketplace/public/marketplace-page', () => ({ MarketplacePage: () => null }))

vi.setConfig({ testTimeout: 30_000 })

const PAGE_ORIGIN = 'https://ui.automatos.app'
const BACKEND = 'http://backend.internal:8000'
const LOGO = `${PAGE_ORIGIN}/brand/automatos-mark-hi.png`
const ARCADE_COVER = `${AUTOMATOS_CLOUD_ORIGIN}${filesOf(ARCADE_DIGEST)}/assets/previews/default.png`
const CATALOG_URL = `${BACKEND}/api/marketplace/catalog`
const MARKETPLACE_CARD_TITLE = 'Automatos Marketplace'

type SearchParams = Record<string, string | string[] | undefined>
type Answer = { status: number; body: unknown }

const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>()

function answering(byUrl: Record<string, Answer>): void {
  fetchMock.mockImplementation(async (url) => {
    const answer = byUrl[url] ?? { status: 404, body: { detail: 'not found' } }
    return new Response(JSON.stringify(answer.body), { status: answer.status, headers: { 'Content-Type': 'application/json' } })
  })
}

function ok(body: unknown): Answer {
  return { status: 200, body }
}

function pageUrl(type: string, offset = 0): string {
  return `${BACKEND}/api/public/marketplace/${type}?limit=100&offset=${offset}`
}

async function metadataFor(params: SearchParams) {
  const { generateMetadata } = await import('@/app/marketplace/page')
  return generateMetadata({ searchParams: Promise.resolve(params) })
}

function requestedUrls(): string[] {
  return fetchMock.mock.calls.map(([url]) => url)
}

beforeEach(() => {
  vi.resetModules()
  edition.saas = true
  request.headers = new Headers({ host: 'ui.automatos.app', 'x-forwarded-proto': 'https' })
  vi.stubEnv('NEXT_PUBLIC_API_URL', AUTOMATOS_CLOUD_ORIGIN)
  vi.stubEnv('BACKEND_INTERNAL_URL', BACKEND)
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  fetchMock.mockReset()
  clerk.auth.mockReset()
  Object.values(api).forEach((spy) => spy.mockReset())
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('a theme link shows the theme, its cover from the public catalogue files route', () => {
  it("names the theme, describes it and pictures its newest version's cover on the API origin", async () => {
    answering({ [CATALOG_URL]: ok(catalog([ARCADE_NEW, ARCADE_OLD, HUD])) })

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.arcade', add: '1' })

    expect(metadata.title).toBe('Arcade · Automatos Marketplace')
    expect(metadata.description).toBe('Automatos as an arcade cabinet.')
    expect(metadata.openGraph).toMatchObject({
      title: 'Arcade',
      description: 'Automatos as an arcade cabinet.',
      url: `${PAGE_ORIGIN}/marketplace?tab=themes&item=automatos.arcade`,
      images: [{ url: ARCADE_COVER, alt: 'Arcade' }],
    })
    expect(metadata.twitter).toMatchObject({ card: 'summary_large_image', title: 'Arcade', images: [ARCADE_COVER] })
    expect(requestedUrls()).toEqual([CATALOG_URL])
  })

  it('pictures the logo for a theme whose cover is not a catalogue file on the API origin', async () => {
    const elsewhere = { ...HUD, cover: 'https://images.example.com/hud.png' }
    answering({ [CATALOG_URL]: ok(catalog([elsewhere])) })

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.hud' })

    expect(metadata.title).toBe('HUD · Automatos Marketplace')
    expect(metadata.twitter).toMatchObject({ images: [LOGO] })
  })
})

describe('any other link shows its item with the Automatos logo', () => {
  it("names an agent from its tab's public read and pictures the logo on the page's origin", async () => {
    answering({ [pageUrl('agents')]: ok(publicAnswer('agents', [publicEntry()])) })

    const metadata = await metadataFor({ tab: 'agents', item: 'research-assistant', add: '1' })

    expect(metadata.title).toBe('Research Assistant · Automatos Marketplace')
    expect(metadata.description).toBe('Finds, reads and summarises sources.')
    expect(metadata.openGraph).toMatchObject({ title: 'Research Assistant', images: [{ url: LOGO, alt: 'Research Assistant' }] })
    expect(metadata.twitter).toMatchObject({ card: 'summary_large_image', images: [LOGO] })
    expect(requestedUrls()).toEqual([pageUrl('agents')])
  })

  it('reads the next page while the item is not on this one, and stops at the last', async () => {
    answering({
      [pageUrl('skills')]: ok(publicAnswer('skills', [publicEntry({ key: 'pdf', name: 'PDF' })], true)),
      [pageUrl('skills', 1)]: ok(publicAnswer('skills', [publicEntry({ key: 'docx', name: 'Word' })], false, 1)),
    })

    const metadata = await metadataFor({ tab: 'skills', item: 'docx' })

    expect(metadata.title).toBe('Word · Automatos Marketplace')
    expect(requestedUrls()).toEqual([pageUrl('skills'), pageUrl('skills', 1)])
  })

  it('stops after SHARE_MAX_PAGES pages without the item, with the Marketplace card', async () => {
    const pages = Array.from({ length: SHARE_MAX_PAGES + 1 }, (_, offset): [string, Answer] => [
      pageUrl('skills', offset),
      ok(publicAnswer('skills', [publicEntry({ key: `skill-${offset}`, name: `Skill ${offset}` })], true, offset)),
    ])
    answering(Object.fromEntries(pages))

    const metadata = await metadataFor({ tab: 'skills', item: `skill-${SHARE_MAX_PAGES}` })

    expect(metadata.title).toBe(MARKETPLACE_CARD_TITLE)
    expect(requestedUrls()).toEqual(pages.slice(0, SHARE_MAX_PAGES).map(([url]) => url))
  })

  it('cuts a long description to MAX_DESCRIPTION_CHARS, ending in an ellipsis', async () => {
    const long = 'word '.repeat(100)
    answering({ [pageUrl('agents')]: ok(publicAnswer('agents', [publicEntry({ description: long })])) })

    const metadata = await metadataFor({ tab: 'agents', item: 'research-assistant' })

    expect(metadata.description).toHaveLength(MAX_DESCRIPTION_CHARS)
    expect(metadata.description?.endsWith('word…')).toBe(true)
    expect(metadata.openGraph?.description).toBe(metadata.description)
  })

  it('describes an item that has no description by its name', async () => {
    answering({ [pageUrl('llms')]: ok(publicAnswer('llms', [publicEntry({ key: 'gpt', name: 'GPT', description: null })])) })

    const metadata = await metadataFor({ tab: 'llms', item: 'gpt' })

    expect(metadata.description).toBe('GPT on the Automatos Marketplace.')
  })
})

describe('one canonical URL per item', () => {
  it('rebuilds it from the tab value and the key only: a label, add and other parameters are dropped', async () => {
    answering({ [pageUrl('playbooks')]: ok(publicAnswer('playbooks', [publicEntry({ key: 'daily-brief', name: 'Daily brief' })])) })

    const metadata = await metadataFor({ tab: 'Playbooks', item: 'daily-brief', add: '1', utm_source: 'x' })

    const canonical = `${PAGE_ORIGIN}/marketplace?tab=recipes&item=daily-brief`
    expect(metadata.alternates?.canonical).toBe(canonical)
    expect(metadata.openGraph).toMatchObject({ url: canonical })
  })

  it('takes the first value of a repeated parameter, and http from X-Forwarded-Proto', async () => {
    request.headers = new Headers({ host: 'localhost:3000', 'x-forwarded-proto': 'http' })
    answering({ [CATALOG_URL]: ok(catalog([HUD])) })

    const metadata = await metadataFor({ tab: 'themes', item: ['automatos.hud', 'automatos.arcade'] })

    expect(metadata.alternates?.canonical).toBe('http://localhost:3000/marketplace?tab=themes&item=automatos.hud')
  })

  it('reads an unknown tab as Themes, the tab the Marketplace opens on, and a Capabilities link by its sub-tab', async () => {
    answering({
      [CATALOG_URL]: ok(catalog([HUD])),
      [pageUrl('plugins')]: ok(publicAnswer('plugins', [publicEntry({ key: 'github', name: 'GitHub' })])),
    })

    const unknown = await metadataFor({ tab: 'bogus', item: 'automatos.hud' })
    const capabilities = await metadataFor({ tab: 'capabilities', item: 'github' })

    expect(unknown.alternates?.canonical).toBe(`${PAGE_ORIGIN}/marketplace?tab=themes&item=automatos.hud`)
    expect(capabilities.alternates?.canonical).toBe(`${PAGE_ORIGIN}/marketplace?tab=plugins&item=github`)
  })

  it('carries no URL when Host is not a plain host name', async () => {
    request.headers = new Headers({ host: 'evil.example/x?y' })
    answering({ [CATALOG_URL]: ok(catalog([HUD])) })

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.hud' })

    expect(metadata.title).toBe('HUD · Automatos Marketplace')
    expect(metadata.alternates).toBeUndefined()
    expect(metadata.openGraph).not.toHaveProperty('url')
  })
})

describe("the Marketplace's own card", () => {
  const MARKETPLACE_CARD = {
    title: MARKETPLACE_CARD_TITLE,
    alternates: { canonical: `${PAGE_ORIGIN}/marketplace` },
    openGraph: { title: MARKETPLACE_CARD_TITLE, url: `${PAGE_ORIGIN}/marketplace`, images: [{ url: LOGO }] },
    twitter: { card: 'summary_large_image', images: [LOGO] },
  }

  it('shows without an item, and reads nothing', async () => {
    const metadata = await metadataFor({ tab: 'agents' })

    expect(metadata).toMatchObject(MARKETPLACE_CARD)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([
    ['the backend cannot be reached', () => fetchMock.mockRejectedValue(new TypeError('fetch failed'))],
    ['the route refuses (429)', () => answering({ [CATALOG_URL]: { status: 429, body: { detail: 'slow down' } } })],
    ['the answer is not JSON', () => fetchMock.mockResolvedValue(new Response('<html>', { status: 200 }))],
    ['the catalogue does not list the item', () => answering({ [CATALOG_URL]: ok(catalog([HUD])) })],
  ])('shows when %s', async (_case, arrange) => {
    arrange()

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.arcade' })

    expect(metadata).toMatchObject(MARKETPLACE_CARD)
  })
})

describe('the read is public: no credential, no workspace, no redirect, kept for a while', () => {
  it('gives up after the timeout, with the Marketplace card', async () => {
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true })
        }),
    )

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.arcade' })

    expect(metadata.title).toBe(MARKETPLACE_CARD_TITLE)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('asks only the public routes, with no Authorization or X-Workspace-ID, never through Clerk or apiClient', async () => {
    answering({ [pageUrl('agents')]: ok(publicAnswer('agents', [publicEntry()])) })

    await metadataFor({ tab: 'agents', item: 'research-assistant' })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url.startsWith(`${BACKEND}/api/public/marketplace/`)).toBe(true)
    const sent = new Headers(init?.headers)
    expect(sent.has('authorization')).toBe(false)
    expect(sent.has('x-workspace-id')).toBe(false)
    expect(init?.credentials).toBeUndefined()
    expect(init?.redirect).toBe('manual')
    expect(init?.signal).toBeInstanceOf(AbortSignal)
    expect(clerk.auth).not.toHaveBeenCalled()
    Object.values(api).forEach((spy) => expect(spy).not.toHaveBeenCalled())
  })

  it('reads the same route once for two cards in a row, and again once the answer is SHARE_READ_KEPT_MS old', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(1_000_000)
    answering({ [CATALOG_URL]: ok(catalog([ARCADE_NEW, HUD])) })

    await metadataFor({ tab: 'themes', item: 'automatos.arcade' })
    const second = await metadataFor({ tab: 'themes', item: 'automatos.hud' })
    clock.mockReturnValue(1_000_000 + SHARE_READ_KEPT_MS + 1)
    await metadataFor({ tab: 'themes', item: 'automatos.hud' })

    expect(second.title).toBe('HUD · Automatos Marketplace')
    expect(requestedUrls()).toEqual([CATALOG_URL, CATALOG_URL])
  })

  it('joins a read already under way: two cards at once ask once', async () => {
    answering({ [CATALOG_URL]: ok(catalog([ARCADE_NEW, HUD])) })

    const [arcade, hud] = await Promise.all([
      metadataFor({ tab: 'themes', item: 'automatos.arcade' }),
      metadataFor({ tab: 'themes', item: 'automatos.hud' }),
    ])

    expect([arcade.title, hud.title]).toEqual(['Arcade · Automatos Marketplace', 'HUD · Automatos Marketplace'])
    expect(requestedUrls()).toEqual([CATALOG_URL])
  })

  it('keeps a failed read for SHARE_READ_FAILED_KEPT_MS, then asks again', async () => {
    const clock = vi.spyOn(Date, 'now').mockReturnValue(1_000_000)
    answering({ [CATALOG_URL]: { status: 429, body: { detail: 'slow down' } } })

    await metadataFor({ tab: 'themes', item: 'automatos.arcade' })
    await metadataFor({ tab: 'themes', item: 'automatos.arcade' })
    expect(requestedUrls()).toEqual([CATALOG_URL])

    clock.mockReturnValue(1_000_000 + SHARE_READ_FAILED_KEPT_MS + 1)
    answering({ [CATALOG_URL]: ok(catalog([ARCADE_NEW])) })
    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.arcade' })

    expect(metadata.title).toBe('Arcade · Automatos Marketplace')
    expect(requestedUrls()).toEqual([CATALOG_URL, CATALOG_URL])
  })
})

describe('the local edition', () => {
  it('has the title only and noindex, and reads nothing', async () => {
    edition.saas = false

    const metadata = await metadataFor({ tab: 'themes', item: 'automatos.arcade' })

    expect(metadata).toEqual({ title: MARKETPLACE_CARD_TITLE, robots: { index: false, follow: false } })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
