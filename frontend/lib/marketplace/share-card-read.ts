/**
 * PRD-270 US-270-10 (K23) — the server-side read behind a Marketplace link's card, and the
 * /marketplace page's metadata (app/marketplace/page.tsx `generateMetadata`).
 *
 * Why a server-side fetch and not apiClient: metadata is rendered on the server, before any page
 * code, and apiClient is the browser's client. The read asks the hosted edition's PUBLIC routes
 * only, with no credential and no workspace: the catalogue (`GET /api/marketplace/catalog`) for a
 * theme, one page after another of the tab's public read (`GET /api/public/marketplace/{type}`)
 * for anything else, up to SHARE_MAX_PAGES pages. It goes to the backend address the theme files
 * proxy uses (app/api/themes/files/[...path]/route.ts), never follows a redirect, and has
 * SHARE_READ_TIMEOUT_MS in all: a crawler waits for the card, so a slow read gives the
 * Marketplace's own card instead. Any failure (an error, a refusal, a timeout, an answer that
 * isn't JSON) is logged and gives that card too; nothing here throws.
 *
 * The routes count requests per IP, and every card this server renders asks from the same
 * address, so each read is shared in this process: a read already under way is joined, a good
 * answer is kept for SHARE_READ_KEPT_MS (the routes' own `max-age`) and a failed one for
 * SHARE_READ_FAILED_KEPT_MS, so a refusing route isn't asked again by every crawler at once.
 */
import type { Metadata } from 'next'

import { publicReadOfTab } from '@/components/marketplace/category-tabs'
import { marketplaceLink } from '@/components/marketplace/deep-link/deep-link-params'
import type { MarketplaceLink } from '@/components/marketplace/deep-link/deep-link-params'
import { THEMES_TAB_VALUE } from '@/components/marketplace/themes/listing-format'
import { isSaaS } from '@/lib/auth-edition'

import { CATALOG_ENDPOINT, publicPage, publicPagePath } from './public-items'
import type { PublicMarketplaceType } from './public-items'
import {
  LOCAL_MARKETPLACE_METADATA,
  catalogSharedItem,
  itemCard,
  marketplaceCard,
  requestOrigin,
} from './share-card'
import type { HeaderReader, SharedItem } from './share-card'

/** The time every read behind one card may take together. */
export const SHARE_READ_TIMEOUT_MS = 1_500
/** How many pages of a public read are searched for an item. */
export const SHARE_MAX_PAGES = 5
/** How long a good answer is kept: the public routes' `Cache-Control: max-age=300`. */
export const SHARE_READ_KEPT_MS = 300_000
/** How long a failed read is kept before the route is asked again. */
export const SHARE_READ_FAILED_KEPT_MS = 15_000
export const SHARE_READ_CACHE_MAX_ENTRIES = 200

const LOG_PREFIX = '[marketplace-share]'
const OK = 200

/** Next's searchParams: a value, several, or none. */
export type PageSearchParams = Readonly<Record<string, string | string[] | undefined>>

type Read = (path: string) => Promise<unknown>

interface Kept {
  /** The answer, or null for a failed read; under way until it settles. */
  readonly answer: Promise<unknown>
  readonly expiresAt: number
}

const kept = new Map<string, Kept>()

function backendUrl(): string {
  // Server-side: BACKEND_INTERNAL_URL (container DNS) beats the browser-facing NEXT_PUBLIC_API_URL,
  // as the theme files proxy reads it (PRD-209). Unset in SaaS ⇒ the public API.
  return process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
}

function keep(path: string, answer: Promise<unknown>, expiresAt: number): void {
  kept.delete(path)
  if (kept.size >= SHARE_READ_CACHE_MAX_ENTRIES) {
    const oldest = kept.keys().next().value
    if (oldest !== undefined) kept.delete(oldest)
  }
  kept.set(path, { answer, expiresAt })
}

/** One public route's JSON answer, or null (logged) for anything but a 200 with JSON in time. */
async function fetchPublic(path: string, signal: AbortSignal): Promise<unknown> {
  try {
    const response = await fetch(`${backendUrl()}${path}`, {
      headers: { Accept: 'application/json' },
      cache: 'no-store',
      redirect: 'manual',
      signal,
    })
    if (response.status === OK) return await response.json()
    await response.body?.cancel()
    console.warn(LOG_PREFIX, 'a public read answered', response.status)
    return null
  } catch (error) {
    console.warn(LOG_PREFIX, 'a public read failed', error)
    return null
  }
}

/** One public route's answer, shared: a kept or under-way read, else a new one, kept by how it ends. */
function readPublic(path: string, signal: AbortSignal): Promise<unknown> {
  const now = Date.now()
  const hit = kept.get(path)
  if (hit && hit.expiresAt > now) return hit.answer
  const answer = fetchPublic(path, signal)
  keep(path, answer, now + SHARE_READ_KEPT_MS)
  void answer.then((value) => {
    const failed = value === null && kept.get(path)?.answer === answer
    if (failed) keep(path, answer, Date.now() + SHARE_READ_FAILED_KEPT_MS)
  })
  return answer
}

/** The item `key` in the public read of `type`, searched from `offset`, or null. */
async function publicSharedItem(read: Read, type: PublicMarketplaceType, key: string, offset = 0, pages = 0): Promise<SharedItem | null> {
  if (pages >= SHARE_MAX_PAGES) return null
  const answer = await read(publicPagePath(type, offset))
  if (answer === null) return null
  const page = publicPage(answer, type, offset)
  const item = page.items.find((entry) => entry.key === key)
  if (item) return { name: item.name, description: item.description, image: null }
  return page.next === null ? null : publicSharedItem(read, type, key, page.next, pages + 1)
}

/** The item a link names, as a card, or null when it can't be read in time or isn't listed. */
export async function readSharedItem(link: Pick<MarketplaceLink, 'tab' | 'item'>): Promise<SharedItem | null> {
  const signal = AbortSignal.timeout(SHARE_READ_TIMEOUT_MS)
  const read: Read = (path) => readPublic(path, signal)
  try {
    if (link.tab === THEMES_TAB_VALUE) return catalogSharedItem(await read(CATALOG_ENDPOINT), link.item)
    const type = publicReadOfTab(link.tab)
    return type ? await publicSharedItem(read, type, link.item) : null
  } catch (error) {
    console.warn(LOG_PREFIX, 'the item behind a Marketplace link could not be read', error)
    return null
  }
}

function linkParams(params: PageSearchParams): URLSearchParams {
  const firsts = Object.entries(params).flatMap(([name, value]) => {
    const first = Array.isArray(value) ? value[0] : value
    return typeof first === 'string' ? [[name, first]] : []
  })
  return new URLSearchParams(firsts)
}

/**
 * The /marketplace page's metadata. Local: the title only, and noindex. SaaS: a link's item card
 * (`?tab=&item=`), else the Marketplace's own card.
 */
export async function marketplaceMetadata(params: PageSearchParams, headers: HeaderReader): Promise<Metadata> {
  if (!isSaaS) return LOCAL_MARKETPLACE_METADATA
  const origin = requestOrigin(headers)
  const link = marketplaceLink(linkParams(params))
  const item = link ? await readSharedItem(link) : null
  return link && item ? itemCard(origin, link, item) : marketplaceCard(origin)
}
