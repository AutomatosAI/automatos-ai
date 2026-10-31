/**
 * PRD-270 US-270-10 (§5.7 Sharing and Search engines, K23) — the card a Marketplace link shows
 * where it is posted, and the /marketplace page's metadata around it.
 *
 * A link to one item, `/marketplace?tab=<tab>&item=<key>` (the link contract of
 * return-after-sign-in.ts and deep-link/deep-link-params.ts), shows that item: its name and
 * description from the hosted edition's public reads, and its picture, a theme's cover on the
 * public catalogue files route of the API origin (the file the visitor's storefront draws), for
 * anything else the Automatos logo on the page's own origin. Each item has one canonical URL,
 * rebuilt from its tab value and key only, so `?tab=playbooks&item=x&add=1` and
 * `?tab=recipes&item=x` are one page to a search engine. A link that names no item, or whose item
 * can't be read, shows the Marketplace's own card. The local edition has no public Marketplace:
 * its page keeps a title and asks not to be indexed.
 *
 * Pure: the reads are share-card-read.ts's. Everything taken from an answer is data: its text goes
 * into the card as text (Next escapes it), and the only URL kept from it is a cover that is a
 * catalogue files path on the API origin (cloud-files.ts isCloudFileUrl).
 */
import type { Metadata } from 'next'

import type { MarketplaceLink } from '@/components/marketplace/deep-link/deep-link-params'
import { CLOUD_FILES_PATH, isCloudFileUrl } from '@/lib/marketplace/cloud-files'
import { ITEM_PARAM, MARKETPLACE_PATH, TAB_PARAM } from '@/lib/marketplace/return-after-sign-in'

export const MARKETPLACE_TITLE = 'Automatos Marketplace'
export const MARKETPLACE_DESCRIPTION =
  'Themes, agents, Playbooks, packages, applications, plugins, skills and LLMs for Automatos. Browse them, then add any of them to your workspace.'
export const SITE_NAME = 'Automatos'
/** The logo every other card uses, on the page's own origin (the sign-in page's logo, providers.tsx). */
export const LOGO_PATH = '/brand/automatos-mark-hi.png'
export const LARGE_IMAGE_CARD = 'summary_large_image'
/** A card's description is cut to this many characters: a card shows two or three lines. */
export const MAX_DESCRIPTION_CHARS = 300

const ELLIPSIS = '…'
const WEB_PROTOCOLS: ReadonlySet<string> = new Set(['http', 'https'])
const DEFAULT_PROTOCOL = 'https'
/** A host name or IPv4 address with an optional port: anything else in Host makes no URL. */
const HOST_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252})(?::\d{1,5})?$/

/** The item a card shows: its name, its description, and its picture (none: the logo). */
export interface SharedItem {
  readonly name: string
  readonly description: string | null
  readonly image: string | null
}

export type HeaderReader = Pick<Headers, 'get'>

type Json = Record<string, unknown>

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

/**
 * The page's origin from the request's Host and X-Forwarded-Proto (the frontend has no app-URL
 * setting), or null when Host isn't a plain host name: then the card carries no URL at all.
 */
export function requestOrigin(headers: HeaderReader): string | null {
  const host = headers.get('host')?.trim()
  if (!host || !HOST_PATTERN.test(host)) return null
  const forwarded = headers.get('x-forwarded-proto')?.split(',')[0]?.trim().toLowerCase()
  const protocol = forwarded && WEB_PROTOCOLS.has(forwarded) ? forwarded : DEFAULT_PROTOCOL
  return `${protocol}://${host.toLowerCase()}`
}

/** The one canonical URL of an item: its tab value and key, nothing else. */
export function canonicalItemUrl(origin: string, link: Pick<MarketplaceLink, 'tab' | 'item'>): string {
  return `${origin}${MARKETPLACE_PATH}?${new URLSearchParams({ [TAB_PARAM]: link.tab, [ITEM_PARAM]: link.item })}`
}

function apiOrigin(): string | null {
  try {
    const url = new URL(process.env.NEXT_PUBLIC_API_URL ?? '')
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.origin : null
  } catch {
    // No API URL in this build: a theme's card shows the logo.
    return null
  }
}

/** A theme's cover as a URL on the API origin's catalogue files route, or null for anything else. */
export function themeCoverUrl(cover: unknown): string | null {
  const origin = apiOrigin()
  if (typeof cover !== 'string' || !cover.startsWith(CLOUD_FILES_PATH) || origin === null) return null
  const url = `${origin}${cover}`
  return isCloudFileUrl(url) ? url : null
}

/**
 * The theme `listingId` in a schema-1 catalogue, as a card: its newest version (the catalogue lists
 * a theme's versions newest first), or null when the catalogue doesn't list it.
 */
export function catalogSharedItem(catalog: unknown, listingId: string): SharedItem | null {
  const items = isRecord(catalog) && Array.isArray(catalog.items) ? catalog.items : []
  const item = items.find((entry): entry is Json => isRecord(entry) && entry.type === 'theme' && entry.listingId === listingId)
  if (!item) return null
  return { name: textOf(item.name) ?? listingId, description: textOf(item.description), image: themeCoverUrl(item.cover) }
}

function shortened(text: string): string {
  return text.length <= MAX_DESCRIPTION_CHARS ? text : `${text.slice(0, MAX_DESCRIPTION_CHARS - 1).trimEnd()}${ELLIPSIS}`
}

interface Card {
  readonly title: string
  readonly cardTitle: string
  readonly description: string
  readonly url: string | null
  readonly image: string | null
  readonly imageAlt: string
}

function cardMetadata({ title, cardTitle, description, url, image, imageAlt }: Card): Metadata {
  const images = image ? [{ url: image, alt: imageAlt }] : undefined
  return {
    title,
    description,
    ...(url ? { alternates: { canonical: url } } : {}),
    openGraph: { type: 'website', siteName: SITE_NAME, title: cardTitle, description, ...(url ? { url } : {}), images },
    twitter: { card: LARGE_IMAGE_CARD, title: cardTitle, description, images: image ? [image] : undefined },
  }
}

/** The Marketplace's own card: a link with no item, or one whose item can't be read. */
export function marketplaceCard(origin: string | null): Metadata {
  return cardMetadata({
    title: MARKETPLACE_TITLE,
    cardTitle: MARKETPLACE_TITLE,
    description: MARKETPLACE_DESCRIPTION,
    url: origin ? `${origin}${MARKETPLACE_PATH}` : null,
    image: origin ? `${origin}${LOGO_PATH}` : null,
    imageAlt: SITE_NAME,
  })
}

/** One item's card, with its canonical URL; no picture of its own is the logo. */
export function itemCard(origin: string | null, link: MarketplaceLink, item: SharedItem): Metadata {
  const logo = origin ? `${origin}${LOGO_PATH}` : null
  return cardMetadata({
    title: `${item.name} · ${MARKETPLACE_TITLE}`,
    cardTitle: item.name,
    description: shortened(item.description ?? `${item.name} on the ${MARKETPLACE_TITLE}.`),
    url: origin ? canonicalItemUrl(origin, link) : null,
    image: item.image ?? logo,
    imageAlt: item.name,
  })
}

/** The local edition: no public Marketplace, so a title and no indexing. */
export const LOCAL_MARKETPLACE_METADATA: Metadata = {
  title: MARKETPLACE_TITLE,
  robots: { index: false, follow: false },
}
