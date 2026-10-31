/**
 * One Marketplace page for everyone (PRD-270 §5.7) — what a signed-out visitor's page reads, as the
 * hosted edition answers it: every page of a tab's public read, the Featured read and the counts.
 * Public, allowlisted, rate-limited and cached routes only (orchestrator/api/public_marketplace.py);
 * nothing here names a workspace or a member route.
 *
 * A tab's read keeps each item twice: its display fields, mapped and checked by `publicItem`
 * (../public-items.ts), and the raw answer, for the type's own plain facts a card shows (an
 * application's action count, a plugin's version, an LLM route's context window). An item with no
 * key or name is dropped, a repeated key is kept once, and at most MAX_PUBLIC_PAGES pages are read.
 */
import { apiClient } from '@/lib/api-client'

import { PUBLIC_COUNTS_ENDPOINT, PUBLIC_FEATURED_ENDPOINT, publicItem, publicPagePath } from '../public-items'
import type { PublicItem, PublicMarketplaceType } from '../public-items'

export type Json = Readonly<Record<string, unknown>>

/** One item of a public read: its checked display fields, and its raw answer for the type's facts. */
export interface PublicEntry {
  readonly item: PublicItem
  readonly raw: Json
}

/** The Featured read: the featured agents, and the most installed agents and Playbooks besides. */
export interface PublicFeatured {
  readonly featured: readonly Json[]
  readonly popular: readonly Json[]
}

/** The counts read: approved Marketplace agents and Playbooks, and their installs added up. */
export interface PublicCounts {
  readonly items: number
  readonly installs: number
}

/** At most this many pages (of 100) of one read: no tab comes close. */
export const MAX_PUBLIC_PAGES = 20

const NO_FEATURED: PublicFeatured = { featured: [], popular: [] }
const NO_COUNTS: PublicCounts = { items: 0, installs: 0 }

export function isJson(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A whole number of 0 or more, else 0. */
export function countOf(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : 0
}

interface ReadPage {
  readonly entries: readonly PublicEntry[]
  readonly next: number | null
}

function entryOf(raw: unknown, type: PublicMarketplaceType): PublicEntry | null {
  const item = publicItem(raw, type)
  return item && isJson(raw) ? { item, raw } : null
}

async function readPage(type: PublicMarketplaceType, offset: number): Promise<ReadPage> {
  const answer = await apiClient.get<unknown>(publicPagePath(type, offset))
  if (!isJson(answer) || answer.type !== type || !Array.isArray(answer.items)) return { entries: [], next: null }
  const entries = answer.items
    .map((raw) => entryOf(raw, type))
    .filter((entry): entry is PublicEntry => entry !== null)
  const more = answer.more === true && answer.items.length > 0
  return { entries, next: more ? offset + answer.items.length : null }
}

/** Every item of the type's public read, in its order, the first of each key only. */
export async function readPublicEntries(type: PublicMarketplaceType): Promise<PublicEntry[]> {
  const pages: (readonly PublicEntry[])[] = []
  let offset: number | null = 0
  while (offset !== null && pages.length < MAX_PUBLIC_PAGES) {
    const page: ReadPage = await readPage(type, offset)
    pages.push(page.entries)
    offset = page.next
  }
  const byKey = new Map<string, PublicEntry>()
  for (const entry of pages.flat()) {
    if (!byKey.has(entry.item.key)) byKey.set(entry.item.key, entry)
  }
  return Array.from(byKey.values())
}

function jsonList(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isJson) : []
}

/** The Featured read; an answer that isn't one reads as nothing featured. */
export async function readPublicFeatured(): Promise<PublicFeatured> {
  const answer = await apiClient.get<unknown>(PUBLIC_FEATURED_ENDPOINT)
  if (!isJson(answer)) return NO_FEATURED
  return { featured: jsonList(answer.featured), popular: jsonList(answer.popular) }
}

/** The counts read; an answer that isn't one reads as none. */
export async function readPublicCounts(): Promise<PublicCounts> {
  const answer = await apiClient.get<unknown>(PUBLIC_COUNTS_ENDPOINT)
  if (!isJson(answer)) return NO_COUNTS
  return { items: countOf(answer.items), installs: countOf(answer.installs) }
}
