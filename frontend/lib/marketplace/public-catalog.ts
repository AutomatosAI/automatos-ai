/**
 * PRD-270 US-270-08b (§5.7, K20) — what a signed-out visitor's Marketplace reads, and the pure
 * mapping that turns the hosted catalogue into the Themes storefront's own shapes.
 *
 * Two public routes of the hosted edition, read through apiClient's generic `get`, which sends no
 * bearer when there is none (api-client.ts doesn't grow, K3):
 *
 * - `GET /api/marketplace/catalog` (US-270-01): `{schema: 1, generatedAt, items}`, one item per
 *   listed, signed theme version, newest first per theme, every URL in it a path on the catalogue
 *   files route (orchestrator/services/marketplace_cloud/catalog.py);
 * - `GET /api/public/marketplace/{type}` (US-270-07a/b): one page of a tab's display fields,
 *   mapped by public-items.ts.
 *
 * `catalogListings` gives one ThemeListingDetail per theme: its newest version, with every listed
 * version in `versions`. Everything in the answer is data: an item that isn't one a theme version
 * can be is dropped, never thrown; a URL is kept only as a path on the catalogue files route,
 * resolved against apiClient.getBaseUrl() (themeFileUrl then draws it only on Automatos Cloud's or
 * this build's API origin, K15). Nothing workspace-scoped is made up: no install, no revocation,
 * and the rows are the hosted edition's own (`source` null). The visitor's page installs nothing, so
 * whether a version runs here is the member page's to say: every version reads as compatible.
 */
import { apiClient } from '@/lib/api-client'
import type {
  ThemeListingDetail,
  ThemeListingPublisher,
  ThemeListingVersion,
  ThemePresetPreviews,
  ThemePreviewShot,
} from '@/lib/api-client'
import { CLOUD_FILES_PATH } from '@/lib/marketplace/cloud-files'
import { THEME_DIGEST_PATTERN } from '@/lib/theme-host/constants'

import { PUBLIC_PAGE_LIMIT, publicPage } from './public-items'
import type { PublicMarketplaceType, PublicPage } from './public-items'

export const CATALOG_ENDPOINT = '/api/marketplace/catalog'
export const PUBLIC_MARKETPLACE_ENDPOINT = '/api/public/marketplace'

const CATALOG_SCHEMA = 1
const THEME_TYPE = 'theme'
const FREE_PRICING = 'Free'
const UNSAFE_PATH = /\.\.|\\|\s/

type Json = Record<string, unknown>

/** theme.json's layout facts, as the member detail reads them (listings.py listing_detail). */
type ManifestFacts = Pick<ThemeListingDetail, 'base' | 'tone' | 'minWidth' | 'pages' | 'routes'>

const NO_FACTS: ManifestFacts = { base: null, tone: null, minWidth: null, pages: [], routes: [] }

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function listOf<T>(value: unknown, map: (entry: unknown) => T | null): T[] {
  return Array.isArray(value) ? value.map(map).filter((entry): entry is T => entry !== null) : []
}

/** A catalogue path on the files route, as a URL on `base`; anything else is none. */
export function catalogFileUrl(path: unknown, base: string): string | null {
  if (typeof path !== 'string' || !path.startsWith(CLOUD_FILES_PATH) || UNSAFE_PATH.test(path)) return null
  return `${base}${path}`
}

function shotOf(value: unknown, base: string): ThemePreviewShot | null {
  if (!isRecord(value)) return null
  const url = catalogFileUrl(value.url, base)
  return url ? { screen: textOf(value.screen), url } : null
}

function presetOf(value: unknown, base: string): ThemePresetPreviews | null {
  const id = isRecord(value) ? textOf(value.id) : null
  if (!isRecord(value) || !id) return null
  const previews = listOf(value.previews, (shot) => shotOf(shot, base))
  const phonePreviews = listOf(value.phonePreviews, (shot) => shotOf(shot, base))
  return { id, default: value.default === true, previews, phonePreviews }
}

function publisherOf(value: unknown): ThemeListingPublisher | null {
  const name = isRecord(value) ? textOf(value.name) : null
  return isRecord(value) && name ? { name, official: value.official === true } : null
}

function decodedJson(base64: unknown): unknown {
  if (typeof base64 !== 'string') return null
  try {
    const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0))
    return JSON.parse(new TextDecoder().decode(bytes))
  } catch {
    // Not base64 of UTF-8 JSON: the detail shows no layout facts rather than failing the storefront.
    return null
  }
}

function manifestFacts(themeJson: unknown): ManifestFacts {
  const manifest = decodedJson(themeJson)
  if (!isRecord(manifest)) return NO_FACTS
  const width = manifest.minWidth
  return {
    base: textOf(manifest.base),
    tone: textOf(manifest.tone),
    minWidth: typeof width === 'number' && Number.isInteger(width) && width > 0 ? width : null,
    pages: isRecord(manifest.pages) ? Object.keys(manifest.pages).sort() : [],
    routes: listOf(manifest.routes, textOf),
  }
}

function enginesOf(value: unknown): Record<string, string> | null {
  if (!isRecord(value)) return null
  return Object.fromEntries(Object.entries(value).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
}

/** One version of a theme in the catalogue: its detail, and its entry in the versions map. */
export interface CatalogVersion {
  readonly detail: ThemeListingDetail
  readonly entry: ThemeListingVersion
}

interface Identity {
  readonly themeId: string
  readonly listingId: string
  readonly version: string
  readonly digest: string
  readonly publisher: ThemeListingPublisher
}

function identityOf(item: Json): Identity | null {
  const [themeId, listingId, version, digest] = [item.id, item.listingId, item.version, item.digest].map(textOf)
  const publisher = publisherOf(item.publisher)
  if (!themeId || !listingId || !version || !digest || !THEME_DIGEST_PATTERN.test(digest) || !publisher) return null
  return { themeId, listingId, version, digest, publisher }
}

function detailOf(item: Json, { digest, ...identity }: Identity, base: string): ThemeListingDetail {
  const keyId = isRecord(item.signature) ? textOf(item.signature.keyId) : null
  return {
    ...identity,
    name: textOf(item.name),
    description: textOf(item.description),
    packageId: digest,
    // The catalogue lists only versions an active Automatos key verifies (catalog.catalogued_verdict).
    signature: { state: 'verified', keyId },
    compatible: true,
    uses: listOf(item.uses, textOf),
    pricing: textOf(item.pricing) ?? FREE_PRICING,
    cover: catalogFileUrl(item.cover, base),
    phoneCover: catalogFileUrl(item.phoneCover, base),
    source: null,
    revoked: null,
    installed: null,
    presets: listOf(item.presets, (preset) => presetOf(preset, base)),
    otherPreviews: listOf(item.otherPreviews, (path) => catalogFileUrl(path, base)),
    otherPhonePreviews: listOf(item.otherPhonePreviews, (path) => catalogFileUrl(path, base)),
    readme: null,
    changelog: null,
    ...manifestFacts(item.themeJson),
    latest: identity.version,
    versions: {},
  }
}

/** One catalogue item as a theme version, or null when it isn't one. */
export function catalogVersion(item: unknown, base: string): CatalogVersion | null {
  const identity = isRecord(item) && item.type === THEME_TYPE ? identityOf(item) : null
  if (!isRecord(item) || !identity) return null
  const detail = detailOf(item, identity, base)
  const entry: ThemeListingVersion = {
    packageId: identity.digest,
    engines: enginesOf(item.engines),
    compatible: true,
    revoked: null,
    signature: detail.signature,
    createdAt: null,
  }
  return { detail, entry }
}

function withVersions(versions: readonly CatalogVersion[]): ThemeListingDetail {
  const versionMap = Object.fromEntries(
    versions.map(({ detail, entry }): [string, ThemeListingVersion] => [detail.version, entry]),
  )
  return { ...versions[0].detail, versions: versionMap }
}

/**
 * The catalogue as the storefront's listings, one per theme in the catalogue's order: each theme's
 * first (newest) version, with every version listed in `versions`. Anything but a schema-1
 * catalogue lists none.
 */
export function catalogListings(catalog: unknown, base: string): ThemeListingDetail[] {
  if (!isRecord(catalog) || catalog.schema !== CATALOG_SCHEMA || !Array.isArray(catalog.items)) return []
  const byTheme = new Map<string, CatalogVersion[]>()
  for (const version of listOf(catalog.items, (item) => catalogVersion(item, base))) {
    const listingId = version.detail.listingId
    byTheme.set(listingId, [...(byTheme.get(listingId) ?? []), version])
  }
  return Array.from(byTheme.values(), withVersions)
}

/** The hosted catalogue's themes, as the visitor's storefront lists them. */
export async function getPublicThemes(): Promise<ThemeListingDetail[]> {
  const catalog = await apiClient.get<unknown>(CATALOG_ENDPOINT)
  return catalogListings(catalog, apiClient.getBaseUrl())
}

/** One page of a public Marketplace read from `offset`. */
export async function getPublicPage(type: PublicMarketplaceType, offset: number): Promise<PublicPage> {
  const query = new URLSearchParams({ limit: String(PUBLIC_PAGE_LIMIT), offset: String(offset) })
  const answer = await apiClient.get<unknown>(`${PUBLIC_MARKETPLACE_ENDPOINT}/${type}?${query}`)
  return publicPage(answer, type, offset)
}
