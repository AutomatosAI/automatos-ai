/**
 * PRD-270 US-270-08b — the hosted edition's public answers, in their wire shapes: the catalogue
 * (orchestrator/services/marketplace_cloud/catalog.py `build_catalog`) and one page of a public
 * read (orchestrator/api/public_marketplace.py over services/marketplace_public/). Shared by the
 * mapping test and the visitor's tabs test.
 */
import { AUTOMATOS_CLOUD_ORIGIN, CLOUD_FILES_PATH } from '@/lib/marketplace/cloud-files'

/** The hosted edition's API origin, as apiClient.getBaseUrl() answers it on SaaS. */
export const BASE = AUTOMATOS_CLOUD_ORIGIN
export const ARCADE_DIGEST = 'a'.repeat(64)
export const ARCADE_OLD_DIGEST = 'c'.repeat(64)
export const HUD_DIGEST = 'd'.repeat(64)

export function filesOf(digest: string): string {
  return `${CLOUD_FILES_PATH}${digest}`
}

export function encodedJson(value: unknown): string {
  return Buffer.from(JSON.stringify(value), 'utf8').toString('base64')
}

/** One catalogue item: a listed, signed version of a theme. */
export function catalogItem(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  const digest = typeof overrides.digest === 'string' ? overrides.digest : ARCADE_DIGEST
  const files = filesOf(digest)
  return {
    type: 'theme',
    id: 'automatos/arcade',
    listingId: 'automatos.arcade',
    version: '1.1.0',
    digest,
    name: 'Arcade',
    description: 'Automatos as an arcade cabinet.',
    publisher: { name: 'automatos', official: true },
    engines: { automatos: '^1.0.0', sdk: '^1.0.0' },
    uses: ['frame', 'looks'],
    pricing: 'Free',
    signature: {
      form: 'automatos-theme-signature', algorithm: 'ed25519', keyId: 'automatos-2026',
      id: 'automatos/arcade', version: '1.1.0', digest, signature: 'c2lnbmF0dXJl',
    },
    files: { 'theme.json': { type: 'application/json', size: 120, sha256: 'e'.repeat(64) } },
    themeJson: encodedJson({
      base: 'studio', tone: 'dark', minWidth: 1024, pages: { '/settings': {}, '/chat': {} }, routes: ['arcade'],
    }),
    package: `${files}/package.autotheme`,
    cover: `${files}/assets/previews/default.png`,
    phoneCover: `${files}/assets/previews/phone/default.png`,
    presets: [
      {
        id: 'default',
        default: true,
        previews: [{ screen: null, url: `${files}/assets/previews/default.png` }],
        phonePreviews: [{ screen: null, url: `${files}/assets/previews/phone/default.png` }],
      },
    ],
    otherPreviews: [],
    otherPhonePreviews: [],
    ...overrides,
  }
}

export const ARCADE_NEW = catalogItem()
export const ARCADE_OLD = catalogItem({ version: '1.0.0', digest: ARCADE_OLD_DIGEST, engines: { automatos: '^0.9.0' } })
export const HUD = catalogItem({
  id: 'automatos/hud', listingId: 'automatos.hud', version: '1.0.0', digest: HUD_DIGEST, name: 'HUD',
  description: 'Automatos as a command deck.',
})

export function catalog(items: readonly unknown[]): Record<string, unknown> {
  return { schema: 1, generatedAt: '2026-10-31T09:00:00+00:00', items }
}

/** One page of `GET /api/public/marketplace/{type}`. */
export function publicAnswer(type: string, items: readonly unknown[], more = false, offset = 0): Record<string, unknown> {
  return { type, items, limit: 100, offset, more }
}

export function publicEntry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    key: 'research-assistant',
    name: 'Research Assistant',
    description: 'Finds, reads and summarises sources.',
    category: 'Research',
    icon: '🔎',
    tags: ['research', 'web'],
    publisher: 'Automatos',
    requiredApps: ['notion'],
    ...overrides,
  }
}
