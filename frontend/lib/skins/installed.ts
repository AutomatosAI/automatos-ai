// PRD-259 §4 — a workspace's installed skins as the Style dropdown reads them
// (GET /api/skins/installed, POST /api/skins/import). The response is checked here, where
// it enters the browser: an item without a UUID id or a name is dropped. Its skin fields
// are untrusted until resolveInstalledSkin runs them through the one validator and the one
// stylesheet writer; anything that fails either is not applied (the caller falls back).

import { skinToCss } from './css'
import { hasOwn, isRecord, type Fields } from './fields'
import { SKIN_ID_PATTERN, SKIN_SCHEMA_ID, type Skin, type SkinTone } from './schema'
import { validateSkin } from './validate'

export const INSTALLED_SKINS_PATH = '/api/skins/installed'
export const IMPORT_SKIN_PATH = '/api/skins/import'
/** The multipart field the import route reads (``file: UploadFile``). */
export const IMPORT_FILE_FIELD = 'file'
/** What the hidden file input offers. */
export const SKIN_FILE_ACCEPT = '.autoskin,.zip'

const LOG_PREFIX = '[Skins]'

/** The skin.json fields the route returns beside the row's own (records.PAYLOAD_FIELDS). */
const PAYLOAD_FIELDS = ['base', 'tones', 'tokens', 'fonts', 'shape', 'mascot', 'preview'] as const

export interface InstalledSkin {
  readonly id: string
  readonly slug: string
  readonly name: string
  readonly version: string
  /** The skin.json fields as the route sent them: unchecked until resolveInstalledSkin. */
  readonly payload: Fields
}

export interface ResolvedSkin {
  readonly id: string
  readonly name: string
  readonly skin: Skin
  readonly css: string
}

function toInstalledSkin(value: unknown): InstalledSkin | null {
  if (!isRecord(value)) return null
  const { id, slug, name, version } = value
  if (typeof id !== 'string' || !SKIN_ID_PATTERN.test(id) || typeof name !== 'string') return null
  // A field the skin doesn't set comes back as null: absent, as far as the validator goes.
  const payload = PAYLOAD_FIELDS.filter((field) => hasOwn(value, field) && value[field] !== null)
    .map((field) => [field, value[field]] as const)
  return {
    id,
    slug: typeof slug === 'string' ? slug : '',
    name,
    version: typeof version === 'string' ? version : '',
    payload: Object.fromEntries(payload),
  }
}

/** The route's answer as a list; an item that isn't one is dropped and logged. */
export function parseInstalledSkins(data: unknown): InstalledSkin[] {
  if (!Array.isArray(data)) {
    console.warn(`${LOG_PREFIX} The installed skins answer wasn't a list.`)
    return []
  }
  const items = data.map(toInstalledSkin)
  if (items.includes(null)) console.warn(`${LOG_PREFIX} Dropped an installed skin the browser couldn't read.`)
  return items.filter((item): item is InstalledSkin => item !== null)
}

/** One item of the import route's answer, or null when it isn't one. */
export function parseInstalledSkin(data: unknown): InstalledSkin | null {
  return toInstalledSkin(data)
}

/** The skin.json the item carries, as validateSkin reads it. */
export function skinInputOf(item: InstalledSkin): Fields {
  return { schema: SKIN_SCHEMA_ID, slug: item.slug, name: item.name, version: item.version, ...item.payload }
}

/** The validated skin and its stylesheet, or null (logged) when either refuses it. */
export function resolveInstalledSkin(item: InstalledSkin): ResolvedSkin | null {
  const result = validateSkin(skinInputOf(item))
  if (!result.ok) {
    console.warn(`${LOG_PREFIX} ${item.id} failed the skin rules:`, result.errors.map((error) => error.code))
    return null
  }
  try {
    return { id: item.id, name: item.name, skin: result.skin, css: skinToCss(result.skin, item.id) }
  } catch (error) {
    console.warn(`${LOG_PREFIX} ${item.id} couldn't be written as a stylesheet.`, error)
    return null
  }
}

/** A single-tone skin locks the tone (§4, decision K4); a two-tone skin locks nothing. */
export function lockedToneOf(skin: Pick<Skin, 'tones'> | null | undefined): SkinTone | undefined {
  return skin && skin.tones.length === 1 ? skin.tones[0] : undefined
}

function messagesOf(detail: unknown): string[] {
  if (!isRecord(detail) || !Array.isArray(detail.errors)) return []
  return detail.errors.flatMap((entry: unknown) =>
    isRecord(entry) && typeof entry.message === 'string' ? [entry.message] : [])
}

/**
 * The validator's messages from a refused import. apiClient throws an Error whose message is
 * the response's `detail`, JSON-encoded when it is an object ({message, errors: [...]}).
 */
export function importErrorMessages(error: unknown): string[] {
  if (!(error instanceof Error)) return []
  try {
    return messagesOf(JSON.parse(error.message))
  } catch {
    return []
  }
}
