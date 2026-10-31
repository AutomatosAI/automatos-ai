/**
 * PRD-270 W2 US-270-14b, US-270-14d (K18, K19) — the Cloud items answer (`GET
 * /api/marketplace/cloud-sync/items`), read and matched with no HTTP client: which Cloud agents, Playbooks,
 * packages and plugins this install holds, each by the key its install route takes (an agent's and a
 * Playbook's integer id, a package's slug, a plugin's uuid).
 *
 * It imports nothing at runtime, so the install gate (cloud-install-gate.ts), which lib/api-client.ts calls,
 * reads the answer without importing the client back. lib/marketplace/cloud-items.ts reads the answer
 * through apiClient with `cloudItemsOf`. An answer that isn't the shape below throws: the badges and the
 * install gate never treat an answer they can't read as "no Cloud items".
 */
export const CLOUD_ITEMS_BASE = '/api/marketplace/cloud-sync'
export const CLOUD_ITEMS_ENDPOINT = `${CLOUD_ITEMS_BASE}/items`
export const CLOUD_ITEMS_UNREADABLE = "Couldn't read which items came from Automatos Cloud."

export type CloudItemKind = 'agent' | 'playbook' | 'package' | 'plugin'
/** The local key an install route takes: an agent's or a Playbook's id, a package's slug, a plugin's uuid. */
export type CloudItemKey = number | string

interface CloudRow {
  externalId: string
  /** Automatos Cloud's version. */
  version: string
  /** The bundle's sha256. */
  revision: string
  name: string
}

export interface CloudAgentRow extends CloudRow { id: number }
export interface CloudPlaybookRow extends CloudRow { id: number }
export interface CloudPackageRow extends CloudRow { slug: string }
export interface CloudPluginRow extends CloudRow { id: string }

/** `GET /items` (local_reads.cloud_items). */
export interface CloudItems {
  agents: CloudAgentRow[]
  playbooks: CloudPlaybookRow[]
  packages: CloudPackageRow[]
  plugins: CloudPluginRow[]
}

type Json = Record<string, unknown>
type RowList = keyof CloudItems

/** Each kind's list in the items answer, and the field its install key is in. */
const LISTED: Readonly<Record<CloudItemKind, readonly [list: RowList, keyField: 'id' | 'slug', keyType: 'number' | 'string']>> = {
  agent: ['agents', 'id', 'number'],
  playbook: ['playbooks', 'id', 'number'],
  package: ['packages', 'slug', 'string'],
  plugin: ['plugins', 'id', 'string'],
}

/** The text fields every row carries besides its key. */
const ROW_TEXTS = ['externalId', 'version', 'revision', 'name'] as const

export function isJson(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRow(row: unknown, keyField: string, keyType: string): boolean {
  return isJson(row) && typeof row[keyField] === keyType && ROW_TEXTS.every((field) => typeof row[field] === 'string')
}

function listOf(answer: Json, kind: CloudItemKind): unknown[] {
  const [list, keyField, keyType] = LISTED[kind]
  const rows = answer[list]
  const readable = Array.isArray(rows) && rows.every((row) => isRow(row, keyField, keyType))
  if (!readable) throw new Error(CLOUD_ITEMS_UNREADABLE)
  return rows
}

/** The items answer as typed lists; throws when it isn't four lists of rows carrying their keys and texts. */
export function cloudItemsOf(answer: unknown): CloudItems {
  if (!isJson(answer)) throw new Error(CLOUD_ITEMS_UNREADABLE)
  return {
    agents: listOf(answer, 'agent') as CloudAgentRow[],
    playbooks: listOf(answer, 'playbook') as CloudPlaybookRow[],
    packages: listOf(answer, 'package') as CloudPackageRow[],
    plugins: listOf(answer, 'plugin') as CloudPluginRow[],
  }
}

/** A key as the items answer and the install routes compare it (a uuid in any case is one plugin). */
function keyText(kind: CloudItemKind, key: CloudItemKey): string {
  return kind === 'plugin' ? String(key).toLowerCase() : String(key)
}

/** A listed row of any kind. */
type CloudItemRow = CloudItems[RowList][number]

/** A listed row's install key: a package's slug, every other kind's id (LISTED's key fields). */
function rowKey(row: CloudItemRow): CloudItemKey {
  return 'slug' in row ? row.slug : row.id
}

/** Whether the row of `kind` with the local `key` is a Cloud item; false while the items aren't read. */
export function isCloudItem(items: CloudItems | undefined, kind: CloudItemKind, key: CloudItemKey): boolean {
  if (!items) return false
  const [list] = LISTED[kind]
  const wanted = keyText(kind, key)
  const rows: readonly CloudItemRow[] = items[list]
  return rows.some((row) => keyText(kind, rowKey(row)) === wanted)
}
