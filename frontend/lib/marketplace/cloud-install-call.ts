/**
 * PRD-270 W2 US-270-14d (K19) — which apiClient requests are K19's four install calls, and which item
 * each installs, by the key its route takes:
 *
 * - `POST /api/marketplace/items/{id}/install`: an agent;
 * - `POST /api/workflow-recipes/install/{id}`: a Playbook;
 * - `POST /api/marketplace/packages/{slug}/install`: a package;
 * - `POST /api/workspaces/{ws}/plugins` with `plugin_id` in its body (a JSON string or an object): a plugin.
 *
 * The path is read the way the browser sends it and the server routes it, so no spelling of one of these
 * calls passes as another request: parsed as a URL (dot segments resolved, backslashes as slashes, tabs
 * and newlines dropped, the query and fragment cut), percent-escapes decoded, trailing slashes dropped
 * (the server redirects them, keeping the POST). An integer id is compared as a number and a plugin's
 * uuid in its canonical form. A plugin call's body is read as request() sends it. An install call whose
 * item can't be told (a body that isn't JSON, an id that isn't an integer, a plugin_id that isn't a uuid)
 * is `UNREADABLE_INSTALL`, never "not an install". Pure: it imports nothing at runtime.
 */
import { isJson } from './cloud-item-rows'
import type { CloudItemKey, CloudItemKind } from './cloud-item-rows'

/** One install call: the kind of item it adds and the key its route names. */
export interface CloudInstallCall {
  kind: CloudItemKind
  key: CloudItemKey
}

/** An install call whose item can't be told, so it can't be known not to be a Cloud item. */
export const UNREADABLE_INSTALL = 'unreadable'

type InstallCallRead = CloudInstallCall | null | typeof UNREADABLE_INSTALL

const INSTALL_PATHS: ReadonlyArray<readonly [pattern: RegExp, kind: CloudItemKind]> = [
  [/^\/*api\/marketplace\/items\/([^/]+)\/install$/, 'agent'],
  [/^\/*api\/workflow-recipes\/install\/([^/]+)$/, 'playbook'],
  [/^\/*api\/marketplace\/packages\/([^/]+)\/install$/, 'package'],
]
const PLUGIN_ENABLE_PATH = /^\/*api\/workspaces\/[^/]+\/plugins$/
const UUID_HEX = /^[0-9a-f]{32}$/
/** A base for parsing an endpoint as the browser resolves it; never requested. */
const PARSE_BASE = 'http://install-gate.invalid'

/** The path the browser sends: dot segments, backslashes, tabs and newlines as a URL parser treats them. */
function sentPath(endpoint: string): string {
  try {
    return new URL(endpoint, PARSE_BASE).pathname
  } catch {
    const cut = endpoint.search(/[?#]/)
    return cut === -1 ? endpoint : endpoint.slice(0, cut)
  }
}

/** The path as the server routes it: escapes decoded, no trailing slash. */
function routedPath(endpoint: string): string {
  const path = sentPath(endpoint)
  let decoded = path
  try {
    decoded = decodeURIComponent(path)
  } catch {
    // A malformed escape: the server keeps it as written, so the path is compared as written.
  }
  return decoded.replace(/\/+$/, '')
}

/** An integer id as the route reads it ("07", "+7", " 7 ", "7.0" and "7" are one agent), or null. */
function integerKey(raw: string): number | null {
  const value = Number(raw.trim().replace(/_/g, ''))
  return Number.isSafeInteger(value) ? value : null
}

/** A uuid in the canonical form, from the forms a uuid parser reads (braces, urn:, no hyphens), or null. */
function uuidKey(raw: string): string | null {
  const lowered = raw.toLowerCase()
  const hex = lowered.replace(/urn:/g, '').replace(/uuid:/g, '').replace(/^[{}]+|[{}]+$/g, '').replace(/-/g, '')
  if (!UUID_HEX.test(hex)) return null
  return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-')
}

/**
 * A key as its route reads it: a package's slug as written, an integer id as a number, a uuid canonical;
 * null when it isn't one its route could take.
 */
export function installKey(kind: CloudItemKind, raw: CloudItemKey): CloudItemKey | null {
  const text = String(raw)
  if (kind === 'package') return text
  return kind === 'plugin' ? uuidKey(text) : integerKey(text)
}

/** The body as request() sends it, parsed back: a string as JSON, an object stringified (FormData isn't JSON). */
function sentJson(body: unknown): unknown {
  if (typeof body === 'string') return JSON.parse(body)
  if (typeof body === 'object' && body !== null && !(body instanceof FormData)) return JSON.parse(JSON.stringify(body))
  return null
}

function callOf(kind: CloudItemKind, raw: CloudItemKey): InstallCallRead {
  const key = installKey(kind, raw)
  return key === null ? UNREADABLE_INSTALL : { kind, key }
}

function pluginCall(body: unknown): InstallCallRead {
  let sent: unknown
  try {
    sent = sentJson(body)
  } catch {
    return UNREADABLE_INSTALL
  }
  // No plugin_id, or one that isn't a string, enables nothing: the route refuses it.
  if (!isJson(sent) || typeof sent.plugin_id !== 'string') return null
  return callOf('plugin', sent.plugin_id)
}

/**
 * The install call `endpoint` and `options` make, `UNREADABLE_INSTALL` for one whose item can't be told,
 * or null for every other request.
 */
export function installCallOf(endpoint: string, options: RequestInit): InstallCallRead {
  if ((options.method ?? 'GET').toUpperCase() !== 'POST') return null
  const path = routedPath(endpoint)
  for (const [pattern, kind] of INSTALL_PATHS) {
    const raw = pattern.exec(path)?.[1]
    if (raw !== undefined) return callOf(kind, raw)
  }
  return PLUGIN_ENABLE_PATH.test(path) ? pluginCall(options.body) : null
}
