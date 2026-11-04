/**
 * PRD-270 W2 US-270-14d (§5.6, §5.8, K19) — the install gate: on the local edition, an Add or Enable of an
 * Automatos Cloud item waits for its disclosure, and nothing is added unless Add is clicked.
 *
 * apiClient.request() calls `cloudInstallGate(endpoint, options, this)` once, before anything else it
 * awaits; it is the one choke point every install of an agent, Playbook, package or plugin passes, from
 * any card, list or dialog (their components can't take a line: K4). For K19's four install calls
 * (cloud-install-call.ts) naming a Cloud item, the gate awaits the registered opener, the disclosure
 * dialog (CloudItemsLayer, mounted in the Marketplace): Add lets that one request go; Close refuses it
 * with `CloudInstallDeclined`. Every other request, and every request on the hosted edition, returns at
 * once, with nothing awaited.
 *
 * Fails closed: when the Cloud items can't be read, or which item an install call names can't be told
 * (cloud-install-call.ts `UNREADABLE_INSTALL`), the install is refused ("Couldn't check …"); when no layer is
 * mounted to show a Cloud item's disclosure, it is refused ("Open it in the Marketplace …"). The calling
 * component's own error toast shows the sentence.
 *
 * The Cloud items come from the layer while one is mounted, read afresh for each install (a Sync in another
 * tab can add one); with none, the gate reads them itself through the `client` it is handed, kept a minute
 * (K19) and dropped whenever a layer comes or goes. `allowNext(kind, key)` lets one install through
 * without asking, for a disclosure already read and Added from an update notice (US-270-14e). Auto's chat
 * installs are server-side and never pass here: they ask through their own approval (US-270-14g, K17).
 *
 * It imports nothing from lib/api-client.ts (which imports it): request() hands itself in as `client`.
 */
import { isLocal } from '@/lib/auth-edition'

import { installCallOf, installKey, UNREADABLE_INSTALL } from './cloud-install-call'
import type { CloudInstallCall } from './cloud-install-call'
import { CLOUD_ITEMS_ENDPOINT, cloudItemsOf, isCloudItem } from './cloud-item-rows'
import type { CloudItemKey, CloudItemKind, CloudItems } from './cloud-item-rows'

export const CLOUD_INSTALL_CLOSED = 'Nothing was added: you closed the review.'
export const CLOUD_INSTALL_UNCHECKED = "Couldn't check where this item comes from. Try again."
export const CLOUD_INSTALL_ELSEWHERE = 'Open it in the Marketplace to see what it will use, then add it there.'
export const CLOUD_INSTALL_BUSY = 'Finish the review that is open first, then add this.'

/** How long the gate keeps the Cloud items it read itself (no layer mounted). */
export const GATE_ITEMS_CACHE_MS = 60_000
/** How long an allowance waits for its install call (it is made right after Add). */
export const ALLOWANCE_MS = 30_000

/** An install refused before it was sent: its message is a sentence for the person. */
export class CloudInstallRefused extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CloudInstallRefused'
  }
}

/** The disclosure was closed without Add: nothing was sent. */
export class CloudInstallDeclined extends CloudInstallRefused {
  constructor() {
    super(CLOUD_INSTALL_CLOSED)
    this.name = 'CloudInstallDeclined'
  }
}

/** What the gate reads the Cloud items through when no layer is mounted (apiClient passes itself). */
export interface CloudGateClient {
  get<T = unknown>(endpoint: string): Promise<T>
}

/** The mounted layer: shows one Cloud item's disclosure (true on Add, false on Close), and reads the items. */
export interface CloudInstallOpener {
  open(call: CloudInstallCall): Promise<boolean>
  items(): Promise<CloudItems>
}

interface Allowance { kind: CloudItemKind; key: string; until: number }

let opener: CloudInstallOpener | null = null
let ownRead: { items: Promise<CloudItems>; at: number } | null = null
let allowances: readonly Allowance[] = []

/** Registers the layer's opener; returns the call that unregisters it (only while it is still the one). */
export function registerCloudInstallOpener(next: CloudInstallOpener): () => void {
  opener = next
  ownRead = null
  return () => {
    if (opener !== next) return
    opener = null
    ownRead = null
  }
}

/** Lets the next install of `kind` `key` through without asking, once, within ALLOWANCE_MS. */
export function allowNext(kind: CloudItemKind, key: CloudItemKey): void {
  const allowed = installKey(kind, key)
  // A key no install route could take allows nothing.
  if (allowed === null) return
  const now = Date.now()
  allowances = [...allowances.filter((held) => held.until > now), { kind, key: String(allowed), until: now + ALLOWANCE_MS }]
}

/** Uses up the allowance this call has, if any. */
function takeAllowance(call: CloudInstallCall): boolean {
  const now = Date.now()
  const key = String(call.key)
  const match = allowances.find((allowed) => allowed.kind === call.kind && allowed.key === key && allowed.until > now)
  allowances = allowances.filter((allowed) => allowed !== match && allowed.until > now)
  return match !== undefined
}

function ownItems(client: CloudGateClient): Promise<CloudItems> {
  const now = Date.now()
  if (ownRead && now - ownRead.at < GATE_ITEMS_CACHE_MS) return ownRead.items
  const items = client.get<unknown>(CLOUD_ITEMS_ENDPOINT).then(cloudItemsOf)
  const read = { items, at: now }
  ownRead = read
  // A failed read isn't kept: the next install reads again.
  items.catch(() => {
    if (ownRead === read) ownRead = null
  })
  return items
}

async function cloudItems(client: CloudGateClient): Promise<CloudItems> {
  try {
    return await (opener ? opener.items() : ownItems(client))
  } catch {
    throw new CloudInstallRefused(CLOUD_INSTALL_UNCHECKED)
  }
}

async function reviewed(call: CloudInstallCall, client: CloudGateClient): Promise<void> {
  if (takeAllowance(call)) return
  const items = await cloudItems(client)
  if (!isCloudItem(items, call.kind, call.key)) return
  const shown = opener
  if (!shown) throw new CloudInstallRefused(CLOUD_INSTALL_ELSEWHERE)
  if (!(await shown.open(call))) throw new CloudInstallDeclined()
}

/**
 * Called once by apiClient.request() before it sends anything. Returns at once (nothing to await) for any
 * request that isn't one of K19's four install calls, and for every request on the hosted edition; for an
 * install call on the local edition, resolves when the request may go and throws a `CloudInstallRefused`
 * when it may not.
 */
export function cloudInstallGate(endpoint: string, options: RequestInit, client: CloudGateClient): Promise<void> | undefined {
  if (!isLocal) return undefined
  const call = installCallOf(endpoint, options)
  if (call === null) return undefined
  if (call === UNREADABLE_INSTALL) return Promise.reject(new CloudInstallRefused(CLOUD_INSTALL_UNCHECKED))
  return reviewed(call, client)
}
