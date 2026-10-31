/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the public reads' items in the shapes the
 * member page's own cards and dialogs draw: agents and Playbooks (the Agents tab's cards, the
 * Featured hero and rail, Popular, the item dialog) and packages (the Packages tab's cards).
 *
 * Only what the public item says is mapped (lib/marketplace/source/public-reads.ts): its name,
 * description, category, icon, tags, publisher, required apps and a package's members; Featured
 * and Popular also carry their version and install count. A field the member shape wants that the
 * public read doesn't carry is left empty, never made up: no steps, no model, no configuration,
 * no setup questions. Per-item install counts are 0 here and the cards don't show them signed out
 * (useMarketplaceAccess). An agent's or a Playbook's key is its catalogue id; an item whose key
 * isn't one is dropped.
 */
import type { MarketplaceItem } from '@/hooks/use-marketplace-api'
import { publicItem } from '@/lib/marketplace/public-items'
import type { PublicItem } from '@/lib/marketplace/public-items'
import { countOf, isJson } from '@/lib/marketplace/source/public-reads'
import type { Json, PublicEntry } from '@/lib/marketplace/source/public-reads'

import type { MarketplacePackage, PackageMember } from '../package-types'

/** The member routes' wire types for an agent and a Playbook. */
export const AGENT_TYPE = 'agent'
export const PLAYBOOK_TYPE = 'recipe'
export type ItemType = typeof AGENT_TYPE | typeof PLAYBOOK_TYPE

/** The public read each member type comes from, and back. */
export const PUBLIC_TYPE_OF = { [AGENT_TYPE]: 'agents', [PLAYBOOK_TYPE]: 'playbooks' } as const
const ITEM_TYPE_OF: Readonly<Record<string, ItemType>> = { agents: AGENT_TYPE, playbooks: PLAYBOOK_TYPE }

const CATALOGUE_ID = /^[1-9]\d{0,9}$/
const NOT_RECORDED = ''

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

/** What an agent or a Playbook keeps in `metadata` from the public item: the apps it needs. */
function metadataOf(item: PublicItem, type: ItemType): Record<string, unknown> {
  if (type === PLAYBOOK_TYPE) return { required_tools: [...item.requiredApps] }
  // The member route answers an agent's enabled tools as upper-case names (api/marketplace.py).
  return { tool_names: item.requiredApps.map((app) => app.toUpperCase()), tool_icons: [] }
}

/** A public agent or Playbook as a member Marketplace item, or null when its key isn't a catalogue id. */
export function marketplaceItemOf(item: PublicItem, type: ItemType): MarketplaceItem | null {
  if (!CATALOGUE_ID.test(item.key)) return null
  return {
    id: Number(item.key),
    type,
    name: item.name,
    description: item.description ?? '',
    creator_name: item.publisher,
    icon: item.icon ?? undefined,
    category: item.category ?? undefined,
    tags: [...item.tags],
    install_count: 0,
    is_featured: false,
    version: NOT_RECORDED,
    metadata: metadataOf(item, type),
    created_at: NOT_RECORDED,
    updated_at: NOT_RECORDED,
    // The Playbook cards read the category from here (marketplace-playbook-cards.tsx).
    ...(type === PLAYBOOK_TYPE ? { marketplace_category: item.category ?? undefined } : {}),
  }
}

/** A tab's public agents or Playbooks as member items. */
export function marketplaceItemsOf(entries: readonly PublicEntry[], type: ItemType): MarketplaceItem[] {
  return entries
    .map((entry) => marketplaceItemOf(entry.item, type))
    .filter((item): item is MarketplaceItem => item !== null)
}

/** One item of the Featured read (Featured or Popular), with its version and install count. */
export function featuredItemOf(raw: Json): MarketplaceItem | null {
  const type = typeof raw.type === 'string' ? ITEM_TYPE_OF[raw.type] : undefined
  const item = type ? publicItem(raw, PUBLIC_TYPE_OF[type]) : null
  const mapped = type && item ? marketplaceItemOf(item, type) : null
  if (!mapped) return null
  return { ...mapped, version: textOf(raw.version) ?? NOT_RECORDED, install_count: countOf(raw.installCount) }
}

export function featuredItemsOf(raws: readonly Json[]): MarketplaceItem[] {
  return raws.map(featuredItemOf).filter((item): item is MarketplaceItem => item !== null)
}

function memberOf(value: unknown): PackageMember | null {
  if (!isJson(value)) return null
  const type = textOf(value.type)
  const name = textOf(value.name)
  return type ? { type, ...(name ? { name } : {}) } : null
}

/** A public package as the Packages tab's card draws it: its members by type and name, its apps. */
export function packageOf({ item, raw }: PublicEntry): MarketplacePackage {
  const members = Array.isArray(raw.members) ? raw.members.map(memberOf).filter((m): m is PackageMember => m !== null) : []
  return {
    id: item.key,
    slug: item.key,
    name: item.name,
    description: item.description ?? undefined,
    vertical_tags: [...item.tags],
    matching: {},
    members,
    setup_manifest: { required_connects: item.requiredApps.map((app) => ({ app_name: app })) },
    showcase: false,
  }
}
