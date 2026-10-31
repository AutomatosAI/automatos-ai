/**
 * PRD-270 US-270-08b (§5.7, K20) — one page of a public Marketplace read
 * (`GET /api/public/marketplace/{type}`, US-270-07a/b) as the visitor's tabs draw it.
 *
 * The route answers `{type, items, limit, offset, more}`, each item only its type's display fields
 * (orchestrator/services/marketplace_public/fields.py): `key`, `name`, `description`, `category`,
 * `icon`, `tags`, `publisher`, `requiredApps`, and the type's own plain facts (a version, counts, an
 * LLM's provider, context window, price tier and capabilities, a package's members). `publicPage`
 * maps it field by field: an item with no key or name is dropped, never thrown; a text that isn't
 * one reads as none, a list keeps its strings only, a count above 0 only. Every value is rendered
 * as text; nothing here is a URL the page fetches, but an `https` icon (an application's logo),
 * which the app's img-src decides.
 */

export const PUBLIC_PAGE_LIMIT = 100

export const PUBLIC_MARKETPLACE_TYPES = [
  'agents', 'playbooks', 'packages', 'applications', 'plugins', 'skills', 'llms',
] as const
export type PublicMarketplaceType = (typeof PUBLIC_MARKETPLACE_TYPES)[number]

/** One plain fact of an item: "Version 1.2.0", "Actions 12", "Context window 200,000". */
export interface PublicItemFact {
  readonly label: string
  readonly value: string
}

export interface PublicItem {
  /** The id the signed-in tab deep-links by (a slug, a catalogue id, a route key). */
  readonly key: string
  readonly name: string
  readonly description: string | null
  readonly category: string | null
  /** An emoji or other short text, or an `https` logo; anything else null. */
  readonly icon: string | null
  readonly tags: readonly string[]
  /** The publisher's display name. */
  readonly publisher: string
  /** The apps it needs connected, as lower-case slugs. */
  readonly requiredApps: readonly string[]
  readonly facts: readonly PublicItemFact[]
  /** A package's members, or an LLM's capabilities with their rating, one line each. */
  readonly includes: readonly string[]
}

export interface PublicPage {
  readonly items: readonly PublicItem[]
  /** The offset of the next page, or null when this is the last. */
  readonly next: number | null
}

type Json = Record<string, unknown>
type FactSpec = readonly [field: string, label: string]

const FACTS: Readonly<Record<PublicMarketplaceType, readonly FactSpec[]>> = {
  agents: [],
  playbooks: [],
  packages: [],
  applications: [['actionCount', 'Actions'], ['triggerCount', 'Triggers']],
  plugins: [
    ['version', 'Version'], ['skillsCount', 'Skills'], ['commandsCount', 'Commands'],
    ['agentsCount', 'Agents'], ['hooksCount', 'Hooks'],
  ],
  skills: [['version', 'Version']],
  llms: [['provider', 'Served by'], ['vendor', 'Made by'], ['contextWindow', 'Context window'], ['priceTier', 'Price tier']],
}

/** A package member's type (core/models/marketplace_packages.py MEMBER_TYPES) in the canonical words. */
const MEMBER_LABELS: Readonly<Record<string, string>> = {
  agent: 'Agent', tool: 'Application', skill: 'Skill', plugin: 'Plugin', playbook: 'Playbook', llm: 'LLM',
}
/** Code points, so an emoji joined from several (🧑‍💻 is three) still counts as short. */
const SHORT_ICON_MAX = 8
export const HTTPS_PREFIX = 'https://'
const LETTERS_OR_DIGITS = /[A-Za-z0-9]/
const NUMBER_FORMAT = new Intl.NumberFormat('en')

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function textsOf(value: unknown): string[] {
  return Array.isArray(value) ? value.map(textOf).filter((text): text is string => text !== null) : []
}

function factValue(value: unknown): string | null {
  if (typeof value === 'number') return Number.isInteger(value) && value > 0 ? NUMBER_FORMAT.format(value) : null
  return textOf(value)
}

/** An `https` logo, or a short text such as an emoji; a name like "bot" or any other URL is none. */
function iconOf(value: unknown): string | null {
  const icon = textOf(value)
  if (!icon) return null
  if (icon.startsWith(HTTPS_PREFIX)) return /\s/.test(icon) ? null : icon
  return Array.from(icon).length <= SHORT_ICON_MAX && !LETTERS_OR_DIGITS.test(icon) ? icon : null
}

function factsOf(item: Json, type: PublicMarketplaceType): PublicItemFact[] {
  return FACTS[type].flatMap(([field, label]) => {
    const value = factValue(item[field])
    return value ? [{ label, value }] : []
  })
}

function memberLine(member: unknown): string | null {
  if (!isRecord(member)) return null
  const type = textOf(member.type)
  const label = type ? MEMBER_LABELS[type] ?? type : null
  const name = textOf(member.name)
  return label && name ? `${label} · ${name}` : label
}

function capabilityLine(capability: unknown): string | null {
  if (!isRecord(capability)) return null
  const name = textOf(capability.name)
  const rating = textOf(capability.rating)
  return name && rating ? `${name} · ${rating}` : name
}

function includesOf(item: Json, type: PublicMarketplaceType): string[] {
  const entries = type === 'packages' ? item.members : type === 'llms' ? item.capabilities : null
  const line = type === 'packages' ? memberLine : capabilityLine
  return Array.isArray(entries) ? entries.map(line).filter((text): text is string => text !== null) : []
}

/** One public item, or null when it has no key or name. */
export function publicItem(value: unknown, type: PublicMarketplaceType): PublicItem | null {
  if (!isRecord(value)) return null
  const key = textOf(value.key)
  const name = textOf(value.name)
  if (!key || !name) return null
  return {
    key,
    name,
    description: textOf(value.description),
    category: textOf(value.category),
    icon: iconOf(value.icon),
    tags: textsOf(value.tags),
    publisher: textOf(value.publisher) ?? '',
    requiredApps: textsOf(value.requiredApps),
    facts: factsOf(value, type),
    includes: includesOf(value, type),
  }
}

/** One page of the read at `offset`; an answer that isn't the type's page reads as an empty last page. */
export function publicPage(answer: unknown, type: PublicMarketplaceType, offset: number): PublicPage {
  if (!isRecord(answer) || answer.type !== type || !Array.isArray(answer.items)) return { items: [], next: null }
  const items = answer.items
    .map((entry) => publicItem(entry, type))
    .filter((item): item is PublicItem => item !== null)
  const more = answer.more === true && answer.items.length > 0
  return { items, next: more ? offset + answer.items.length : null }
}

function searchText(item: PublicItem): string {
  return [item.name, item.description, item.category, item.publisher, ...item.tags, ...item.requiredApps]
    .filter((part): part is string => typeof part === 'string')
    .join(' ')
    .toLowerCase()
}

/** The items of every page read so far, the first of each key only (a page can repeat one as the list moves). */
export function pagedItems(pages: readonly PublicPage[]): PublicItem[] {
  const byKey = new Map<string, PublicItem>()
  for (const item of pages.flatMap((page) => page.items)) {
    if (!byKey.has(item.key)) byKey.set(item.key, item)
  }
  return Array.from(byKey.values())
}

/** The items whose name, description, category, publisher, tags or apps hold every word of the search. */
export function filterPublicItems(items: readonly PublicItem[], query: string): PublicItem[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (words.length === 0) return [...items]
  return items.filter((item) => {
    const text = searchText(item)
    return words.every((word) => text.includes(word))
  })
}
