/**
 * One Marketplace page for everyone (PRD-270 §5.7) — the public reads of Applications, Plugins,
 * Skills and LLMs in the shapes the member tabs' own cards draw (ToolCard, PluginCard, SkillGridCard,
 * LLMModelCard).
 *
 * Only what the public item says is mapped: its display fields (checked by `publicItem`) and the
 * type's plain facts from the raw answer (an application's action and trigger counts, a plugin's
 * version and counts, a skill's version, an LLM route's provider, vendor, context window, price tier
 * and capability ratings). What the member shape wants and the public read doesn't carry is left
 * empty, never made up: no auth schemes, no token estimates, no prices, no max output. Counts of
 * use (a plugin's enables, a model's installs) are 0 here and the cards don't show them signed out
 * (useMarketplaceAccess); a 0 token estimate is one the skill cards already hide.
 */
import type { ComposioApp } from '@/hooks/use-composio-api'
import { countOf, isJson } from '@/lib/marketplace/source/public-reads'
import type { Json, PublicEntry } from '@/lib/marketplace/source/public-reads'

import type { LLMModel } from '../llm-model-card'
import type { PluginSummary } from '../marketplace-plugins-tab'
import type { Skill } from '../marketplace-skills-tab'

/** An LLM route's key, `<serving provider>:<model id>` (api/llm_marketplace.py route_key). */
const ROUTE_SEPARATOR = ':'
/** The price tier the catalogue gives a free route (api/llm_marketplace.py price_tier_for). */
const FREE_TIER = 'free'
const ACTIVE_ROUTE = 'active'
const NOT_RECORDED = ''

function textOf(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

/** An application as the Applications tab's ToolCard draws it. */
export function appOf({ item, raw }: PublicEntry): ComposioApp {
  return {
    name: item.key,
    display_name: item.name,
    description: item.description ?? undefined,
    logo_url: item.icon?.startsWith('https://') ? item.icon : undefined,
    categories: [...item.tags],
    is_connected: false,
    action_count: countOf(raw.actionCount),
    trigger_count: countOf(raw.triggerCount),
    auth_schemes: [],
    triggers: [],
  }
}

/** A plugin as the Capabilities › Plugins cards draw it. */
export function pluginOf({ item, raw }: PublicEntry): PluginSummary {
  return {
    id: item.key,
    slug: item.key,
    name: item.name,
    version: textOf(raw.version) ?? NOT_RECORDED,
    description: item.description,
    category_id: null,
    category_name: item.category,
    tags: [...item.tags],
    skills_count: countOf(raw.skillsCount),
    commands_count: countOf(raw.commandsCount),
    agents_count: countOf(raw.agentsCount),
    hooks_count: countOf(raw.hooksCount),
    token_estimate: 0,
    enable_count: 0,
    is_featured: false,
    security_status: null,
    approval_status: null,
    original_author: item.publisher || null,
    license: null,
    created_at: null,
  }
}

/** A skill as the Capabilities › Skills cards draw it; `id` is its place in the list (cards never show it). */
export function skillOf({ item, raw }: PublicEntry, index: number): Skill {
  return {
    id: index,
    name: item.name,
    description: item.description,
    category: item.category,
    skill_version: textOf(raw.version),
    tags: [...item.tags],
    estimated_tokens: 0,
    skill_source: null,
    is_enabled: false,
  }
}

function ratingsOf(value: unknown): Record<string, string> {
  if (!Array.isArray(value)) return {}
  const pairs = value.filter(isJson).flatMap((entry: Json) => {
    const name = textOf(entry.name)
    const rating = textOf(entry.rating)
    return name && rating ? [[name, rating] as const] : []
  })
  return Object.fromEntries(pairs)
}

/** A route key's two parts: the serving provider before the first separator, the model id after it. */
function routeOf(key: string): { provider: string | undefined; modelId: string } {
  const at = key.indexOf(ROUTE_SEPARATOR)
  return at > 0 ? { provider: key.slice(0, at), modelId: key.slice(at + 1) } : { provider: undefined, modelId: key }
}

/** An LLM route as the LLMs tab's LLMModelCard draws it; `id` is its place in the list. */
export function llmOf({ item, raw }: PublicEntry, index: number): LLMModel {
  const route = routeOf(item.key)
  // The key's provider is the route's own, so the card's link (routeKeyOf) is the item's key.
  const servingProvider = route.provider ?? textOf(raw.provider) ?? undefined
  const priceTier = textOf(raw.priceTier) ?? undefined
  return {
    id: index,
    provider: textOf(raw.vendor) ?? servingProvider ?? NOT_RECORDED,
    vendor: textOf(raw.vendor) ?? undefined,
    serving_provider: servingProvider,
    serving_provider_label: item.publisher || undefined,
    is_free: priceTier === FREE_TIER,
    price_tier: priceTier,
    model_id: route.modelId,
    display_name: item.name,
    model_family: null,
    description: item.description,
    context_window: countOf(raw.contextWindow),
    max_output_tokens: 0,
    input_cost_per_1k: 0,
    output_cost_per_1k: 0,
    capabilities: ratingsOf(raw.capabilities),
    recommended_for: [],
    supports_functions: false,
    supports_vision: false,
    supports_streaming: false,
    status: ACTIVE_ROUTE,
    category: item.category,
    tags: [...item.tags],
    is_featured: false,
    is_default: false,
    requires_plan: null,
    install_count: 0,
    is_installed: false,
  }
}
