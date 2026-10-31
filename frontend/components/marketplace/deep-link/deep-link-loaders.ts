/**
 * PRD-270 US-270-09b (K22) — a deep link's item, read through the route its tab already calls, and
 * the two adds the deep link makes itself: an application's and a plugin's live inside their
 * untouchable tab components (K4), so these repeat that one request each, on the person's click.
 *
 * A key that isn't one of its kind (an agent's or a Playbook's id is a number, a plugin's a UUID, an
 * LLM's route `<serving provider>:<model id>`) reads as not found with no request, so a crafted link
 * never reaches another path, and a key always rides encoded. Not found (a 404, or a list without
 * it) is null, which opens nothing; any other failure throws, for the opener to report.
 */
import type { ComposioApp } from '@/hooks/use-composio-api'
import { apiClient } from '@/lib/api-client'
import type { ApiHttpError } from '@/lib/api-client'

import type { LLMModel } from '../llm-model-card'
import type { MarketplacePackage } from '../package-types'

const NOT_FOUND = 404
const ITEM_ID = /^[1-9]\d{0,9}$/
const SLUG = /^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$/
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const LLM_ROUTE = /^([a-z0-9][a-z0-9_-]{0,63}):(\S+)$/
/** The Applications tab asks its catalogue read for this many (marketplace-tools-tab.tsx). */
const APPS_LIMIT = '1000'
/** The catalogue read's own default page: one model id has few routes on one provider. */
const LLM_ROUTES_LIMIT = '200'
const ACTIVE_TOOL = 'active'

/** The Applications tab's query of this workspace's tools (marketplace-tools-tab.tsx): one cache. */
export const WORKSPACE_TOOLS_KEY = ['workspace-tools'] as const
export const WORKSPACE_TOOLS_STALE_MS = 30 * 1000

/** The two item types MarketplaceItemModal opens (the member routes' wire types). */
export type DeepLinkItemType = 'agent' | 'recipe'

export interface FoundItem {
  readonly id: number
}

export interface FoundPlugin {
  readonly id: string
  readonly name: string
}

/** One app of `GET /api/tools/marketplace` (api/tools.py AppOut). */
interface CatalogueApp {
  readonly app_name?: string
  readonly display_name?: string
  readonly description?: string | null
  readonly logo_url?: string | null
  readonly categories?: string[]
  readonly is_connected?: boolean
  readonly action_count?: number
  readonly trigger_count?: number
  readonly auth_schemes?: string[]
  readonly triggers?: ComposioApp['triggers']
}

async function orNotFound<T>(read: Promise<T>): Promise<T | null> {
  try {
    return await read
  } catch (error) {
    if ((error as ApiHttpError | null)?.status === NOT_FOUND) return null
    throw error
  }
}

/** An agent or a Playbook by its id, through the homepage's own detail read. */
export async function loadMarketplaceItem(key: string, type: DeepLinkItemType): Promise<FoundItem | null> {
  if (!ITEM_ID.test(key)) return null
  const item = await orNotFound(apiClient.get<{ id?: unknown }>(`/api/marketplace/items/${key}?type=${type}`))
  return typeof item?.id === 'number' ? { id: item.id } : null
}

/** A package by its slug (`GET /api/marketplace/packages/{slug}`, the popup's own deep-link read). */
export async function loadPackage(slug: string): Promise<MarketplacePackage | null> {
  if (!SLUG.test(slug)) return null
  const found = await orNotFound(apiClient.get<MarketplacePackage>(`/api/marketplace/packages/${encodeURIComponent(slug)}`))
  return typeof found?.slug === 'string' ? found : null
}

/** The Applications tab's own mapping of a catalogue app to the shape its details modal takes. */
function composioAppOf(app: CatalogueApp, name: string): ComposioApp {
  return {
    name,
    display_name: app.display_name ?? name,
    description: app.description ?? undefined,
    logo_url: app.logo_url ?? undefined,
    categories: app.categories ?? [],
    is_connected: app.is_connected ?? false,
    action_count: app.action_count ?? 0,
    trigger_count: app.trigger_count ?? 0,
    auth_schemes: app.auth_schemes ?? [],
    triggers: app.triggers ?? [],
  }
}

/** An application by its slug (the public read's key: its app name, lower case). */
export async function loadApplication(slug: string): Promise<ComposioApp | null> {
  if (!SLUG.test(slug)) return null
  const query = new URLSearchParams({ search: slug, limit: APPS_LIMIT })
  const answer = await apiClient.get<{ apps?: CatalogueApp[] }>(`/api/tools/marketplace?${query}`)
  const wanted = slug.toLowerCase()
  const app = (answer?.apps ?? []).find((entry) => entry.app_name?.toLowerCase() === wanted)
  return app?.app_name ? composioAppOf(app, app.app_name) : null
}

/** One tool of `GET /api/tools/connected`, as the Applications tab reads it. */
export interface WorkspaceTool {
  readonly app_name?: string
  readonly status?: string
}

/** This workspace's tools, in the Applications tab's query shape (its `apps`). */
export async function workspaceTools(): Promise<WorkspaceTool[]> {
  const answer = await apiClient.get<{ apps?: WorkspaceTool[] } | null>('/api/tools/connected')
  return answer?.apps ?? []
}

/** Connected as the Applications tab shows it: an active workspace tool of the app's name. */
export function isConnectedApp(tools: readonly WorkspaceTool[], app: ComposioApp): boolean {
  const name = app.name.toUpperCase()
  return tools.some((tool) => tool.status === ACTIVE_TOOL && (tool.app_name ?? '').toUpperCase() === name)
}

/** Adds an application as its tab does; true when it was already in the workspace. */
export async function addApplication(app: ComposioApp): Promise<boolean> {
  const result = await apiClient.post<{ status?: string } | null>('/api/tools/add-to-workspace', { app_name: app.name })
  return result?.status === 'already_added'
}

/** A plugin by its id (the read its detail modal makes again once open). */
export async function loadPlugin(id: string): Promise<FoundPlugin | null> {
  if (!UUID.test(id)) return null
  const plugin = await orNotFound(apiClient.get<{ id?: unknown; name?: unknown }>(`/api/marketplace/plugins/${encodeURIComponent(id)}`))
  if (typeof plugin?.id !== 'string') return null
  return { id: plugin.id, name: typeof plugin.name === 'string' ? plugin.name : '' }
}

/** The plugins this workspace has enabled, as the Plugins tab reads them. */
export async function enabledPluginIds(workspaceId: string): Promise<ReadonlySet<string>> {
  const answer = await apiClient.get<{ items?: { plugin_id?: unknown }[] }>(
    `/api/workspaces/${encodeURIComponent(workspaceId)}/plugins`,
  )
  return new Set((answer?.items ?? []).flatMap((item) => (typeof item.plugin_id === 'string' ? [item.plugin_id] : [])))
}

/** Enables a plugin for the workspace, as its tab does. */
export async function enablePlugin(workspaceId: string, pluginId: string): Promise<void> {
  await apiClient.post(`/api/workspaces/${encodeURIComponent(workspaceId)}/plugins`, { plugin_id: pluginId })
}

/** An LLM route by its key, from the LLMs tab's catalogue read filtered to its provider and model. */
export async function loadLlmRoute(key: string): Promise<LLMModel | null> {
  const route = LLM_ROUTE.exec(key)
  if (route === null) return null
  const [, provider, modelId] = route
  const query = new URLSearchParams({ provider, search: modelId, limit: LLM_ROUTES_LIMIT })
  const answer = await apiClient.get<{ models?: LLMModel[] }>(`/api/marketplace/llm/catalog?${query}`)
  const routes = answer?.models ?? []
  return routes.find((model) => model.model_id === modelId && (model.serving_provider || model.provider) === provider) ?? null
}
