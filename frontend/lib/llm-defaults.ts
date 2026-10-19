/**
 * A new agent's model settings, and where its model comes from.
 *
 * The model is data: the platform default (system_settings orchestrator_llm), which a
 * platform admin sets in System Settings > System & Embeddings. The UI reads it from
 * GET /api/models/platform-default (the backend's default_route) into a module-level
 * cache and keeps no provider or model id of its own, so changing the default needs no
 * frontend release; orchestrator/tests/test_auto_model_from_config_frontend_parity.py
 * fails if one comes back. Only the sampling defaults below live here.
 *
 * LLM_DEFAULTS.provider and .model_id are getters on that cache. Read before the API
 * has answered (the first paint after a page load, or a read that went out before the
 * auth token was installed) they are '' and the read is retried. A model_config that
 * names no model is harmless: the backend refuses to save one (422,
 * api/agent_model_config.py), so the agent keeps the model it has.
 */
import { apiClient } from '@/lib/api-client'

const PLATFORM_DEFAULT_ROUTE_PATH = '/api/models/platform-default'

/** A model route: the model and the provider that serves it. */
export interface ModelRoute {
  provider: string
  model_id: string
}

/** No model picked yet: the platform default fills it in once the API answers. */
export const NO_MODEL_ROUTE: ModelRoute = { provider: '', model_id: '' }

/** A new agent's model_config: the platform default's route and the sampling defaults. */
export interface LLMDefaults extends ModelRoute {
  temperature: number
  max_tokens: number
  top_p: number
  frequency_penalty: number
  presence_penalty: number
  fallback_model_id: string | null
}

// The platform default as last read from the API, and the read in flight (one at a time).
let platformDefaultRoute: ModelRoute = NO_MODEL_ROUTE
let priming: Promise<void> | null = null

/** Remember the platform default route (a reply naming no model leaves the cache as it was). */
export function setPlatformDefaultRoute(route: ModelRoute | null | undefined): void {
  if (route?.model_id) platformDefaultRoute = { provider: route.provider, model_id: route.model_id }
}

/** The platform default route, from the API. */
export function fetchPlatformDefaultRoute(): Promise<ModelRoute> {
  return apiClient.get<ModelRoute>(PLATFORM_DEFAULT_ROUTE_PATH)
}

/**
 * Read the platform default into the cache. Concurrent calls share one read; a read
 * that leaves the cache empty (a failure, a reply naming no model, a stub apiClient
 * that throws or returns no promise) is swallowed, and the next call reads again.
 */
export function primePlatformDefaultRoute(): Promise<void> {
  if (!priming) {
    priming = Promise.resolve()
      .then(() => fetchPlatformDefaultRoute())
      .then((route) => setPlatformDefaultRoute(route))
      .catch(() => undefined)
      .then(() => { if (!platformDefaultRoute.model_id) priming = null })
  }
  return priming
}

/** The cached route; an empty cache on the client asks the API (again). */
function cachedRoute(): ModelRoute {
  if (!platformDefaultRoute.model_id && typeof window !== 'undefined') void primePlatformDefaultRoute()
  return platformDefaultRoute
}

export const LLM_DEFAULTS: LLMDefaults = {
  get provider() { return cachedRoute().provider },
  get model_id() { return cachedRoute().model_id },
  temperature: 0.7,
  max_tokens: 2000,
  top_p: 1.0,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  fallback_model_id: null,
}

/**
 * A fresh model_config for useState initializers: a snapshot (the spread reads the
 * getters once) on ``route`` when given, else on the cached platform default.
 */
export function getDefaultModelConfig(route?: ModelRoute): LLMDefaults {
  return { ...LLM_DEFAULTS, ...route }
}

if (typeof window !== 'undefined') void primePlatformDefaultRoute()
