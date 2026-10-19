/**
 * A new agent's model settings, and where its model comes from.
 *
 * The model is data: the platform default (system_settings orchestrator_llm), which a
 * platform admin sets in System Settings > LLM tiers. The UI reads it from
 * GET /api/models/platform-default (the backend's default_route) and keeps no
 * provider or model id of its own, so changing the default needs no frontend
 * release; orchestrator/tests/test_auto_model_from_config_frontend_parity.py fails
 * if one comes back. Only the sampling defaults below live here.
 */
import { apiClient } from '@/lib/api-client'

export const PLATFORM_DEFAULT_ROUTE_PATH = '/api/models/platform-default'

/** A model route: the model and the provider that serves it. */
export interface ModelRoute {
  provider: string
  model_id: string
}

/** No model picked yet: the platform default fills it in once the API answers. */
export const NO_MODEL_ROUTE: ModelRoute = { provider: '', model_id: '' }

export const LLM_DEFAULTS = {
  temperature: 0.7,
  max_tokens: 2000,
  top_p: 1.0,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  fallback_model_id: null as string | null,
} as const

/** A fresh model_config on ``route`` (default: none yet), for useState initializers. */
export function getDefaultModelConfig(route: ModelRoute = NO_MODEL_ROUTE) {
  return { provider: route.provider, model_id: route.model_id, ...LLM_DEFAULTS }
}

/** ``config`` on the platform default's route when it names no model; a NEW object. */
export function withDefaultRoute<T extends { model_id?: string }>(config: T, route?: ModelRoute): T {
  if (config.model_id || !route) return config
  return { ...config, provider: route.provider, model_id: route.model_id }
}

/**
 * ``config`` as a save may send it: on the platform default's route when it names no
 * model, or null while it still names none (the route not loaded yet). A blank model
 * is never saved: the backend would store it over the agent's real one.
 */
export function savableModelConfig<T extends { model_id?: string }>(config: T | null | undefined, route?: ModelRoute): T | null {
  if (!config) return null
  const routed = withDefaultRoute(config, route)
  return routed.model_id ? routed : null
}

/** The platform default route, from the API. */
export function fetchPlatformDefaultRoute(): Promise<ModelRoute> {
  return apiClient.get<ModelRoute>(PLATFORM_DEFAULT_ROUTE_PATH)
}
