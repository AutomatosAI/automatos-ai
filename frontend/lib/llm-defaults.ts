/**
 * Single source of truth for LLM defaults across the frontend.
 *
 * The REAL source of truth is Settings > Orchestrator (Auto agent row).
 * These defaults are only used as fallbacks when the API hasn't responded
 * yet or when creating brand-new agents before the user picks a model.
 *
 * Change here → changes everywhere.
 */
export const LLM_DEFAULTS = {
  provider: 'openrouter',
  model_id: 'google/gemini-2.5-flash',
  temperature: 0.7,
  max_tokens: 2000,
  top_p: 1.0,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  fallback_model_id: null as string | null,
} as const

/** Fresh copy for useState initializers (avoids shared mutation). */
export function getDefaultModelConfig() {
  return { ...LLM_DEFAULTS }
}
