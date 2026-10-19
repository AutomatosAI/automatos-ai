/**
 * The frontend's copy of the backend's last-resort LLM default.
 *
 * The live default is data: system_settings orchestrator_llm (Settings >
 * Orchestrator), which each Auto row carries in its model_config. These values
 * are only fallbacks while the API hasn't answered, and the pair "reset to
 * defaults" writes.
 *
 * provider and model_id MUST equal DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL in
 * orchestrator/core/llm/defaults.py until this copy is replaced by an API read;
 * orchestrator/tests/test_auto_model_from_config_frontend_parity.py fails when
 * they differ.
 */
export const LLM_DEFAULTS = {
  provider: 'anthropic',
  model_id: 'claude-haiku-5-5',
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
