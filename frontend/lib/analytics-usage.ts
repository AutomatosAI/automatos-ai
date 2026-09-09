/**
 * Analytics usage helpers (2026-09-09 cost tracking).
 *
 * Pure functions over the LLM analytics API shapes so the tabs read one
 * vocabulary: a ROUTE is model × serving provider (the same vendor model on
 * NVIDIA for free and on OpenRouter for $3/M is two routes), a provider's
 * BILLING is metered / free / subscription, and an agent's spend for the
 * selected period comes from ``llm_usage`` rows — never from the cumulative
 * ``model_usage_stats`` blob, which ignores the time range.
 */

export type Billing = 'metered' | 'free' | 'subscription' | 'unknown'

export interface UsageGroup {
  key: string
  request_count: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
  total_cost: number
  cache_read_tokens?: number
  cache_write_tokens?: number
  error_count?: number
  avg_latency_ms?: number | null
  model_id?: string | null
  provider?: string | null
  provider_label?: string | null
  billing?: Billing | string | null
  label?: string | null
}

export interface ProviderUsage {
  provider: string
  label: string
  billing: Billing | string
  kind: string
  request_count: number
  total_tokens: number
  total_cost: number
  error_count?: number
}

export interface BillingBadge {
  label: string
  tone: 'paid' | 'free' | 'subscription' | 'unknown'
}

export function billingBadge(billing: string | null | undefined): BillingBadge {
  switch ((billing || '').toLowerCase()) {
    case 'metered':
      return { label: 'Paid API', tone: 'paid' }
    case 'free':
      return { label: 'Free', tone: 'free' }
    case 'subscription':
      return { label: 'Subscription', tone: 'subscription' }
    default:
      return { label: 'Unknown', tone: 'unknown' }
  }
}

/** ``anthropic/claude-opus-4.6`` → ``claude-opus-4.6``; a bare id stays. */
export function shortenModelName(name: string | null | undefined): string {
  if (!name) return 'unknown'
  const slash = name.lastIndexOf('/')
  return slash >= 0 ? name.slice(slash + 1) : name
}

/** ``<model>@<provider>`` → ``{ model, provider }``; a bare key has no provider. */
export function splitRouteKey(key: string): { model: string; provider: string | null } {
  const at = key.lastIndexOf('@')
  if (at < 0) return { model: key, provider: null }
  return { model: key.slice(0, at), provider: key.slice(at + 1) }
}

/** The legend/table label for a route group: short model · provider. */
export function routeLabel(group: Pick<UsageGroup, 'key' | 'model_id' | 'provider_label' | 'provider'>): string {
  const { model, provider } = splitRouteKey(group.key)
  const modelName = shortenModelName(group.model_id || model)
  const providerName = group.provider_label || group.provider || provider
  return providerName ? `${modelName} · ${providerName}` : modelName
}

export interface AgentRuntimeFacts {
  runtime: 'api' | 'cli'
  provider: string | null
  model: string | null
  /** What the Model column shows: the route for an API agent, the CLI for a session agent. */
  label: string
  billing: Billing
}

/**
 * Where an agent's calls go. A session agent (``configuration.runtime = cli``)
 * runs on the user's own Claude Code / Codex plan — its ``agent_model_config``
 * (an API route) is irrelevant and used to be shown as the model.
 */
export function agentRuntimeFacts(agent: any): AgentRuntimeFacts {
  const configuration = agent?.configuration || {}
  const runtime = String(configuration.runtime || '').toLowerCase() === 'cli' ? 'cli' : 'api'
  if (runtime === 'cli') {
    const cli = String(configuration.provider || 'claude').toLowerCase()
    const cliLabel = cli === 'codex' ? 'Codex' : 'Claude Code'
    const model = configuration.model ? String(configuration.model) : null
    return {
      runtime,
      provider: cli === 'codex' ? 'codex' : 'claude_code',
      model,
      label: `${cliLabel}${model ? ` · ${model}` : ''}`,
      billing: 'subscription',
    }
  }
  const modelConfig = agent?.agent_model_config || {}
  const model = modelConfig.model_id ? String(modelConfig.model_id) : null
  const provider = modelConfig.provider ? String(modelConfig.provider) : null
  return {
    runtime,
    provider,
    model,
    label: model ? shortenModelName(model) : 'unknown',
    billing: provider === 'nvidia' ? 'free' : 'metered',
  }
}

export interface AgentUsageRow {
  id: number
  name: string
  status: string
  runtime: 'api' | 'cli'
  model: string
  modelLabel: string
  billing: Billing
  requests: number
  tokens: number
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cost: number
  errors: number
}

/**
 * Join the period's per-agent usage (``/usage?group_by=agent``) onto the agent
 * list. Every agent with usage appears, including one deleted since (as
 * ``Agent #<id>``); sorted by cost, then tokens, so a subscription agent with
 * no dollars still ranks by the work it did.
 */
export function mergeAgentUsage(agents: any[], usageByAgent: UsageGroup[]): AgentUsageRow[] {
  const byId = new Map<string, any>()
  for (const agent of agents || []) byId.set(String(agent.id), agent)
  const rows: AgentUsageRow[] = []
  for (const group of usageByAgent || []) {
    if (group.key === 'unknown' || group.key === 'None') continue
    const agent = byId.get(group.key)
    const facts = agentRuntimeFacts(agent || {})
    // The route the agent actually used most this period (from llm_usage) beats
    // its configuration; the backend's label names agents the list does not
    // carry (the system agent Auto, a deleted agent).
    const usedRoute = group.model_id && group.provider
      ? { model: group.model_id, label: `${shortenModelName(group.model_id)} · ${group.provider_label || group.provider}`, billing: (group.billing as Billing) || facts.billing }
      : null
    rows.push({
      id: Number(group.key),
      name: agent?.name || group.label || `Agent #${group.key}`,
      status: agent?.status || (group.label ? 'system' : 'deleted'),
      runtime: facts.runtime,
      model: usedRoute?.model || facts.model || 'unknown',
      modelLabel: usedRoute?.label || facts.label,
      billing: usedRoute?.billing || facts.billing,
      requests: group.request_count || 0,
      tokens: group.total_tokens || 0,
      inputTokens: group.input_tokens || 0,
      outputTokens: group.output_tokens || 0,
      cacheReadTokens: group.cache_read_tokens || 0,
      cost: group.total_cost || 0,
      errors: group.error_count || 0,
    })
  }
  return rows.sort((a, b) => (b.cost - a.cost) || (b.tokens - a.tokens))
}

/** Share of all prompt tokens that were served from the providers' caches. */
export function cacheShare(routes: UsageGroup[]): number {
  let input = 0
  let cached = 0
  for (const r of routes || []) {
    input += r.input_tokens || 0
    cached += r.cache_read_tokens || 0
  }
  return input > 0 ? cached / input : 0
}

/** Spend that the operator actually pays for (metered routes only). */
export function meteredCost(providers: ProviderUsage[]): number {
  return (providers || []).reduce((sum, p) => sum + (p.billing === 'metered' ? p.total_cost || 0 : 0), 0)
}
