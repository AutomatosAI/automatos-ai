/**
 * 2026-09-09 — the Analytics page reads one vocabulary: routes, providers,
 * billing, and period-scoped agent spend.
 */
import { describe, expect, it } from 'vitest'

import {
  agentRuntimeFacts,
  billingBadge,
  cacheShare,
  mergeAgentUsage,
  meteredCost,
  routeLabel,
  shortenModelName,
  splitRouteKey,
} from '@/lib/analytics-usage'

describe('routes and billing', () => {
  it('labels a route as short model · provider and keeps free and paid apart', () => {
    expect(splitRouteKey('moonshotai/kimi-k3@nvidia')).toEqual({ model: 'moonshotai/kimi-k3', provider: 'nvidia' })
    expect(splitRouteKey('claude-fable-5')).toEqual({ model: 'claude-fable-5', provider: null })
    expect(routeLabel({ key: 'moonshotai/kimi-k3@nvidia', provider_label: 'NVIDIA' })).toBe('kimi-k3 · NVIDIA')
    expect(routeLabel({ key: 'claude-fable-5@claude_code', provider_label: 'Claude Code' })).toBe('claude-fable-5 · Claude Code')
    expect(shortenModelName('google/gemini-2.5-flash')).toBe('gemini-2.5-flash')
    expect(shortenModelName(null)).toBe('unknown')
  })

  it('names the billing of every provider kind', () => {
    expect(billingBadge('metered')).toEqual({ label: 'Paid API', tone: 'paid' })
    expect(billingBadge('free')).toEqual({ label: 'Free', tone: 'free' })
    expect(billingBadge('subscription')).toEqual({ label: 'Subscription', tone: 'subscription' })
    expect(billingBadge(undefined)).toEqual({ label: 'Unknown', tone: 'unknown' })
  })

  it('sums cache share and metered spend', () => {
    expect(cacheShare([
      { key: 'a', request_count: 1, input_tokens: 800, output_tokens: 0, total_tokens: 800, total_cost: 0, cache_read_tokens: 600 },
      { key: 'b', request_count: 1, input_tokens: 200, output_tokens: 0, total_tokens: 200, total_cost: 0, cache_read_tokens: 0 },
    ])).toBeCloseTo(0.6)
    expect(cacheShare([])).toBe(0)
    expect(meteredCost([
      { provider: 'openrouter', label: 'OpenRouter', billing: 'metered', kind: 'aggregator', request_count: 1, total_tokens: 1, total_cost: 1.25 },
      { provider: 'nvidia', label: 'NVIDIA', billing: 'free', kind: 'hosted_open', request_count: 1, total_tokens: 1, total_cost: 0 },
      { provider: 'claude_code', label: 'Claude Code', billing: 'subscription', kind: 'runtime', request_count: 1, total_tokens: 1, total_cost: 0 },
    ])).toBe(1.25)
  })
})

describe('agents', () => {
  const bob = { id: 15, name: 'Bob', status: 'active', configuration: { runtime: 'cli', provider: 'claude', model: 'fable' }, agent_model_config: { provider: 'openai', model_id: 'gpt-4o' } }
  const researcher = { id: 57, name: 'Researcher', status: 'active', configuration: { runtime: 'api' }, agent_model_config: { provider: 'nvidia', model_id: 'moonshotai/kimi-k3' } }

  it('shows a session agent as its CLI, never its unused API route', () => {
    expect(agentRuntimeFacts(bob)).toMatchObject({ runtime: 'cli', provider: 'claude_code', label: 'Claude Code · fable', billing: 'subscription' })
    expect(agentRuntimeFacts(researcher)).toMatchObject({ runtime: 'api', provider: 'nvidia', label: 'kimi-k3', billing: 'free' })
    expect(agentRuntimeFacts({})).toMatchObject({ runtime: 'api', label: 'unknown', billing: 'metered' })
  })

  it('joins the period usage onto agents and ranks by cost then tokens', () => {
    const rows = mergeAgentUsage([bob, researcher], [
      { key: '57', request_count: 3, input_tokens: 900, output_tokens: 100, total_tokens: 1000, total_cost: 0 },
      { key: '15', request_count: 2, input_tokens: 236000, output_tokens: 700, total_tokens: 236700, total_cost: 0, cache_read_tokens: 117000 },
      { key: '999', request_count: 1, input_tokens: 10, output_tokens: 1, total_tokens: 11, total_cost: 0.5 },
      { key: '1', request_count: 211, input_tokens: 4, output_tokens: 1, total_tokens: 5, total_cost: 0.4, label: 'Auto', model_id: 'anthropic/claude-opus-4.6', provider: 'openrouter', provider_label: 'OpenRouter', billing: 'metered' },
      { key: 'unknown', request_count: 4, input_tokens: 1, output_tokens: 1, total_tokens: 2, total_cost: 9 },
    ])
    expect(rows.map((r) => r.name)).toEqual(['Agent #999', 'Auto', 'Bob', 'Researcher'])
    expect(rows[2]).toMatchObject({ runtime: 'cli', modelLabel: 'Claude Code · fable', cacheReadTokens: 117000, billing: 'subscription' })
    expect(rows[0].status).toBe('deleted')
    // the system agent is not in the workspace list: the backend's label names it,
    // and the route it actually used this period is what "runs on" shows
    expect(rows[1]).toMatchObject({ status: 'system', modelLabel: 'claude-opus-4.6 · OpenRouter', billing: 'metered' })
  })

  it('prefers the route an agent actually used over its configured model', () => {
    const rows = mergeAgentUsage([researcher], [
      { key: '57', request_count: 3, input_tokens: 9, output_tokens: 1, total_tokens: 10, total_cost: 0.01, model_id: 'moonshotai/kimi-k3', provider: 'openrouter', provider_label: 'OpenRouter', billing: 'metered' },
    ])
    expect(rows[0]).toMatchObject({ modelLabel: 'kimi-k3 · OpenRouter', billing: 'metered' })
  })
})
