/**
 * PRD-239 S5 — the agent model picker is keyed by ROUTE: "Kimi K3 via NVIDIA"
 * and "Kimi K3 via OpenRouter" are two picks, the stored pair resolves to the
 * exact route, and a saved model that is gone says so.
 */
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

const hook = vi.hoisted(() => ({ useWorkspaceModels: vi.fn() }))
vi.mock('@/hooks/use-model-api', () => ({ useWorkspaceModels: hook.useWorkspaceModels }))
vi.mock('framer-motion', () => ({ motion: { div: (p: any) => <div>{p.children}</div> } }))

import { ModelSelector, resolveRoute, routeKey, servingProviderOf } from '../model-selector'
import type { ModelInfo } from '@/hooks/use-model-api'

function row(over: Partial<ModelInfo>): ModelInfo {
  return {
    id: 1,
    provider: 'moonshotai',
    serving_provider: 'openrouter',
    serving_provider_label: 'OpenRouter',
    model_id: 'moonshotai/kimi-k3',
    display_name: 'Kimi K3',
    model_family: 'moonshotai',
    context_window: 262144,
    max_output_tokens: 32768,
    input_cost_per_1k: 0.003,
    output_cost_per_1k: 0.015,
    capabilities: {},
    recommended_for: [],
    supports_functions: true,
    supports_vision: false,
    supports_streaming: true,
    status: 'active',
    tier: 'openrouter',
    ...over,
  }
}

const kimiOpenRouter = row({ id: 1555 })
const kimiNvidia = row({ id: 1114, serving_provider: 'nvidia', serving_provider_label: 'NVIDIA', tier: 'byok', input_cost_per_1k: 0 })
const opus = row({ id: 1306, provider: 'anthropic', model_id: 'anthropic/claude-opus-4.6', display_name: 'Claude Opus 4.6' })
const models = [kimiOpenRouter, kimiNvidia, opus]

afterEach(() => {
  cleanup()
  hook.useWorkspaceModels.mockReset()
})

describe('route helpers', () => {
  it('keys an option by route and model id, like the backend installed-ids', () => {
    expect(routeKey('nvidia', 'moonshotai/kimi-k3')).toBe('nvidia:moonshotai/kimi-k3')
    expect(servingProviderOf({ serving_provider: undefined, provider: 'openai' })).toBe('openai')
  })

  it('resolves the stored pair to the exact route, else the first route offering the id', () => {
    expect(resolveRoute(models, 'moonshotai/kimi-k3', 'nvidia')).toBe(kimiNvidia)
    expect(resolveRoute(models, 'moonshotai/kimi-k3', 'openrouter')).toBe(kimiOpenRouter)
    expect(resolveRoute(models, 'moonshotai/kimi-k3', undefined)).toBe(kimiOpenRouter)
    expect(resolveRoute(models, 'deepseek/deepseek-coder', 'openrouter')).toBeNull()
    expect(resolveRoute(undefined, 'x', 'y')).toBeNull()
  })
})

describe('ModelSelector', () => {
  it('shows the stored route and its serving provider', () => {
    hook.useWorkspaceModels.mockReturnValue({ data: models, isLoading: false, error: null })
    render(<ModelSelector value="moonshotai/kimi-k3" provider="nvidia" onChange={vi.fn()} />)
    expect(screen.queryByTestId('model-unavailable')).toBeNull()
    expect(screen.getByText(/served by/)).toHaveTextContent('served by NVIDIA')
  })

  it('says so when the saved model is not installed any more', () => {
    hook.useWorkspaceModels.mockReturnValue({ data: models, isLoading: false, error: null })
    render(<ModelSelector value="deepseek/deepseek-coder" provider="openrouter" onChange={vi.fn()} />)
    expect(screen.getByTestId('model-unavailable')).toHaveTextContent('deepseek/deepseek-coder')
    expect(screen.getByTestId('model-unavailable')).toHaveTextContent('via openrouter')
  })
})
