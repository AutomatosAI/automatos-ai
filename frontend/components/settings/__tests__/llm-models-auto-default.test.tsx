/**
 * 19 Oct 2026 — the platform default for Auto is an admin's card, and the
 * Orchestrator tab's Reset returns a workspace's Auto to it.
 *
 * System Settings > LLM Tiers shows a "Platform default for Auto" card with only
 * orchestrator_llm provider + model, saved through the same bulk-update path as the
 * other tiers (which moves the Auto agents on the platform default). Its Reset saves
 * the shipped default through that path too. Settings > Orchestrator's Reset saves
 * the platform default read from GET /api/models/platform-default: no model id is
 * kept in the frontend.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, render, renderHook, screen } from '@testing-library/react'

import type { SystemSetting } from '@/lib/api/system-settings'

const cards = vi.hoisted(() => [] as any[])
const api = vi.hoisted(() => ({ request: vi.fn(), get: vi.fn() }))

vi.mock('../LLMTierCard', () => ({
  default: (props: any) => {
    cards.push(props)
    return <div>{props.title}</div>
  },
}))
vi.mock('@/lib/api-client', () => ({ apiClient: api }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import LLMModelsSettingsTab from '../LLMModelsSettingsTab'
import { useLLMSave } from '../use-orchestrator-settings'
import { fallbackOrchestratorConfig } from '../orchestrator-settings-config'
import { savableModelConfig } from '@/lib/llm-defaults'

function row(key: string, value: string, default_value: string): SystemSetting {
  return {
    id: key.length, category: 'orchestrator_llm', key, value, value_type: 'string', description: null,
    is_sensitive: false, is_required: true, default_value, validation_rules: null,
    created_at: '2026-10-19T00:00:00Z', updated_at: '2026-10-19T00:00:00Z', created_by: 'system',
  }
}

const orchestratorSettings = [
  row('provider', 'openrouter', 'anthropic'),
  row('model', 'openai/gpt-5.5', 'claude-haiku-5-5'),
  row('temperature', '0.2', '0.7'),
  row('llm_model', 'x', 'y'),
]

function renderTab(onSaveOrchestrator = vi.fn()) {
  render(
    <LLMModelsSettingsTab
      orchestratorSettings={orchestratorSettings}
      systemSettings={[]}
      embeddingsSettings={[]}
      onSaveOrchestrator={onSaveOrchestrator}
      onSaveSystem={vi.fn()}
      onSaveEmbeddings={vi.fn()}
      onResetSystem={vi.fn()}
      onResetEmbeddings={vi.fn()}
      saving={false}
    />,
  )
  return onSaveOrchestrator
}

beforeEach(() => {
  cards.length = 0
  api.request.mockReset()
  api.get.mockReset()
})

describe('Platform default for Auto card', () => {
  it('shows only the orchestrator_llm provider and model, saved through the tab save', () => {
    const onSave = renderTab()
    const auto = cards.find((card) => card.category === 'orchestrator_llm')

    expect(auto.title).toBe('Platform default for Auto')
    expect(auto.settings.map((s: SystemSetting) => s.key)).toEqual(['provider', 'model'])
    expect(auto.onSave).toBe(onSave)
    expect(screen.queryByText(/is configured in the/)).not.toBeInTheDocument()
  })

  it('resets by saving the shipped default, so the Auto agents move too', () => {
    const onSave = renderTab()
    cards.find((card) => card.category === 'orchestrator_llm').onReset()

    expect(onSave).toHaveBeenCalledWith({ provider: 'anthropic', model: 'claude-haiku-5-5' })
  })
})

describe('Settings > Orchestrator Reset', () => {
  it('saves the platform default the API returns', async () => {
    api.get.mockResolvedValue({ provider: 'anthropic', model_id: 'claude-sonnet-5-5' })
    api.request.mockResolvedValue(fallbackOrchestratorConfig())
    const { result } = renderHook(() => useLLMSave(fallbackOrchestratorConfig(), vi.fn()))

    await act(() => result.current.handleReset())

    expect(api.get).toHaveBeenCalledWith('/api/models/platform-default')
    const [route, init] = api.request.mock.calls[0]
    expect(route).toBe('/api/workspaces/current/orchestrator')
    const { llm } = JSON.parse(init.body)
    expect([llm.provider, llm.model_id, llm.max_tokens]).toEqual(['anthropic', 'claude-sonnet-5-5', 4000])
  })
})

describe('Settings > Orchestrator Save with no model', () => {
  it('never saves the blank model of the fallback config over Auto\'s', async () => {
    const { result } = renderHook(() => useLLMSave(fallbackOrchestratorConfig(), vi.fn()))

    await act(() => result.current.handleSaveLLM())

    expect(api.request).not.toHaveBeenCalled()
  })
})

describe('savableModelConfig', () => {
  it('fills a blank model from the platform default, and is null while none is known', () => {
    const blank = { provider: '', model_id: '', temperature: 0.2 }

    expect(savableModelConfig(blank, undefined)).toBeNull()
    expect(savableModelConfig(blank, { provider: 'anthropic', model_id: 'claude-haiku-5-5' }))
      .toEqual({ provider: 'anthropic', model_id: 'claude-haiku-5-5', temperature: 0.2 })
    expect(savableModelConfig({ provider: 'openrouter', model_id: 'x/y' }, undefined))
      .toEqual({ provider: 'openrouter', model_id: 'x/y' })
    expect(blank.model_id).toBe('')
  })
})
