/**
 * 19 Oct 2026 — the platform default for Auto is an admin's card, and the
 * Orchestrator tab's Reset returns a workspace's Auto to it.
 *
 * System Settings > System & Embeddings shows a "Platform default for Auto" card with
 * only orchestrator_llm provider + model. The card reads its own rows from the
 * by-category endpoint and saves through bulk-update, the path that moves the Auto
 * agents on the platform default; its Reset saves the shipped defaults through that
 * path too. Settings > Orchestrator's Reset saves the platform default read from
 * GET /api/models/platform-default: no model id is kept in the frontend.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { act, render, renderHook, screen, waitFor } from '@testing-library/react'
import { toast } from 'sonner'

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

const BY_CATEGORY = '/api/system-settings/by-category'
const BULK_UPDATE = '/api/system-settings/bulk-update'

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

const settingsByCategory = [
  { category: 'orchestrator_llm', settings: orchestratorSettings, total_count: 4 },
  { category: 'system_llm', settings: [], total_count: 0 },
]

/** The admin settings API: the rows by category, and a bulk-update that moved two Autos. */
function serveSettingsApi() {
  api.request.mockImplementation(async (route: string) => {
    if (route === BY_CATEGORY) return settingsByCategory
    if (route === BULK_UPDATE) return { message: 'Updated 2 settings successfully', auto_models: { moved: 2, by_route: {} } }
    throw new Error(`unexpected request ${route}`)
  })
}

/** The latest render's props for one tier card. */
const card = (category: string) => [...cards].reverse().find((c) => c.category === category)

const bulkUpdateCalls = () => api.request.mock.calls.filter(([route]) => route === BULK_UPDATE)

async function renderTab() {
  render(
    <LLMModelsSettingsTab
      systemSettings={[]}
      embeddingsSettings={[]}
      onSaveSystem={vi.fn()}
      onSaveEmbeddings={vi.fn()}
      onResetSystem={vi.fn()}
      onResetEmbeddings={vi.fn()}
      saving={false}
    />,
  )
  await waitFor(() => expect(card('orchestrator_llm').settings).toHaveLength(2))
}

beforeEach(() => {
  cards.length = 0
  api.request.mockReset()
  api.get.mockReset()
  vi.mocked(toast.success).mockReset()
  serveSettingsApi()
})

describe('Platform default for Auto card', () => {
  it('shows only the orchestrator_llm provider and model, read from the by-category endpoint', async () => {
    await renderTab()
    const auto = card('orchestrator_llm')

    expect(api.request).toHaveBeenCalledWith(BY_CATEGORY)
    expect(auto.title).toBe('Platform default for Auto')
    expect(auto.settings.map((s: SystemSetting) => s.key)).toEqual(['provider', 'model'])
    expect(screen.queryByText(/is configured in the/)).not.toBeInTheDocument()
  })

  it('saves the pair through bulk-update and reports the Auto agents moved', async () => {
    await renderTab()

    await act(() => card('orchestrator_llm').onSave({ provider: 'anthropic', model: 'claude-sonnet-5-5' }))

    const [[, init]] = bulkUpdateCalls()
    expect(init.method).toBe('POST')
    expect(init.body).toEqual([{ id: 8, value: 'anthropic' }, { id: 5, value: 'claude-sonnet-5-5' }])
    expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('2 Auto agents moved'))
  })

  it('resets by saving the shipped defaults through bulk-update, so the Auto agents move too', async () => {
    await renderTab()

    await act(() => card('orchestrator_llm').onReset())

    const [[, init]] = bulkUpdateCalls()
    expect(init.body).toEqual([{ id: 8, value: 'anthropic' }, { id: 5, value: 'claude-haiku-5-5' }])
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
  it('never saves a blank model over Auto\'s', async () => {
    const config = fallbackOrchestratorConfig()
    const blank = { ...config, llm: { ...config.llm, provider: '', model_id: '' } }
    const { result } = renderHook(() => useLLMSave(blank, vi.fn()))

    await act(() => result.current.handleSaveLLM())

    expect(api.request).not.toHaveBeenCalled()
  })
})
