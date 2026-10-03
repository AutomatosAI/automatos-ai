/**
 * F242 (night 7b): an edit in the playbook editor keeps the settings the editor
 * does not show.
 *
 * Playbook 113 "Weekly Instagram posts" was set to wait for the owner (Auto set
 * execution_config.wait_for_me), and its daily 20:25 timer was switched off with
 * enabled: false, on UK time (Europe/London). The editor's save rebuilt both
 * configs from its own fields, so editing the name dropped "wait for me",
 * switched the timer back on and reset its zone (and emptied the tags). These
 * tests run the real hook and the real apiClient against a fake backend, and
 * check the save that reached it.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

import type { PlaybookFormValues } from '@/components/workflows/create-playbook-modal'

vi.mock('sonner', () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }) }))

import { usePlaybookForm } from '@/hooks/use-playbook-form'

const ADDRESS = 'custom-14896903'
const CAPTIONS = 'Write the two Instagram captions for this week.'
const STORED = {
  id: 113,
  template_id: ADDRESS,
  name: 'Weekly Instagram posts',
  description: 'Two captions a week.',
  tags: ['socials'],
  steps: [{ step_id: 's1', order: 1, agent_id: 324, prompt_template: CAPTIONS, error_handling: 'stop' }],
  execution_config: {
    mode: 'sequential', max_retries: 3, per_step_timeout: 120, total_timeout: 600, auto_learning: true,
    memory_isolation: 'shared', wait_for_me: true,
  },
  schedule_config: { type: 'cron', cron_expression: '25 20 * * *', timezone: 'Europe/London', enabled: false },
}

const server = vi.hoisted(() => ({ saves: [] as Array<Record<string, any>> }))

function answer(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

/** The backend's playbook 113: read it, save it. */
async function backend(url: string, init?: RequestInit): Promise<Response> {
  const path = new URL(url, 'http://app.test').pathname
  if (path !== `/api/workflow-recipes/${ADDRESS}`) throw new Error(`unexpected ${init?.method ?? 'GET'} ${path}`)
  if (init?.method === 'PUT') {
    const body = JSON.parse(String(init.body))
    server.saves.push(body)
    return answer({ message: 'Recipe updated successfully.', recipe: { ...STORED, ...body } })
  }
  return answer(STORED)
}

/** Playbook 113 in the editor, the way the Playbooks tab's Edit fills it in. */
function inTheEditor(changes: Partial<PlaybookFormValues> = {}): PlaybookFormValues {
  return {
    name: STORED.name,
    description: STORED.description,
    inputs: '{}',
    outputs: '{}',
    steps: [{ step_id: 's1', order: 1, agent_id: '324', prompt_template: CAPTIONS, error_handling: 'stop', pre_exec: '' }],
    execution_config: {
      mode: 'sequential', max_retries: 3, timeout_per_step: 120000, total_timeout: 600000, auto_learning: true,
      parallel_limit: 5, memory_isolation: 'shared',
    },
    schedule_config: { type: 'cron', cron_expression: '25 20 * * *', trigger_config: {} },
    ...changes,
  }
}

async function saved(values: PlaybookFormValues) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client }, children)
  const { result } = renderHook(() => usePlaybookForm(), { wrapper })
  const done = vi.fn()
  await act(async () => {
    await result.current.updatePlaybook(ADDRESS, values, done)
  })
  expect(done).toHaveBeenCalledTimes(1)
  expect(server.saves).toHaveLength(1)
  return server.saves[0]
}

beforeEach(() => {
  server.saves = []
  vi.stubGlobal('fetch', vi.fn(backend))
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('an edit keeps what the editor does not show', () => {
  it('a new name keeps "wait for me" and the switched-off timer on UK time', async () => {
    const body = await saved(inTheEditor({ name: 'Weekly Instagram posts (Huila)' }))

    expect(body.name).toBe('Weekly Instagram posts (Huila)')
    expect(body.execution_config).toMatchObject({ wait_for_me: true, max_retries: 3, per_step_timeout: 120 })
    expect(body.schedule_config).toEqual(
      { type: 'cron', cron_expression: '25 20 * * *', timezone: 'Europe/London', enabled: false })
    expect(body).not.toHaveProperty('tags')            // the edit emptied them
    expect(body).not.toHaveProperty('is_public')
    expect(body).not.toHaveProperty('template_id')
  })

  it('what the editor does change still wins', async () => {
    const body = await saved(inTheEditor({
      execution_config: { ...inTheEditor().execution_config, max_retries: 1 },
      schedule_config: { type: 'cron', cron_expression: '0 9 * * 1', trigger_config: {} },
    }))

    expect(body.execution_config).toMatchObject({ max_retries: 1, wait_for_me: true })
    expect(body.schedule_config).toMatchObject({ cron_expression: '0 9 * * 1', timezone: 'Europe/London', enabled: false })
  })
})
