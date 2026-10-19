/**
 * PRD-261 A-12g — the SDK's pages/playbooks area wraps the core hooks the Playbooks views read
 * (hooks/use-playbook-api.ts): the same endpoints through the host's backend client, with the same
 * cache and invalidation, handed to a theme as small frozen views (never react-query's result), with
 * a theme's inputs checked before they reach a URL or a body. The area's names are its surface file's
 * and its .d.ts's. The backend client is mocked; react-query and the core hooks are real.
 */
import type { ReactNode } from 'react'
import { readFileSync } from 'fs'
import path from 'path'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const client = vi.hoisted(() => ({
  listWorkflowRecipes: vi.fn(),
  getWorkflowRecipeById: vi.fn(),
  getRecipeExecutions: vi.fn(),
  createWorkflowRecipe: vi.fn(),
  updateWorkflowRecipe: vi.fn(),
  deleteWorkflowRecipe: vi.fn(),
  executeRecipe: vi.fn(),
}))

vi.mock('@/lib/api-client', () => ({ apiClient: client, default: client }))

import * as area from '../pages/playbooks'
import {
  type PlaybookFilters,
  type PlaybookRunFilters,
  type PlaybookRunInput,
  useCreatePlaybook,
  useDeletePlaybook,
  useExecutePlaybook,
  usePlaybookExecutions,
  useUpdatePlaybook,
  useWorkflowPlaybook,
  useWorkflowPlaybooks,
} from '../pages/playbooks'

const SURFACE_FILE = path.resolve(__dirname, '../surface/pages-playbooks.json')
const DECLARATION_FILE = path.resolve(__dirname, '../../../packages/theme-sdk/areas/pages-playbooks.d.ts')
const DECLARED_FUNCTION = /^export declare function ([A-Za-z_$][\w$]*)/gm
const TEMPLATE_ID = 'weekly-notes'
const PLAYBOOK = { id: 7, template_id: TEMPLATE_ID, name: 'Weekly notes', steps: [] }
const LIST = { items: [PLAYBOOK], total: 1, skip: 0, limit: 5 }
const RUNS = { items: [{ id: 3, execution_id: 'run-1', status: 'completed' }], total: 1, skip: 0, limit: 20, recipe_id: TEMPLATE_ID, recipe_quality_score: null }
const STARTED = { recipe_execution_id: 'run-2', recipe_id: TEMPLATE_ID, status: 'started', total_steps: 2, message: 'running', task_id: null, number: null }
const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']
const ACTION_VIEW_KEYS = ['error', 'isRunning', 'reset', 'result', 'run', 'runAsync']
const NOT_A_PLAYBOOK_ID = 'Not a playbook id.'
const SILENT = { log: () => undefined, warn: () => undefined, error: () => undefined }

function wrapper() {
  const queryClient = new QueryClient({ logger: SILENT, defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return function Providers({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

beforeEach(() => {
  for (const method of Object.values(client)) method.mockReset()
})

describe('PRD-261 A-12g — the area exports what A-10c planned', () => {
  it('exports exactly the names of surface/pages-playbooks.json, built, and declares each in its .d.ts', () => {
    const surface = JSON.parse(readFileSync(SURFACE_FILE, 'utf8')) as { status: string; names: string[] }
    const declared = Array.from(readFileSync(DECLARATION_FILE, 'utf8').matchAll(DECLARED_FUNCTION), (match) => match[1])
    expect(surface.status).toBe('built')
    expect(Object.keys(area).sort()).toEqual([...surface.names].sort())
    expect(declared.sort()).toEqual([...surface.names].sort())
  })
})

describe('PRD-261 A-12g — queries', () => {
  it('lists the playbooks through the backend client and hands back a frozen view', async () => {
    client.listWorkflowRecipes.mockResolvedValue(LIST)
    const { result } = renderHook(() => useWorkflowPlaybooks({ limit: 5, sort_by: 'name' }), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(LIST))
    expect(client.listWorkflowRecipes).toHaveBeenCalledWith({ limit: 5, sort_by: 'name' })
    expect(Object.keys(result.current).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)
    expect(result.current).toMatchObject({ isLoading: false, error: null })
  })

  it('keeps only booleans, a trimmed search, whole-number paging in bounds and a known sort', async () => {
    client.listWorkflowRecipes.mockResolvedValue(LIST)
    const filters = { is_featured: 'yes', is_public: false, search: '  notes ', skip: -1, limit: 500, sort_by: 'name;drop' } as unknown as PlaybookFilters
    const { result } = renderHook(() => useWorkflowPlaybooks(filters), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(LIST))
    const sent = client.listWorkflowRecipes.mock.calls[0][0] as Record<string, unknown>
    expect(Object.fromEntries(Object.entries(sent).filter(([, value]) => value !== undefined))).toEqual({ is_public: false, search: 'notes' })
  })

  it('reads one playbook by its template id or by its number', async () => {
    client.getWorkflowRecipeById.mockResolvedValue(PLAYBOOK)
    const byId = renderHook(() => useWorkflowPlaybook(TEMPLATE_ID), { wrapper: wrapper() }).result
    await waitFor(() => expect(byId.current.data).toEqual(PLAYBOOK))
    expect(client.getWorkflowRecipeById).toHaveBeenCalledWith(TEMPLATE_ID)

    const byNumber = renderHook(() => useWorkflowPlaybook(7), { wrapper: wrapper() }).result
    await waitFor(() => expect(byNumber.current.data).toEqual(PLAYBOOK))
    expect(client.getWorkflowRecipeById).toHaveBeenCalledWith('7')
  })

  it('asks for nothing without an id, and is not loading', () => {
    const { result } = renderHook(() => useWorkflowPlaybook(null), { wrapper: wrapper() })

    expect(client.getWorkflowRecipeById).not.toHaveBeenCalled()
    expect(result.current).toMatchObject({ data: null, isLoading: false, error: null })
  })

  it.each([['../workspaces'], ['a/b'], ['.hidden'], [-1], [1.5]])('asks for nothing with the malformed id %s, and says why', (id) => {
    const { result } = renderHook(() => useWorkflowPlaybook(id), { wrapper: wrapper() })

    expect(client.getWorkflowRecipeById).not.toHaveBeenCalled()
    expect(result.current).toMatchObject({ data: null, isLoading: false, error: NOT_A_PLAYBOOK_ID })
  })

  it("hands back a failed read's message, not the error", async () => {
    client.getWorkflowRecipeById.mockRejectedValue(new Error("Playbook 'gone' not found in this workspace"))
    const { result } = renderHook(() => useWorkflowPlaybook('gone'), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.error).toBe("Playbook 'gone' not found in this workspace"))
    expect(result.current.data).toBeNull()
  })

  it("reads a playbook's runs, with a known status and paging in bounds only", async () => {
    client.getRecipeExecutions.mockResolvedValue(RUNS)
    const filters = { status: 'running&admin=1', skip: 2, limit: 0 } as unknown as PlaybookRunFilters
    const { result } = renderHook(() => usePlaybookExecutions(TEMPLATE_ID, filters), { wrapper: wrapper() })

    await waitFor(() => expect(result.current.data).toEqual(RUNS))
    const [id, sent] = client.getRecipeExecutions.mock.calls[0] as [string, Record<string, unknown>]
    expect(id).toBe(TEMPLATE_ID)
    expect(sent).toEqual({ status: undefined, skip: 2, limit: undefined })
  })

  it('asks for no runs with a malformed id', () => {
    const { result } = renderHook(() => usePlaybookExecutions('../x'), { wrapper: wrapper() })

    expect(client.getRecipeExecutions).not.toHaveBeenCalled()
    expect(result.current.error).toBe(NOT_A_PLAYBOOK_ID)
  })
})

describe('PRD-261 A-12g — actions', () => {
  it('runs a playbook through the backend client and hands back the answer', async () => {
    client.executeRecipe.mockResolvedValue(STARTED)
    const { result } = renderHook(() => useExecutePlaybook(), { wrapper: wrapper() })
    expect(Object.keys(result.current).sort()).toEqual(ACTION_VIEW_KEYS)
    expect(Object.isFrozen(result.current)).toBe(true)

    act(() => result.current.run({ id: TEMPLATE_ID, inputs: { week: '42' }, waitForMe: true }))

    await waitFor(() => expect(result.current.result).toEqual(STARTED))
    expect(client.executeRecipe).toHaveBeenCalledWith(TEMPLATE_ID, { week: '42' }, true)
    expect(result.current).toMatchObject({ isRunning: false, error: null })
  })

  it("leaves a run's 'wait for me' to the playbook when it is not a boolean", async () => {
    client.executeRecipe.mockResolvedValue(STARTED)
    const { result } = renderHook(() => useExecutePlaybook(), { wrapper: wrapper() })
    const input = { id: 7, waitForMe: 'yes' } as unknown as PlaybookRunInput

    await act(async () => {
      await result.current.runAsync(input)
    })
    expect(client.executeRecipe).toHaveBeenCalledWith('7', undefined, undefined)
  })

  it('throws a TypeError for a malformed id or inputs that are not an object, and sends nothing', async () => {
    const { result } = renderHook(() => useExecutePlaybook(), { wrapper: wrapper() })
    const listInputs = { id: TEMPLATE_ID, inputs: ['a'] } as unknown as PlaybookRunInput

    expect(() => result.current.run({ id: '../workspaces' })).toThrow(TypeError)
    await expect(result.current.runAsync({ id: 'a/b' })).rejects.toThrow(NOT_A_PLAYBOOK_ID)
    expect(() => result.current.run(listInputs)).toThrow(TypeError)
    expect(client.executeRecipe).not.toHaveBeenCalled()
  })

  it("hands back a failed action's message", async () => {
    client.executeRecipe.mockRejectedValue(new Error('Playbook has no steps'))
    const { result } = renderHook(() => useExecutePlaybook(), { wrapper: wrapper() })

    act(() => result.current.run({ id: TEMPLATE_ID }))

    await waitFor(() => expect(result.current.error).toBe('Playbook has no steps'))
    expect(result.current.result).toBeNull()
  })

  it('reads the list again after a run, through the core cache keys', async () => {
    client.listWorkflowRecipes.mockResolvedValue(LIST)
    client.executeRecipe.mockResolvedValue(STARTED)
    const { result } = renderHook(() => ({ list: useWorkflowPlaybooks(), execute: useExecutePlaybook() }), { wrapper: wrapper() })
    await waitFor(() => expect(result.current.list.data).toEqual(LIST))

    await act(async () => {
      await result.current.execute.runAsync({ id: TEMPLATE_ID })
    })

    await waitFor(() => expect(client.listWorkflowRecipes).toHaveBeenCalledTimes(2))
  })

  it('creates, changes and deletes a playbook, and refuses a body that is not an object', async () => {
    client.createWorkflowRecipe.mockResolvedValue({ message: 'created', recipe: PLAYBOOK })
    client.updateWorkflowRecipe.mockResolvedValue({ message: 'updated', recipe: PLAYBOOK })
    client.deleteWorkflowRecipe.mockResolvedValue({ message: 'deleted', recipe_id: TEMPLATE_ID })
    const { result } = renderHook(
      () => ({ create: useCreatePlaybook(), update: useUpdatePlaybook(), remove: useDeletePlaybook() }),
      { wrapper: wrapper() },
    )

    await act(async () => {
      await result.current.create.runAsync({ name: 'Weekly notes' })
      await result.current.update.runAsync({ id: TEMPLATE_ID, body: { name: 'Notes' } })
      await result.current.remove.runAsync(7)
    })
    expect(client.createWorkflowRecipe).toHaveBeenCalledWith({ name: 'Weekly notes' })
    expect(client.updateWorkflowRecipe).toHaveBeenCalledWith(TEMPLATE_ID, { name: 'Notes' })
    expect(client.deleteWorkflowRecipe).toHaveBeenCalledWith('7')

    const notAnObject = ['name'] as unknown as Record<string, unknown>
    expect(() => result.current.create.run(notAnObject)).toThrow(TypeError)
    expect(() => result.current.update.run({ id: '..', body: {} })).toThrow(TypeError)
    expect(client.createWorkflowRecipe).toHaveBeenCalledTimes(1)
    expect(client.updateWorkflowRecipe).toHaveBeenCalledTimes(1)
  })
})
