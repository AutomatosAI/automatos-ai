/**
 * PRD-261 W-A4P — internal/query-view.ts, the views every page area hands a theme: a query as
 * `{ data, isLoading, isFetching, error, refetch }`, an action as `{ run, runAsync, isRunning,
 * error, result, reset }`, the data and results as deep, frozen copies of what react-query holds,
 * so a theme can never change the objects the core page reads from the shared cache. The
 * documents lane's field picks (internal/documents-view.ts) copy just as deep.
 */
import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { rowsOf } from '../internal/documents-view'
import { frozenCopy, useActionView, useQueryView, type CoreMutation, type CoreQuery } from '../internal/query-view'

const QUERY_VIEW_KEYS = ['data', 'error', 'isFetching', 'isLoading', 'refetch']
const ACTION_VIEW_KEYS = ['error', 'isRunning', 'reset', 'result', 'run', 'runAsync']
const NESTED = { items: [{ id: 1, tags: ['a'], owner: { name: 'Ada' } }], total: 1 }

function cached<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function coreQuery(overrides: Partial<CoreQuery> = {}): CoreQuery {
  return { data: undefined, error: null, isInitialLoading: false, isFetching: false, refetch: vi.fn(), ...overrides }
}

function coreMutation(overrides: Partial<CoreMutation<string>> = {}): CoreMutation<string> {
  return {
    mutate: vi.fn(),
    mutateAsync: vi.fn(async () => cached(NESTED)),
    isLoading: false,
    error: null,
    data: undefined,
    reset: vi.fn(),
    ...overrides,
  }
}

describe('PRD-261 W-A4P — frozenCopy', () => {
  it('copies and freezes every list and plain object, however deep', () => {
    const source = cached(NESTED)
    const copy = frozenCopy(source)
    expect(copy).toEqual(source)
    expect(copy).not.toBe(source)
    expect(copy.items).not.toBe(source.items)
    expect(copy.items[0].owner).not.toBe(source.items[0].owner)
    expect(Object.isFrozen(copy.items[0].tags)).toBe(true)
    expect(Object.isFrozen(copy.items[0].owner)).toBe(true)
    expect(Object.isFrozen(source.items[0])).toBe(false)
  })

  it('hands on values that are not lists or plain objects as they are', () => {
    const when = new Date('2026-10-19T09:00:00Z')
    expect(frozenCopy(when)).toBe(when)
    expect(frozenCopy('text')).toBe('text')
    expect(frozenCopy(null)).toBeNull()
  })
})

describe('PRD-261 W-A4P — useQueryView', () => {
  it("hands on a frozen copy of the data, never the cache's own object, and the same view while nothing changes", () => {
    const data = cached(NESTED)
    const query = coreQuery({ data })
    const { result, rerender } = renderHook(() => useQueryView<typeof NESTED>(query))
    const view = result.current
    expect(Object.keys(view).sort()).toEqual(QUERY_VIEW_KEYS)
    expect(view.data).toEqual(NESTED)
    expect(view.data).not.toBe(data)
    expect(() => (view.data?.items as unknown as unknown[]).splice(0, 1)).toThrow(TypeError)
    expect(data).toEqual(NESTED)
    rerender()
    expect(result.current).toBe(view)
  })

  it('is null until the core query answers, with its error as a message and a refusal before it', () => {
    const failed = renderHook(() => useQueryView(coreQuery({ error: new Error('Boom') }))).result.current
    expect(failed).toMatchObject({ data: null, error: 'Boom' })
    const refused = renderHook(() => useQueryView(coreQuery({ error: new Error('Boom') }), 'Not an id.')).result.current
    expect(refused.error).toBe('Not an id.')
  })

  it("refetches through the core query and returns nothing of react-query's", () => {
    const query = coreQuery({ refetch: vi.fn(async () => ({ data: 'observer result' })) })
    const view = renderHook(() => useQueryView(query)).result.current
    expect(view.refetch()).toBeUndefined()
    expect(query.refetch).toHaveBeenCalledTimes(1)
  })
})

describe('PRD-261 W-A4P — useActionView', () => {
  it('hands on frozen copies of the answer, from result and from runAsync', async () => {
    const data = cached(NESTED)
    const mutation = coreMutation({ data })
    const view = renderHook(() => useActionView<string, string, typeof NESTED>(mutation, (id) => id)).result.current
    expect(Object.keys(view).sort()).toEqual(ACTION_VIEW_KEYS)
    expect(view.result).toEqual(NESTED)
    expect(view.result).not.toBe(data)
    expect(Object.isFrozen(view.result?.items[0])).toBe(true)
    const resolved = await view.runAsync('mission-1')
    expect(Object.isFrozen(resolved.items)).toBe(true)
  })

  it('sends nothing when prepare refuses the variables', async () => {
    const mutation = coreMutation()
    const refuseAll = (): string => {
      throw new TypeError('Automatos theme SDK: Not an id.')
    }
    const view = renderHook(() => useActionView<string, string, unknown>(mutation, refuseAll)).result.current
    expect(() => view.run('../x')).toThrow(TypeError)
    await expect(view.runAsync('../x')).rejects.toThrow(TypeError)
    expect(mutation.mutate).not.toHaveBeenCalled()
    expect(mutation.mutateAsync).not.toHaveBeenCalled()
  })
})

describe('PRD-261 W-A4P — the documents lane copies as deep', () => {
  it("copies a named field's nested objects, never the cache's own", () => {
    const raw = cached([{ id: 3, name: 'Revenue', parameters: [{ name: 'month', type: 'string' }], hidden: 1 }])
    const [row] = rowsOf<{ readonly parameters: readonly { readonly name: string }[] }>(raw, ['id', 'name', 'parameters'])
    expect(row.parameters[0]).toEqual(raw[0].parameters[0])
    expect(row.parameters[0]).not.toBe(raw[0].parameters[0])
    expect(Object.isFrozen(row.parameters[0])).toBe(true)
  })
})
