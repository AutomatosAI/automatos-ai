/**
 * PRD-268 §6.2.5, §6.5.1 (US-W01a) — "From suggestions" on the Board: BoardFilters.from_suggestions keeps
 * the tickets a suggestion's Start filed (source_type 'suggestion'), toggled in Classic's BoardFiltersBar
 * and in Studio's board toolbar; useBoardTasks stays at most 50 code lines.
 */
import { readFileSync } from 'fs'
import path from 'path'
import { cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { BoardTask } from '@/types/board'

const requestMock = vi.hoisted(() => vi.fn())
vi.mock('@/lib/api-client', () => ({ apiClient: { request: (...args: unknown[]) => requestMock(...args) } }))
vi.mock('@/hooks/use-agent-api', () => ({ useAssignableAgents: () => ({ data: [] }) }))
vi.mock('@/components/shared', () => ({ PremiumIcon: () => null }))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { BoardFiltersBar } from '../board-filters'
import { FROM_SUGGESTIONS_LABEL } from '../from-suggestions'
import { BoardToolbar } from '@/components/command-center/board-toolbar'
import { SUGGESTION_SOURCE, keptByFilters, useBoardTasks, type BoardFilters } from '@/hooks/use-board-tasks'

const RAW_TASKS = [
  { id: 1, title: 'Fix the Lisbon flights', status: 'inbox', source_type: 'suggestion', source_id: 's-1' },
  { id: 2, title: 'Price list for Tidewater', status: 'inbox', source_type: 'activity' },
  { id: 3, title: 'Renew the domain', status: 'done', source_type: 'suggestion', source_id: 's-2' },
  { id: 4, title: 'Weekly report', status: 'done' },
]

function task(over: Partial<BoardTask> = {}): BoardTask {
  return { id: '1', type: 'task', name: 'A ticket', status: 'inbox', priority: 'medium', tags: [], review_mode: 'auto', source_id: '1', ...over }
}

function boardHook(filters: BoardFilters) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return renderHook(() => useBoardTasks(filters), { wrapper })
}

const columnNames = (columns: { status: string; tasks: BoardTask[] }[], status: string) =>
  columns.find((column) => column.status === status)?.tasks.map((t) => t.name)

beforeEach(() => {
  requestMock.mockReset()
  requestMock.mockResolvedValue({ tasks: RAW_TASKS, total: RAW_TASKS.length })
})
afterEach(cleanup)

describe('BoardFilters.from_suggestions', () => {
  it("keeps only the tickets a suggestion's Start filed, and every ticket when it is off", () => {
    const filed = task({ source_type: SUGGESTION_SOURCE })
    const other = task({ source_type: 'activity' })
    expect(SUGGESTION_SOURCE).toBe('suggestion')
    expect([keptByFilters(filed, { from_suggestions: true }), keptByFilters(other, { from_suggestions: true })]).toEqual([true, false])
    expect([keptByFilters(filed, {}), keptByFilters(other, { from_suggestions: false }), keptByFilters(other)]).toEqual([true, true, true])
  })

  it('still applies the type filter beside it', () => {
    const mission = task({ type: 'mission', source_type: SUGGESTION_SOURCE })
    expect(keptByFilters(mission, { type: 'task', from_suggestions: true })).toBe(false)
    expect(keptByFilters(mission, { type: 'mission', from_suggestions: true })).toBe(true)
  })

  it('useBoardTasks keeps the started suggestions in their columns, and the whole board without the filter', async () => {
    const filtered = boardHook({ from_suggestions: true })
    await waitFor(() => expect(columnNames(filtered.result.current.columns, 'inbox')).toEqual(['Fix the Lisbon flights']))
    expect(columnNames(filtered.result.current.columns, 'done')).toEqual(['Renew the domain'])

    const whole = boardHook({})
    await waitFor(() => expect(columnNames(whole.result.current.columns, 'inbox')).toEqual(['Fix the Lisbon flights', 'Price list for Tidewater']))
    expect(columnNames(whole.result.current.columns, 'done')).toEqual(['Renew the domain', 'Weekly report'])
  })

  it('useBoardTasks stays at most 50 code lines', () => {
    const lines = readFileSync(path.resolve(__dirname, '..', '..', '..', '..', 'hooks', 'use-board-tasks.ts'), 'utf8').split('\n')
    const start = lines.findIndex((line) => line.startsWith('export function useBoardTasks('))
    const end = lines.findIndex((line, at) => at > start && line === '}')
    const code = lines.slice(start, end + 1).filter((line) => line.trim() !== '' && !line.trim().startsWith('//'))
    expect(start).toBeGreaterThan(-1)
    expect(code.length).toBeLessThanOrEqual(50)
  })
})

describe('the "From suggestions" toggle', () => {
  it("Classic's BoardFiltersBar turns it on and off", () => {
    const onFiltersChange = vi.fn()
    const bar = (filters: BoardFilters) => (
      <BoardFiltersBar filters={filters} onFiltersChange={onFiltersChange} selectedAgentId={null} onSelectAgent={vi.fn()} agentCount={0} taskCount={0} />
    )
    const { rerender } = render(bar({ priority: 'high' }))
    const toggle = screen.getByRole('button', { name: FROM_SUGGESTIONS_LABEL })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(toggle)
    expect(onFiltersChange).toHaveBeenLastCalledWith({ priority: 'high', from_suggestions: true })

    rerender(bar({ priority: 'high', from_suggestions: true }))
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(toggle)
    expect(onFiltersChange).toHaveBeenLastCalledWith({ priority: 'high', from_suggestions: false })
  })

  it("Studio's board toolbar turns it on and off", () => {
    const onFromSuggestions = vi.fn()
    const toolbar = (on: boolean) => (
      <BoardToolbar
        mode="column" onMode={vi.fn()} density="comfortable" onDensity={vi.fn()} priority={null} onPriority={vi.fn()}
        agentId={null} onAgentId={vi.fn()} agents={[]} search="" onSearch={vi.fn()}
        fromSuggestions={on} onFromSuggestions={onFromSuggestions}
      />
    )
    const { rerender } = render(toolbar(false))
    const toggle = screen.getByRole('button', { name: FROM_SUGGESTIONS_LABEL })
    expect(toggle).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(toggle)
    expect(onFromSuggestions).toHaveBeenLastCalledWith(true)

    rerender(toolbar(true))
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    expect(toggle).toHaveClass('on')
    fireEvent.click(toggle)
    expect(onFromSuggestions).toHaveBeenLastCalledWith(false)
  })

  it("Studio's board hands the toggle to useBoardTasks", () => {
    const source = readFileSync(path.resolve(__dirname, '..', '..', '..', 'command-center', 'board-tab.tsx'), 'utf8')
    expect(source).toContain('from_suggestions: fromSuggestions,')
    expect(source).toContain('onFromSuggestions={setFromSuggestions}')
  })
})
