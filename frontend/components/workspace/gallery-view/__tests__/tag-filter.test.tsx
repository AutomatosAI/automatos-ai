/**
 * 7 Oct — the Deliverables page filters by tag: the filter bar's tag box (debounced,
 * the tag read as the platform stores it), Clear empties it, and the list asks
 * GET /api/deliverables with ?tag=.
 */
import { useState } from 'react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react'

vi.mock('@/components/ui/select', () => ({
  Select: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectTrigger: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SelectContent: () => null,
  SelectItem: () => null,
  SelectValue: () => null,
}))

import { DEFAULT_FILTERS, buildListQuery, type FilterState } from '@/hooks/use-deliverables-api'
import { FilterBar } from '../filter-bar'
import { TAG_FILTER_DEBOUNCE_MS, TagFilter, cleanTag } from '../tag-filter'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

/** The tag box as the filter bar holds it: the chosen tag is the bar's state. */
function Harness({ onChange }: { onChange: (tag: string | null) => void }) {
  const [tag, setTag] = useState<string | null>(null)
  return (
    <TagFilter
      value={tag}
      onChange={(next) => {
        setTag(next)
        onChange(next)
      }}
    />
  )
}

describe('the tag box', () => {
  it('reads a tag as the platform stores it', () => {
    expect(cleanTag('  Q3   Report ')).toBe('q3 report')
    expect(cleanTag('   ')).toBeNull()
  })

  it('filters by the typed tag once typing stops, and an empty box clears it', () => {
    vi.useFakeTimers()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)
    const box = screen.getByLabelText('Filter by tag')

    fireEvent.change(box, { target: { value: '  Harbourline ' } })
    expect(onChange).not.toHaveBeenCalled()
    act(() => {
      vi.advanceTimersByTime(TAG_FILTER_DEBOUNCE_MS)
    })
    expect(onChange).toHaveBeenLastCalledWith('harbourline')

    fireEvent.change(box, { target: { value: '' } })
    act(() => {
      vi.advanceTimersByTime(TAG_FILTER_DEBOUNCE_MS)
    })
    expect(onChange).toHaveBeenLastCalledWith(null)
  })
})

describe('the filter bar', () => {
  function renderBar(filters: FilterState) {
    const onFiltersChange = vi.fn()
    render(
      <FilterBar filters={filters} onFiltersChange={onFiltersChange} total={3} viewMode="grid" onViewModeChange={() => {}} />,
    )
    return onFiltersChange
  }

  it('shows the active tag, and Clear removes it', () => {
    const onFiltersChange = renderBar({ ...DEFAULT_FILTERS, tag: 'invoice' })

    expect((screen.getByLabelText('Filter by tag') as HTMLInputElement).value).toBe('invoice')
    fireEvent.click(screen.getByRole('button', { name: /clear/i }))
    expect(onFiltersChange).toHaveBeenCalledWith(expect.objectContaining({ tag: null }))
  })

  it('offers no Clear when no filter is set', () => {
    renderBar({ ...DEFAULT_FILTERS })
    expect(screen.queryByRole('button', { name: /clear/i })).toBeNull()
  })
})

describe('the list request', () => {
  it('asks for the tag', () => {
    const query = new URLSearchParams(buildListQuery({ ...DEFAULT_FILTERS, tag: ' Invoice ' }, 0))
    expect(query.get('tag')).toBe('invoice')
    expect(new URLSearchParams(buildListQuery(DEFAULT_FILTERS, 0)).has('tag')).toBe(false)
  })
})
