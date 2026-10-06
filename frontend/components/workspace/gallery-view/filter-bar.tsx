/**
 * FilterBar — PRD-129 Workspace Outputs Hub
 * ==========================================
 *
 * Filter controls for the Gallery view:
 *   - Debounced search (300ms)
 *   - Artifact type dropdown
 *   - Source type dropdown
 *   - Date range dropdown
 *   - Tag box (7 Oct: only Deliverables carrying the tag; tag-filter.tsx)
 *   - Clear filters button (appears when any filter active)
 *   - Total count on the right
 */

'use client'

import { useEffect, useState } from 'react'
import { Calendar, FileType, Search, X, Zap } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ViewToggle, type ViewMode } from '@/components/shared/view-toggle'
import {
  DELIVERABLE_ACCENTS,
  DeliverableIcon,
  isDeliverableType,
  type DeliverableType,
} from '@/components/icons/deliverable-icon'
import {
  DEFAULT_FILTERS,
  type DateRange,
  type FilterState,
} from '@/hooks/use-deliverables-api'

import { TagFilter } from './tag-filter'
import { useDebouncedValue } from './use-debounced-value'

// ---- Option definitions ----
const TYPE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'all', label: 'All types' },
  { value: 'report', label: 'Reports' },
  { value: 'image', label: 'Images' },
  { value: 'document', label: 'Documents' },
  { value: 'code', label: 'Code' },
  { value: 'slide', label: 'Slides' },
  { value: 'spreadsheet', label: 'Spreadsheets' },
  { value: 'blog_post', label: 'Blog Posts' },
  { value: 'video', label: 'Videos' },
]

const SOURCE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'all', label: 'All sources' },
  { value: 'chat', label: 'Chat' },
  { value: 'task', label: 'Tasks' },
  { value: 'mission', label: 'Missions' },
  { value: 'heartbeat', label: 'Heartbeats' },
  { value: 'playbook', label: 'Playbooks' },
  { value: 'upload', label: 'Uploads' },
]

const DATE_OPTIONS: Array<{ value: DateRange; label: string }> = [
  { value: 'all', label: 'All time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'This week' },
  { value: 'month', label: 'This month' },
]

// ---- Props ----
export interface FilterBarProps {
  filters: FilterState
  onFiltersChange: (filters: FilterState) => void
  total: number
  viewMode: ViewMode
  onViewModeChange: (mode: ViewMode) => void
}

function hasActiveFilters(filters: FilterState): boolean {
  return (
    filters.artifact_type !== null ||
    filters.source_type !== null ||
    filters.agent_id !== null ||
    filters.date_range !== 'all' ||
    filters.search.trim() !== '' ||
    Boolean(filters.tag)
  )
}

function TypeSelect({ value, onChange }: { value: string | null; onChange: (value: string) => void }) {
  const typedValue = value && isDeliverableType(value) ? (value as DeliverableType) : null
  return (
    <Select value={value ?? 'all'} onValueChange={onChange}>
      <SelectTrigger className="w-[170px]" aria-label="Filter by type">
        <div className="flex items-center gap-2">
          {typedValue ? (
            <span className={DELIVERABLE_ACCENTS[typedValue].tw}>
              <DeliverableIcon type={typedValue} size="badge" />
            </span>
          ) : (
            <FileType className="h-4 w-4 text-muted-foreground" />
          )}
          <SelectValue />
        </div>
      </SelectTrigger>
      <SelectContent>
        {TYPE_OPTIONS.map((opt) => {
          const typed = isDeliverableType(opt.value) ? (opt.value as DeliverableType) : null
          return (
            <SelectItem key={opt.value} value={opt.value}>
              <div className="flex items-center gap-2">
                {typed ? (
                  <span className={DELIVERABLE_ACCENTS[typed].tw}>
                    <DeliverableIcon type={typed} size="badge" />
                  </span>
                ) : (
                  <FileType className="h-4 w-4 text-muted-foreground" />
                )}
                <span>{opt.label}</span>
              </div>
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}

export function FilterBar({ filters, onFiltersChange, total, viewMode, onViewModeChange }: FilterBarProps) {
  // Local search state so typing feels instant; debounce pushes to parent.
  const [searchInput, setSearchInput] = useState(filters.search)
  const debouncedSearch = useDebouncedValue(searchInput, 300)

  // Keep local state in sync if parent resets filters externally (e.g., Clear).
  useEffect(() => {
    setSearchInput(filters.search)
  }, [filters.search])

  // Push debounced search up — skip no-op updates to avoid extra renders.
  useEffect(() => {
    if (debouncedSearch !== filters.search) {
      onFiltersChange({ ...filters, search: debouncedSearch })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch, filters.search, onFiltersChange])

  const active = hasActiveFilters(filters)

  const handleTypeChange = (value: string) => {
    onFiltersChange({
      ...filters,
      artifact_type: value === 'all' ? null : value,
    })
  }

  const handleSourceChange = (value: string) => {
    onFiltersChange({
      ...filters,
      source_type: value === 'all' ? null : value,
    })
  }

  const handleDateChange = (value: string) => {
    onFiltersChange({
      ...filters,
      date_range: value as DateRange,
    })
  }

  const handleClear = () => {
    setSearchInput('')
    onFiltersChange({ ...DEFAULT_FILTERS })
  }

  return (
    <div className="flex flex-wrap items-center gap-3 pb-4">
      {/* Search */}
      <div className="relative flex-1 min-w-[220px] max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Search outputs..."
          className="pl-9"
          aria-label="Search outputs"
        />
      </div>

      {/* Type */}
      <TypeSelect value={filters.artifact_type} onChange={handleTypeChange} />

      {/* Source */}
      <Select
        value={filters.source_type ?? 'all'}
        onValueChange={handleSourceChange}
      >
        <SelectTrigger className="w-[160px]" aria-label="Filter by source">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-muted-foreground" />
            <SelectValue />
          </div>
        </SelectTrigger>
        <SelectContent>
          {SOURCE_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Date range */}
      <Select value={filters.date_range} onValueChange={handleDateChange}>
        <SelectTrigger className="w-[150px]" aria-label="Filter by date range">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <SelectValue />
          </div>
        </SelectTrigger>
        <SelectContent>
          {DATE_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Tag */}
      <TagFilter value={filters.tag} onChange={(tag) => onFiltersChange({ ...filters, tag })} />

      {/* Clear */}
      {active && (
        <Button
          variant="ghost"
          size="sm"
          onClick={handleClear}
          className="gap-1 text-muted-foreground hover:text-foreground"
        >
          <X className="h-4 w-4" />
          Clear
        </Button>
      )}

      {/* Total count + view toggle (right-aligned) */}
      <div className="ml-auto flex items-center gap-3">
        <span className="text-sm text-muted-foreground tabular-nums">
          {total.toLocaleString()} {total === 1 ? 'output' : 'outputs'}
        </span>
        <ViewToggle value={viewMode} onChange={onViewModeChange} />
      </div>
    </div>
  )
}
