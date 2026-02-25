'use client'

/**
 * MemorySearch Component for PRD-38.2 Extended Widgets
 *
 * Search interface for memories with debounced filtering (300ms)
 * and collapsible "Add Memory" form (type dropdown + content textarea).
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Search, X, Plus, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { MemoryType } from '../types'

interface MemorySearchProps {
  onSearch: (query: string) => void
  onAdd?: (memory: { type: MemoryType; content: string }) => void
  isSearching?: boolean
  className?: string
}

/**
 * Custom hook for debounced value updates.
 * Returns the debounced value after the specified delay.
 */
function useDebounce(value: string, delay: number): string {
  const [debouncedValue, setDebouncedValue] = useState(value)

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value)
    }, delay)

    return () => {
      clearTimeout(timer)
    }
  }, [value, delay])

  return debouncedValue
}

const MEMORY_TYPES: { value: MemoryType; label: string }[] = [
  { value: 'fact', label: 'Fact' },
  { value: 'preference', label: 'Preference' },
  { value: 'context', label: 'Context' },
  { value: 'instruction', label: 'Instruction' },
]

export function MemorySearch({
  onSearch,
  onAdd,
  isSearching,
  className,
}: MemorySearchProps) {
  const [query, setQuery] = useState('')
  const [showAddForm, setShowAddForm] = useState(false)
  const [newType, setNewType] = useState<MemoryType>('fact')
  const [newContent, setNewContent] = useState('')

  // Debounce search query by 300ms
  const debouncedQuery = useDebounce(query, 300)

  // Track previous debounced value to avoid redundant calls
  const prevDebouncedRef = useRef(debouncedQuery)

  useEffect(() => {
    if (prevDebouncedRef.current !== debouncedQuery) {
      prevDebouncedRef.current = debouncedQuery
      onSearch(debouncedQuery.trim())
    }
  }, [debouncedQuery, onSearch])

  const handleClear = useCallback(() => {
    setQuery('')
    // Immediately fire search with empty string (bypass debounce)
    prevDebouncedRef.current = ''
    onSearch('')
  }, [onSearch])

  const handleAddSubmit = useCallback(() => {
    if (!newContent.trim() || !onAdd) return
    onAdd({ type: newType, content: newContent.trim() })
    setNewContent('')
    setNewType('fact')
    setShowAddForm(false)
  }, [newType, newContent, onAdd])

  return (
    <div className={cn('space-y-2', className)}>
      {/* Search input */}
      <div className="relative flex items-center">
        <Search className="absolute left-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search memories..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-9 pr-8 h-8 text-sm"
          disabled={isSearching}
        />
        {query && (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-1 h-6 w-6"
            onClick={handleClear}
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Add Memory toggle */}
      {onAdd && (
        <div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="w-full justify-between text-xs text-muted-foreground h-7"
            onClick={() => setShowAddForm((prev) => !prev)}
          >
            <span className="flex items-center gap-1">
              <Plus className="h-3.5 w-3.5" />
              Add Memory
            </span>
            {showAddForm ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </Button>

          {/* Collapsible add form */}
          {showAddForm && (
            <div className="mt-1.5 space-y-2 rounded-lg border border-border/40 bg-muted/20 p-2.5">
              <Select
                value={newType}
                onValueChange={(v) => setNewType(v as MemoryType)}
              >
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  {MEMORY_TYPES.map((t) => (
                    <SelectItem key={t.value} value={t.value}>
                      {t.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Textarea
                placeholder="What should I remember?"
                value={newContent}
                onChange={(e) => setNewContent(e.target.value)}
                rows={3}
                className="text-sm min-h-[60px] resize-none"
              />

              <div className="flex justify-end gap-1.5">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => {
                    setShowAddForm(false)
                    setNewContent('')
                    setNewType('fact')
                  }}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={!newContent.trim()}
                  onClick={handleAddSubmit}
                >
                  <Plus className="h-3.5 w-3.5 mr-1" />
                  Add
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
