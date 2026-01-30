'use client'

/**
 * MemorySearch Component for PRD-38.2 Extended Widgets
 *
 * Search interface for memories
 */

import { useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Search, X } from 'lucide-react'

interface MemorySearchProps {
  onSearch: (query: string) => void
  isSearching?: boolean
  className?: string
}

export function MemorySearch({
  onSearch,
  isSearching,
  className,
}: MemorySearchProps) {
  const [query, setQuery] = useState('')

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault()
      if (query.trim()) {
        onSearch(query.trim())
      }
    },
    [query, onSearch]
  )

  const handleClear = useCallback(() => {
    setQuery('')
    onSearch('')
  }, [onSearch])

  return (
    <form
      onSubmit={handleSubmit}
      className={className}
    >
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
    </form>
  )
}
