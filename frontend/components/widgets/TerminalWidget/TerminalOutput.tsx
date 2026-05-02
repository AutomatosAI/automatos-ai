'use client'

/**
 * TerminalOutput Component for PRD-38.2 Extended Widgets
 *
 * ANSI-aware terminal output renderer using ansi-to-react
 */

import { useRef, useEffect, useState, useMemo } from 'react'
import Ansi from 'ansi-to-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Search, ChevronUp, ChevronDown, X } from 'lucide-react'
import { cn } from '@/lib/utils'

interface TerminalOutputProps {
  output: string
  isStreaming?: boolean
  className?: string
}

export function TerminalOutput({
  output,
  isStreaming,
  className,
}: TerminalOutputProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showSearch, setShowSearch] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchIndex, setSearchIndex] = useState(0)

  // Auto-scroll to bottom when output changes (if streaming)
  useEffect(() => {
    if (isStreaming && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [output, isStreaming])

  // Find search matches
  const searchMatches = useMemo(() => {
    if (!searchQuery) return []
    const matches: number[] = []
    const lines = output.split('\n')
    lines.forEach((line, index) => {
      if (line.toLowerCase().includes(searchQuery.toLowerCase())) {
        matches.push(index)
      }
    })
    return matches
  }, [output, searchQuery])

  // Navigate search results
  const navigateSearch = (direction: 'next' | 'prev') => {
    if (searchMatches.length === 0) return

    let newIndex = searchIndex
    if (direction === 'next') {
      newIndex = (searchIndex + 1) % searchMatches.length
    } else {
      newIndex = (searchIndex - 1 + searchMatches.length) % searchMatches.length
    }
    setSearchIndex(newIndex)

    // Scroll to match (simplified - would need ref to specific line)
    if (scrollRef.current) {
      const lineHeight = 20 // Approximate line height
      scrollRef.current.scrollTop = searchMatches[newIndex] * lineHeight
    }
  }

  // Toggle search with keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'f') {
        e.preventDefault()
        setShowSearch(true)
      }
      if (e.key === 'Escape') {
        setShowSearch(false)
        setSearchQuery('')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Handle empty output
  if (!output) {
    return (
      <div
        className={cn(
          'flex items-center justify-center h-full',
          'text-muted-foreground text-sm font-mono',
          'bg-background',
          className
        )}
      >
        {isStreaming ? 'Waiting for output...' : 'No output'}
      </div>
    )
  }

  return (
    <div className={cn('flex flex-col h-full bg-background', className)}>
      {/* Search bar */}
      {showSearch && (
        <div className="flex items-center gap-2 px-3 py-2 bg-secondary border-b border-border">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search output..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value)
              setSearchIndex(0)
            }}
            className="h-7 bg-background border-border text-sm font-mono"
            autoFocus
          />
          {searchQuery && searchMatches.length > 0 && (
            <>
              <span className="text-xs text-muted-foreground">
                {searchIndex + 1}/{searchMatches.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => navigateSearch('prev')}
              >
                <ChevronUp className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => navigateSearch('next')}
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            </>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => {
              setShowSearch(false)
              setSearchQuery('')
            }}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Output area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto overflow-x-auto p-3 font-mono text-sm leading-5 text-gray-100"
      >
        <Ansi>{output}</Ansi>

        {/* Cursor for streaming */}
        {isStreaming && (
          <span className="inline-block w-2 h-4 bg-gray-300 animate-pulse ml-0.5" />
        )}
      </div>
    </div>
  )
}
