/**
 * ToolSuggestionBar Component (PRD-40: Dynamic Tool Suggestions)
 *
 * Container component that displays tool-specific suggestion chips.
 * Shows when a tool icon is clicked, providing contextual prompts for that tool.
 */

import * as React from 'react'
import { useEffect } from 'react'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { SuggestionChip } from './SuggestionChip'
import { Button } from '@/components/ui/button'

export interface ToolSuggestionBarProps {
  /** Array of suggestion text strings to display */
  suggestions: string[]
  /** Name of the active tool (e.g., "GMAIL", "SLACK") */
  activeTool: string | null
  /** Callback when a suggestion is clicked */
  onSuggestionClick: (suggestion: string) => void
  /** Callback when close button is clicked */
  onClose: () => void
  /** Optional loading state */
  isLoading?: boolean
  /** Optional className for custom styling */
  className?: string
}

export function ToolSuggestionBar({
  suggestions,
  activeTool,
  onSuggestionClick,
  onClose,
  isLoading = false,
  className,
}: ToolSuggestionBarProps) {
  // Keyboard accessibility: Escape key closes suggestions
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && activeTool) {
        e.preventDefault()
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [activeTool, onClose])

  // Don't render if no active tool or no suggestions
  if (!activeTool || suggestions.length === 0) {
    return null
  }

  return (
    <div
      className={cn(
        // Base layout - match welcome screen suggestions
        'flex items-center justify-center gap-2 flex-wrap',

        // Animation
        'animate-in fade-in slide-in-from-bottom-2 duration-200',

        className
      )}
    >
      {/* Suggestions container */}
      {isLoading ? (
        // Loading skeleton
        <>
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="h-9 w-32 rounded-full bg-muted/50 animate-pulse"
            />
          ))}
        </>
      ) : (
        // Suggestion chips
        suggestions.map((suggestion, index) => (
          <SuggestionChip
            key={`${suggestion}-${index}`}
            text={suggestion}
            onClick={() => onSuggestionClick(suggestion)}
          />
        ))
      )}

      {/* Close button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={onClose}
        className="h-8 w-8 shrink-0 hover:bg-destructive/10 hover:text-destructive"
        aria-label="Close suggestions"
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  )
}
