/**
 * SuggestionChip Component (PRD-40: Dynamic Tool Suggestions)
 *
 * Individual suggestion chip that displays a single prompt suggestion.
 * Used within ToolSuggestionBar to show clickable suggestion prompts.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'

export interface SuggestionChipProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  text: string
}

export function SuggestionChip({ text, className, onClick, ...props }: SuggestionChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        // Base styles - rounded, same size for all
        'flex items-center',
        'px-4 py-2',
        'text-sm font-normal',
        'rounded-full',
        'border border-warning/20',
        'bg-background/80',

        // Fixed width - all boxes same size
        'w-full',
        'h-10',

        // Text alignment - left
        'text-left',
        'justify-start',

        // Hover effects
        'hover:border-warning/40',
        'hover:bg-warning/5',

        // Active/pressed state
        'active:scale-95',

        // Transitions
        'transition-all duration-220',

        // Focus styles
        'focus-visible:outline-none',
        'focus-visible:ring-2',
        'focus-visible:ring-warning/50',

        // Disabled state
        'disabled:pointer-events-none',
        'disabled:opacity-50',

        // Text - truncate if needed
        'truncate',
        'cursor-pointer',

        className
      )}
      {...props}
    >
      {text}
    </button>
  )
}
