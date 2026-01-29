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
        // Base styles - match "Create an Agent" size
        'inline-flex items-center justify-center gap-2',
        'px-3 py-1.5',
        'text-xs font-normal',
        'rounded-md',
        'border border-border/50',
        'bg-background',

        // Size constraints - match bottom buttons
        'max-w-[160px]',
        'h-8',

        // Hover effects
        'hover:bg-accent/50',
        'hover:border-accent',

        // Active/pressed state
        'active:scale-95',

        // Transitions
        'transition-all duration-200',

        // Focus styles
        'focus-visible:outline-none',
        'focus-visible:ring-1',
        'focus-visible:ring-ring',

        // Disabled state
        'disabled:pointer-events-none',
        'disabled:opacity-50',

        // Text - truncate if too long
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
