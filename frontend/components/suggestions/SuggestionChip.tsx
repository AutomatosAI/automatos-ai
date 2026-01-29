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
        // Base styles
        'inline-flex items-center justify-center',
        'px-3 py-1.5',
        'text-sm font-medium',
        'rounded-full',
        'border border-border/50',
        'bg-background/50 backdrop-blur',
        'supports-[backdrop-filter]:bg-background/40',

        // Hover effects
        'hover:border-primary/30',
        'hover:bg-primary/5',
        'hover:text-foreground',
        'hover:scale-105',

        // Active/pressed state
        'active:scale-95',

        // Transitions
        'transition-all duration-200',

        // Focus styles
        'focus-visible:outline-none',
        'focus-visible:ring-2',
        'focus-visible:ring-ring',
        'focus-visible:ring-offset-2',

        // Disabled state
        'disabled:pointer-events-none',
        'disabled:opacity-50',

        // Text
        'whitespace-nowrap',
        'truncate',
        'max-w-[280px]',

        className
      )}
      {...props}
    >
      {text}
    </button>
  )
}
